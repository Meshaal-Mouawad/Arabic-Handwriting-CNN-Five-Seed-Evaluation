"""Build sealed MADBase/AHCD train/validation/test arrays from *local* CSV distributions.

Dataset downloads are NOT bundled or silently selected. User supplies source files,
chooses all potentially ambiguous geometry and polarity conversions, and records origin.
Original test rows are never used for training/validation selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def load_csv(path:Path,dtype,has_header:bool):
    """Handle canonical CSVs. Reject nonnumeric/corrupted cells."""
    arr=np.loadtxt(path,delimiter=",",dtype=dtype,skiprows=int(has_header))
    return arr


def read_family(prefix:str,native_size:int,base:int,expected_train:int,expected_test:int,args):
    record={}
    sources={}
    outputs={}
    for part in ("train","test"):
        px=Path(getattr(args,f"{prefix}_{part}_images")); lab=Path(getattr(args,f"{prefix}_{part}_labels"))
        if not px.exists() or not lab.exists(): raise FileNotFoundError(f"Missing {px} or {lab}")
        x=load_csv(px,np.float32,args.has_header)
        y=load_csv(lab,np.int64,args.has_header).reshape(-1)
        if x.ndim!=2 or x.shape[1]!=native_size*native_size:
            raise ValueError(f"{px} expected N rows x {native_size**2} columns, got {x.shape}")
        if len(x)!=len(y):raise ValueError(f"{prefix} {part}: pixel rows/label mismatch")
        expected=expected_train if part=="train" else expected_test
        if len(x)!=expected:
            raise ValueError(f"{prefix} {part}: got {len(x)} rows; canonical expectation {expected}. Wrong mirror/version? Quarantine for source inspection.")
        if not np.isfinite(x).all() or x.min()<0 or x.max()>255:
            raise ValueError(f"{prefix} {part}: nonfinite/out-of-range pixel(s)")
        if not np.allclose(x,np.round(x)):raise ValueError("Pixel CSV contains noninteger intensities")
        if not np.array_equal(np.unique(y),np.arange(base,base+(10 if prefix=='digits' else 28))):
            raise ValueError(f"Unexpected label map in {lab}: unique={np.unique(y).tolist()}")
        if prefix=="letters":
            # Explicit choice: originals are commonly 1..28; map to global 10..37.
            y=y-base+10
        else:
            y=y-base
        x=x.reshape(len(x),native_size,native_size).astype(np.uint8)
        if prefix=="letters" and args.letters_transpose:
            x=x.transpose(0,2,1)
        if getattr(args,f"{prefix}_invert"):
            x=255-x
        # Bilinear resize retained uniformly for every model; zero augmentation.
        resized=np.empty((len(x),64,64),dtype=np.uint8)
        for i,arr in enumerate(x):
            resized[i]=np.asarray(Image.fromarray(arr).resize((64,64),resample=Image.Resampling.BILINEAR))
        outputs[part]=(resized,y)
        sources[part]={"pixel_csv":str(px.resolve()),"pixel_sha256":sha256(px),
                       "label_csv":str(lab.resolve()),"label_sha256":sha256(lab),
                       "n":len(x),"native_size":native_size,"expected_label_base":base,
                       "label_counts":{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}}
    return outputs,sources


def fingerprints(arr):
    return [hashlib.blake2b(z.tobytes(),digest_size=16).hexdigest() for z in arr]



def quarantine_training_test_collisions(training_images, training_labels, test_images, test_labels):
    """Exclude SOURCE TRAIN rows with a pixel-identical processed TEST image.

    Source test remains immutable. This check precedes train/validation selection,
    so neither training nor model selection sees a held-out image. The choice is
    source-independent, uses no test labels, and is auditable by SHA fingerprints.
    """
    test_digests=set(fingerprints(test_images))
    train_digests=fingerprints(training_images)
    rejected=np.fromiter((d in test_digests for d in train_digests),dtype=bool,count=len(train_digests))
    bad=np.flatnonzero(rejected)
    report={
      'policy':'retain every original source-test row; exclude all matching source-training rows BEFORE train/validation split',
      'fingerprint_basis':'BLAKE2b-128 of post-transpose, post-polarity, bilinear-resized 64x64 uint8 pixels',
      'excluded_source_training_rows':int(len(bad)),
      'excluded_unique_fingerprints':len(set(train_digests[i] for i in bad)),
      'excluded_training_rows_by_class':{str(k):int(v) for k,v in zip(*np.unique(training_labels[bad],return_counts=True))},
      'excluded_source_training_indices':bad.astype(int).tolist(),
      'test_rows_preserved':int(len(test_images)),
      'test_labels_not_used_for_filtering':True,
      'heldout_test_duplicate_rows':int(len(test_images)-len(test_digests)),
    }
    retained=np.flatnonzero(~rejected)
    if not len(retained):raise ValueError('All source train rows overlap the test set')
    return training_images[retained],training_labels[retained],retained,report


def duplicate_safe_stratified_split(images, labels, fraction, seed):
    """Assign identical processed images to one split, preserving class proportions.

    The split is decided from SOURCE TRAIN images only. A hash shared by different
    labels is a contradictory annotation and must be resolved before training.
    """
    digests=fingerprints(images)
    buckets={}
    for index, (digest, label) in enumerate(zip(digests, labels)):
        entry=buckets.setdefault(digest, [int(label), []])
        if entry[0]!=int(label):
            raise ValueError(f'Identical processed image has conflicting labels {entry[0]} and {label}; image fingerprint={digest}')
        entry[1].append(index)
    rng=np.random.default_rng(seed)
    val=[]
    for cls in np.unique(labels):
        groups=[idx for label,idx in buckets.values() if label==int(cls)]
        if len(groups)<2:
            raise ValueError(f'Class {cls} has fewer than two unique images; cannot form leakage-free validation split')
        rng.shuffle(groups)
        total_class=sum(len(g) for g in groups)
        target=round(total_class*fraction)
        total=0
        for group in groups:
            # Prefer target without splitting identical-image groups; retain
            # at least one group for training even in very small classes.
            if abs(total+len(group)-target) <= abs(total-target) and total+len(group)<total_class:
                val.extend(group)
                total+=len(group)
        if total==0:val.extend(groups[0])
    mask=np.zeros(len(labels),dtype=bool)
    mask[np.asarray(val,dtype=np.int64)]=True
    if not mask.any() or mask.all():raise ValueError('Duplicate-aware validation split is empty or exhaustive')
    return np.flatnonzero(~mask),np.flatnonzero(mask),{
        'unique_source_training_images':len(buckets),
        'source_training_duplicate_rows':len(labels)-len(buckets),
        'validation_rows':int(mask.sum()),
        'train_rows':int((~mask).sum()),
        'grouping':'exact resized uint8 image fingerprint, same-label only',
    }


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d,dr=read_family('digits',28,args.digit_label_base,60000,10000,args)
    l,lr=read_family('letters',32,args.letter_label_base,13440,3360,args)
    # Preserve the original source test set. Remove post-preprocessing pixel collisions
    # exclusively from SOURCE TRAIN before selecting the validation partition.
    # This prevents train/test and validation/test leakage without touching test data.
    source_x=np.concatenate((d['train'][0],l['train'][0]))
    source_y=np.concatenate((d['train'][1],l['train'][1]))
    source_origin=np.concatenate((np.full(len(d['train'][1]),0),np.full(len(l['train'][1]),1)))
    x_test=np.concatenate((d['test'][0],l['test'][0]))
    y_test=np.concatenate((d['test'][1],l['test'][1]))
    orig_x,orig_y,retained_indices,test_collision_audit=quarantine_training_test_collisions(
        source_x,source_y,x_test,y_test)
    org=source_origin[retained_indices]
    split_note='exact-image-grouped class-stratified; no verified writer IDs supplied'
    duplicate_group_audit=None
    if args.writer_groups:
        supplied_groups=np.loadtxt(args.writer_groups,delimiter=',',dtype=str)
        if supplied_groups.shape!=(len(source_y),):raise ValueError('Writer group manifest must contain one ID per original source training row, digits first, then letters')
        groups=supplied_groups[retained_indices]
        groups=np.char.add(np.where(org==0,'D:','L:'),groups)
        splitter=GroupShuffleSplit(n_splits=1,test_size=args.validation_fraction,random_state=args.seed)
        itr,iv=next(splitter.split(orig_x,orig_y,groups))
        if set(groups[itr])&set(groups[iv]):raise AssertionError('Writer leakage')
        split_note='writer-disjoint group shuffle split, supplied manifest SHA256 '+sha256(args.writer_groups)
        # Writer splits also need duplicate-image separation across train/validation.
        if set(fingerprints(orig_x[itr])) & set(fingerprints(orig_x[iv])):
            raise ValueError('Writer-disjoint split includes duplicate image(s) across training/validation; inspect writer IDs')
    else:
        itr,iv,duplicate_group_audit=duplicate_safe_stratified_split(orig_x,orig_y,args.validation_fraction,args.seed)
    ytr,yv=orig_y[itr],orig_y[iv]
    for family,(lo,hi) in (('digits',(0,10)),('letters',(10,38))):
        if not ((ytr>=lo)&(ytr<hi)).any() or not ((yv>=lo)&(yv<hi)).any():
            raise ValueError('Validation split lacks '+family)
    partitions={'train':(orig_x[itr],ytr),'val':(orig_x[iv],yv),'untouched_test':(x_test,y_test)}
    # Reject any *exact* image overlap between designated partitions, regardless of label.
    fps={key:set(fingerprints(x)) for key,(x,y) in partitions.items()}
    overlap={}
    for a,b in (('train','val'),('train','untouched_test'),('val','untouched_test')):
        num=len(fps[a]&fps[b]);overlap[f'{a}/{b}']=num
        if num:raise ValueError(f'{num} exact same processed image(s) across {a} and {b}; investigate potential leakage before using outputs')
    test_writer_groups=None
    if args.test_writer_groups:
        test_writer_groups=np.loadtxt(args.test_writer_groups,delimiter=',',dtype=str).reshape(-1)
        if test_writer_groups.shape!=(len(y_test),):
            raise ValueError('Expected 13,360 source test writer IDs (digit rows first, letter rows second)')
        test_writer_groups=np.char.add(np.where(y_test<10,'D:','L:'),test_writer_groups)
    outputs={}
    for part,(images,labels) in partitions.items():
        fn=out/f'{part}.npz'
        data={'images':images[:,None,:,:].astype(np.float32)/255.,'labels':labels}
        if part=='untouched_test' and test_writer_groups is not None:
            data['writer_groups']=test_writer_groups
        np.savez_compressed(fn,**data)
        outputs[part]={"file":fn.name,"sha256":sha256(fn),"n":int(len(labels)),
             "label_counts":{str(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))}}
    # Preview is mandatory before approving orientation/polarity for use in results.
    from PIL import ImageDraw
    canvas=Image.new('RGB',(8*86,5*104),'white')
    draw=ImageDraw.Draw(canvas)
    rng=np.random.default_rng(args.seed+99)
    for row in range(5):
        family=0 if row<2 else 1
        candidates=np.flatnonzero((orig_y<10)==(family==0))
        selected=rng.choice(candidates,8,replace=False)
        for j,idx in enumerate(selected):
            box=Image.fromarray(orig_x[idx]).resize((76,76))
            canvas.paste(box.convert('RGB'),(j*86,row*104))
            draw.text((j*86,row*104+76),f'y={orig_y[idx]}',fill='black')
    preview=out/'preview_orientation_and_labels.png';canvas.save(preview)
    report={"source_provenance":{"digits":dr,"letters":lr},
            "transforms":{"pixels_input_range":[0,255],"pixels_output_range":[0,1],
             "resize":"PIL bilinear 28/32 -> 64, fixed", "letters_transpose":args.letters_transpose,
             "digits_invert":args.digits_invert,"letters_invert":args.letters_invert,
             "labels":"digits 0..9, letters 10..37 (original letter labels 1..28 by default)",
             "image_augmentation":"none"},
            "validation":{"fraction_requested":args.validation_fraction,"seed":args.seed,
              "selection":split_note,"duplicate_group_audit":duplicate_group_audit,"test_collision_quarantine":test_collision_audit,"file_overlap_count":overlap,
              "writer_generalization":"uncertified unless writer groups supplied; official AHCD test split is writer-exclusive by source description",
              "test_writer_group_manifest_sha256":sha256(args.test_writer_groups) if args.test_writer_groups else None},
            "partitions":outputs,"visual_approval_required":"Check preview_orientation_and_labels.png against the original Arabic symbol legend before training",
            "research_usage":"Original test files are processed but are not loaded during model training or checkpoint selection"}
    (out/'source_and_split_manifest.json').write_text(json.dumps(report,indent=2))
    print('EXCLUDED',test_collision_audit['excluded_source_training_rows'],'source-training rows identical to original test images (original test retained).')
    print('Wrote',out,'\nREVIEW preview_orientation_and_labels.png before approving preprocessing.')


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    for prefix in ('digits','letters'):
        for split in ('train','test'):
            for fld in ('images','labels'):
                p.add_argument(f'--{prefix}-{split}-{fld}',dest=f'{prefix}_{split}_{fld}',required=True)
    p.add_argument('--digit-label-base',type=int,choices=[0,1],default=0)
    p.add_argument('--letter-label-base',type=int,choices=[0,1],default=1)
    p.add_argument('--letters-transpose',action='store_true')
    p.add_argument('--digits-invert',action='store_true')
    p.add_argument('--letters-invert',action='store_true')
    p.add_argument('--has-header',action='store_true')
    p.add_argument('--writer-groups',default=None,help='Optional training-row writer CSV, digits first then letters')
    p.add_argument('--test-writer-groups',default=None,help='Optional test writer CSV for cluster bootstrap, digit rows then letters')
    p.add_argument('--validation-fraction',type=float,default=0.15)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if not 0.05<=args.validation_fraction<=0.25:raise ValueError('Validation fraction expected 0.05..0.25')
    run(args)
if __name__=='__main__':cli()
