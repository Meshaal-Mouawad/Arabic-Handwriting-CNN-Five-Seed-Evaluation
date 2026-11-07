"""Public MADBase/AHCD source acquisition with immutable provenance checks.

Run in an internet-enabled Colab or equivalent environment. No files are
fabricated, repaired or silently substituted when a source is unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import time
from urllib.parse import quote
import urllib.error
import time
from urllib.parse import quote
import zipfile
from pathlib import Path

SOURCES = {
 'digits': {
   'repository':'mloey/Arabic-Handwritten-Digits-Dataset',
   'url':'https://raw.githubusercontent.com/mloey/Arabic-Handwritten-Digits-Dataset/master/Arabic%20Handwritten%20Digits%20Dataset%20CSV.zip',
   'sha1_git_blob':'fdf29173103cb2f6073911ce9d70a379bc7f1b1d',
   'size':13911478, 'native_size':28,
 },
 'letters': {
   'repository':'mloey/Arabic-Handwritten-Characters-Dataset',
   'url':'https://raw.githubusercontent.com/mloey/Arabic-Handwritten-Characters-Dataset/master/Arabic%20Handwritten%20Characters%20Dataset%20CSV.zip',
   'sha1_git_blob':'6a5668c13561693176ef2a99d7a74fe6f22cf7a5',
   'size':3181704, 'native_size':32,
 },
}


def blob_digest(path):
    path=Path(path)
    h=hashlib.sha1()
    h.update(f'blob {path.stat().st_size}\0'.encode('ascii'))
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def download_checked(name,folder):
    """Download a source-pinned archive with alternate official GitHub routes.

    All successful routes must match the ORIGINAL expected file size and Git blob SHA1.
    Incomplete downloaded parts are removed, never confused with scientific data.
    """
    src=SOURCES[name]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'{name}_upstream_csv.zip'
    if dest.exists() and dest.stat().st_size==src['size'] and blob_digest(dest)==src['sha1_git_blob']:
        print(f'[{name}] Cached original source verified ({src["size"]:,} bytes)',flush=True)
        return dest
    if dest.exists():
        raise ValueError(f'Existing {dest} differs from registered upstream Git blob. Preserve or quarantine manually; do not overwrite evidence.')
    repo=src['repository']
    fname=('Arabic Handwritten Digits Dataset CSV.zip' if name=='digits'
           else 'Arabic Handwritten Characters Dataset CSV.zip')
    escaped=quote(fname)
    urls=[
        src['url'],
        f'https://media.githubusercontent.com/media/{repo}/master/{escaped}',
        f'https://github.com/{repo}/raw/refs/heads/master/{escaped}',
    ]
    errors=[]
    for u in urls:
        for attempt in (1,2):
            with tempfile.NamedTemporaryFile(prefix=name+'_',suffix='.part',dir=folder,delete=False) as temp:
                part=Path(temp.name)
            try:
                print(f'[{name}] Downloading official archive (route {urls.index(u)+1}/3, attempt {attempt}/2)...',flush=True)
                req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 Academic-Reproducibility-Experiment/1.1'})
                with urllib.request.urlopen(req,timeout=120) as inp,part.open('wb') as out:
                    shutil.copyfileobj(inp,out,1<<20)
                got_size=part.stat().st_size
                got_hash=blob_digest(part)
                if got_size!=src['size'] or got_hash!=src['sha1_git_blob']:
                    raise ValueError(f'{name} source archive did not match pinned Git object; downloaded {got_size} bytes, hash {got_hash}, expected {src["size"]} and {src["sha1_git_blob"]}. STOP: do not use altered data.')
                part.replace(dest)
                print(f'[{name}] Source verified: {got_size:,} bytes, Git blob SHA1={got_hash}',flush=True)
                return dest
            except ValueError:
                raise
            except (OSError,urllib.error.URLError) as exc:
                detail=f'{type(exc).__name__}: {exc}'
                errors.append((urls.index(u)+1,attempt,detail))
                print(f'[{name}] Route {urls.index(u)+1} attempt {attempt} failed: {detail}',flush=True)
                if attempt==1:time.sleep(2)
            finally:
                if part.exists():part.unlink()
    raise RuntimeError(f'All official download routes failed for {name}. Network/HTTP details: {errors}')


def extract_source(zippath, outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(zippath) as z:
        for inf in z.infolist():
            # allow CSV contents only; avoid traversal and symlink surprises.
            raw=Path(inf.filename)
            if inf.is_dir() or raw.suffix.lower()!='.csv':continue
            if '..' in raw.parts or raw.is_absolute() or len(raw.parts)>12:
                raise ValueError(f'Unsafe zip member {inf.filename}')
            if (inf.external_attr>>16)&0o170000==0o120000:
                raise ValueError(f'Unsafe symlink {inf.filename}')
            target=outdir/raw.name
            if target.exists():
                # distinguish names that collide but have different source paths
                raise ValueError(f'Duplicate CSV member name: {raw.name}')
            with z.open(inf) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
            records.append({'zip_member':inf.filename,'csv_file':str(target.resolve()),'bytes':target.stat().st_size,'sha256':sha256(target)})
    if len(records)!=4:
        raise ValueError(f'Expected exactly four train/test image/label CSVs in {zippath}, got {[r["zip_member"] for r in records]}')
    return records


def classify_four(files):
    mapping={}
    for f in files:
        norm=re.sub(r'[^a-z0-9]+','',Path(f['csv_file']).stem.lower())
        part='train' if 'train' in norm else 'test' if 'test' in norm else None
        kind='images' if 'image' in norm else 'labels' if 'label' in norm else None
        if part is None or kind is None:raise ValueError(f'Cannot identify role of CSV {f["zip_member"]}')
        key=f'{part}_{kind}'
        if key in mapping:raise ValueError(f'Duplicate role {key}: {mapping[key]}, {f}')
        mapping[key]=f['csv_file']
    if set(mapping)!={'train_images','train_labels','test_images','test_labels'}:
        raise ValueError(f'Missing source file roles: {mapping}')
    return mapping


def validate_cached_extraction(zippath, outdir):
    """Verify CSVs against immutable archive without trusting absent receipts."""
    outdir=Path(outdir)
    records=[]
    with zipfile.ZipFile(zippath) as z:
        for entry in z.infolist():
            raw=Path(entry.filename)
            if entry.is_dir() or raw.suffix.lower()!='.csv':continue
            if '..' in raw.parts or raw.is_absolute():raise ValueError(f'Unsafe archive member {entry.filename}')
            target=outdir/raw.name
            if not target.exists():raise FileNotFoundError(f'Partial archive extraction: {target}')
            # read in chunks; require byte-for-byte equivalence to the pinned archive
            h=hashlib.sha256();n=0
            with z.open(entry) as inp:
                for block in iter(lambda:inp.read(1<<20),b''):
                    h.update(block);n+=len(block)
            if n!=target.stat().st_size or h.hexdigest()!=sha256(target):
                raise ValueError(f'Cached extracted CSV differs from pinned upstream archive: {target}')
            records.append({'zip_member':entry.filename,'csv_file':str(target.resolve()),'bytes':n,'sha256':h.hexdigest()})
    classify_four(records)
    return records


def infer_original_background(csv_path,native_size):
    """Pre-specified first-256-image corner test; applies before any training."""
    import numpy as np
    arr=np.loadtxt(csv_path,delimiter=',',dtype=np.float32,max_rows=256)
    if arr.ndim!=2 or arr.shape[1]!=native_size*native_size:raise ValueError(f'Cannot infer raw background from {csv_path}')
    corners=arr[:,[0,native_size-1,native_size*(native_size-1),native_size*native_size-1]]
    mean=float(corners.mean()/255.)
    if mean<.20: decision='dark_background';invert=False
    elif mean>.80:decision='bright_background';invert=True
    else:raise ValueError(f'Ambiguous source background in {csv_path}: corner normalized mean={mean:.4f}. Explicit investigator polarity decision required.')
    return {'first_n_rows':len(arr),'normalized_corner_mean':mean,'decision':decision,
            'invert_to_dark_background':invert,
            'rule':'raw train-set corner pixels first 256 rows; mean <.20 is dark, >.80 is bright, otherwise stop'}


def prepare(root, code_dir, *, validation=0.15, seed=2026, transpose_letters=True):
    root=Path(root).resolve();code_dir=Path(code_dir).resolve()
    print("[prepare] Starting source acquisition and hash verification...",flush=True)
    original=root/'official_source_archives'
    manifest_path=root/'upstream_source_receipts.json'
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text())
        for family in SOURCES:
            fp=Path(saved[family]['archive'])
            if not fp.exists() or blob_digest(fp)!=SOURCES[family]['sha1_git_blob']:
                raise ValueError(f'Existing source receipt for {family} cannot be verified')
    receipts={};paths={}
    for family in SOURCES:
        print(f"[prepare] Processing {family}...",flush=True)
        archive=download_checked(family,original)
        extracted_dir=root/f'{family}_original_csv'
        if extracted_dir.exists():
            found=validate_cached_extraction(archive,extracted_dir)
        else:
            # Atomic directory commit; interrupted extraction can be safely retried.
            with tempfile.TemporaryDirectory(prefix=f'_{family}_extract_',dir=root) as tmp:
                found=extract_source(archive,Path(tmp))
                classify_four(found)
                Path(tmp).rename(extracted_dir)
            found=validate_cached_extraction(archive,extracted_dir)
        print(f"[prepare] {family}: validated extracted original CSVs",flush=True)
        paths[family]=classify_four(found)
        receipts[family]={**SOURCES[family],'archive':str(archive.resolve()),'sha256':sha256(archive),'extracted':found}
    manifest_path.write_text(json.dumps(receipts,indent=2))
    print("[prepare] Computing image polarity checks...",flush=True)
    polarity={family:infer_original_background(paths[family]['train_images'],SOURCES[family]['native_size'])
              for family in ('digits','letters')}
    polarity_path=root/'PRETRAINING_POLARITY_AUDIT.json'
    if polarity_path.exists() and json.loads(polarity_path.read_text())!=polarity:
        raise ValueError('Pretraining polarity protocol changed from prior run; new study folder required')
    polarity_path.write_text(json.dumps(polarity,indent=2))
    print('Pretraining background normalization:',{k:('INVERT' if v['invert_to_dark_background'] else 'RETAIN') for k,v in polarity.items()})
    dest=root/'prepared_canonical'
    source_manifest=dest/'source_and_split_manifest.json'
    if source_manifest.exists():
        print('Prepared dataset already exists; run standalone integrity audit before reusing.')
        return dest
    prep=[sys.executable,str(code_dir/'prepare_canonical_csv.py')]
    for family in ('digits','letters'):
        for part in ('train','test'):
            for kind in ('images','labels'):
                prep +=[f'--{family}-{part}-{kind}',paths[family][f'{part}_{kind}']]
    prep +=['--out',str(dest),'--validation-fraction',str(validation),'--seed',str(seed)]
    if transpose_letters:prep.append('--letters-transpose')
    for family,info in polarity.items():
        if info['invert_to_dark_background']:prep.append(f'--{family}-invert')
    print("[prepare] Converting official CSV rows into controlled train/val/test arrays...",flush=True)
    subprocess.run(prep,check=True,cwd=code_dir)
    return dest

def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--code-dir',default=str(Path(__file__).parent/'research_code'))
    ap.add_argument('--val-fraction',type=float,default=.15)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--no-letters-transpose',action='store_true')
    a=ap.parse_args()
    dest=prepare(a.root,a.code_dir,validation=a.val_fraction,seed=a.seed,transpose_letters=not a.no_letters_transpose)
    print('PREPARED:',dest)
if __name__=='__main__':cli()
