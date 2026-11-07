"""Validate source-derived NPZ partitions without estimating model performance."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from benchmark import load_partition, sha256


def hash_row(a:np.ndarray)->str:
    q=np.asarray(np.rint(a*255),dtype=np.uint8)
    return hashlib.blake2b(q.tobytes(),digest_size=16).hexdigest()


def audit(train:str,val:str,test:str,manifest:str|None=None):
    parts={}
    for key,fn in [('train',train),('validation',val),('test',test)]:
        x,y=load_partition(fn)
        ids=[hash_row(row) for row in x]
        if set(np.unique(y))!=set(range(38)):
            raise ValueError(f'{key}: expected all 38 classes; got {np.unique(y)}')
        hashes={}
        for idx,digest in enumerate(ids):hashes.setdefault(digest,[]).append(idx)
        conflicting=[digest for digest,ind in hashes.items() if len(set(y[ind]))>1]
        if conflicting:raise ValueError(f'{key}: same processed image maps to conflicting class labels ({len(conflicting)} image hashes)')
        parts[key]={'file':fn,'file_sha256':sha256(fn),'records':len(y),
                    'class_counts':{str(z):int((y==z).sum()) for z in range(38)},
                    'digit_count':int((y<10).sum()),'letter_count':int((y>=10).sum()),
                    'duplicate_identical_images_within_partition':len(ids)-len(hashes),
                    'hashes':set(hashes)}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        cross=parts[a]['hashes']&parts[b]['hashes']
        if cross:raise ValueError(f'FAIL cross-partition duplicate images: {a} vs {b} = {len(cross)}')
    report={key:{k:v for k,v in p.items() if k!='hashes'} for key,p in parts.items()}
    report['cross_partition_exact_image_overlap']='PASS: zero pairs'
    if manifest:
        info=json.loads(Path(manifest).read_text())
        for key,alias in [('train','train'),('validation','val'),('test','untouched_test')]:
            if info['partitions'][alias]['sha256']!=parts[key]['file_sha256']:
                raise ValueError(f'{key} SHA256 does not match data provenance manifest')
        report['source_manifest_sha256']=sha256(manifest)
        report['source_manifest_integrity']='PASS'
    return report


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True)
    p.add_argument('--val',required=True)
    p.add_argument('--test',required=True)
    p.add_argument('--manifest')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    result=audit(a.train,a.val,a.test,a.manifest)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print('PASS integrity gate: all 38 labels represented; no cross-partition identical images')
if __name__=='__main__':cli()
