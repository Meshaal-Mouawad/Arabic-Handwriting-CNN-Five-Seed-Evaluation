"""Offline tests for the public-source acquisition and locked experiment controller."""
from __future__ import annotations
import hashlib,io,json,os,sys,tempfile,zipfile
from pathlib import Path
import numpy as np

from dataset_acquisition import blob_digest,extract_source,classify_four,download_checked,SOURCES,infer_original_background
from run_real_experiment import CONFIGS,SEEDS,completed_training,evaluate
from research_code.models import CONFIGS as MODEL_CONFIGS


def make_mini_zip(folder, unsafe=False):
    fp=folder/'source.zip'
    names=('csvTrainImages 3x4.csv','csvTrainLabel 3x1.csv','csvTestImages 2x4.csv','csvTestLabel 2x1.csv')
    with zipfile.ZipFile(fp,'w') as z:
        for name in names:z.writestr(name,'0,1,2,3\n')
        if unsafe:z.writestr('../../evil.csv','attack')
    return fp


def test_archive_integrity_and_roles():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        z=make_mini_zip(d)
        records=extract_source(z,d/'unpacked')
        roles=classify_four(records)
        assert sorted(roles)==['test_images','test_labels','train_images','train_labels']
        known='a93d55e5e7d6b1f6ab581cf1b7c5da22a6540ee2'
        # Independently recompute the documented git blob convention.
        data=b'hello world\n';p=d/'known.txt';p.write_bytes(data)
        ref=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        assert blob_digest(p)==ref
        print('PASS archive role inference and git blob verification')

def test_unsafe_member():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);z=make_mini_zip(d,unsafe=True)
        try:extract_source(z,d/'bad')
        except ValueError as e:assert 'Unsafe zip member' in str(e)
        else:raise AssertionError('Traversal not rejected')
        assert not (d/'evil.csv').exists()
        print('PASS dangerous archive member rejected')

def test_original_background_polarity():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        for name,val,expected in [('dark',0,False),('bright',255,True)]:
            fp=d/f'{name}.csv'
            np.savetxt(fp,np.full((8,16),val,dtype=np.uint8),delimiter=',',fmt='%d')
            verdict=infer_original_background(fp,4)
            assert verdict['invert_to_dark_background'] is expected
        fp=d/'ambiguous.csv';np.savetxt(fp,np.full((8,16),120,dtype=np.uint8),delimiter=',',fmt='%d')
        try:infer_original_background(fp,4)
        except ValueError as e: assert 'Ambiguous' in str(e)
        else:raise AssertionError('Ambiguous source intensity should be an explicit review gate')
    print('PASS source-level foreground/background harmonization audit')

def test_registration_pins_and_missing_full_matrix():
    assert len(CONFIGS)==6 and len(SEEDS)==5
    assert set(CONFIGS)==set(MODEL_CONFIGS)
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        plan={'epochs':20,'train_sha256':'x','validation_sha256':'y','test_sha256':'z'}
        assert not completed_training(d,plan,CONFIGS[0],SEEDS[0])
        class Dummy:
            max_new_eval_runs=30
        try:evaluate(d,d,plan,Dummy())
        except ValueError as e:assert 'LOCKED' in str(e)
        else:raise AssertionError('Unsealed evaluation without training')
        print('PASS test set sealed until all 30 official training records exist')

if __name__=='__main__':
    for f in (test_archive_integrity_and_roles,test_unsafe_member,test_original_background_polarity,test_registration_pins_and_missing_full_matrix):f()
    print('PASS Stage 4 orchestration tests: 4/4')
