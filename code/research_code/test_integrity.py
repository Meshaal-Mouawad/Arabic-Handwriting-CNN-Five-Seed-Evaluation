import tempfile
from pathlib import Path
import numpy as np
from audit_partitions import audit

def make(path,offset):
    y=np.repeat(np.arange(38),2).astype('int64')
    x=np.zeros((len(y),1,64,64),dtype='float32')
    for i in range(len(y)):
        # Ensure unique per sample and across splits after quantized u8 hashing.
        x[i,0,i%64,(i*3)%64]=float((offset+i+1)%240+10)/255
        x[i,0,(i+11)%64,(i*5+12)%64]=1.
    np.savez_compressed(path,images=x,labels=y)


def test_audit():
    with tempfile.TemporaryDirectory() as td:
        tr=Path(td)/'tr.npz';val=Path(td)/'val.npz';test=Path(td)/'test.npz'
        make(tr,11);make(val,51);make(test,101)
        result=audit(str(tr),str(val),str(test))
        assert result['train']['records']==76
        assert result['cross_partition_exact_image_overlap']=='PASS: zero pairs'
        x=np.load(val)['images'];y=np.load(val)['labels'];x[0]=np.load(tr)['images'][0]
        np.savez_compressed(val,images=x,labels=y)
        try:audit(str(tr),str(val),str(test))
        except ValueError as e:assert 'duplicate images' in str(e)
        else:raise AssertionError('Cross-split leakage was not rejected')
    print('PASS: integrity preflight accepts disjoint partitions and rejects cross-partition exact-image leakage')
if __name__=='__main__':test_audit()
