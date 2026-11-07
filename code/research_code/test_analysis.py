import numpy as np
from analyze_predictions import worst_macro_f1,stratified_indices,paired_effect
from sklearn.metrics import f1_score

def main():
    rng=np.random.default_rng(5)
    y=np.repeat(np.arange(38),5)
    p=y.copy();p[[0,50,100,140,150]]=np.array([10,0,12,1,2])
    w=min(f1_score(y,p,labels=list(range(10)),average='macro',zero_division=0),
          f1_score(y,p,labels=list(range(10,38)),average='macro',zero_division=0))
    assert abs(worst_macro_f1(y,p)-w)<1e-12
    idx=stratified_indices(y,rng)
    assert all((y[idx]==i).sum()==5 for i in range(38))
    base={'pred_none_0.0':y.copy(),'pred_noise_0.1':p.copy()}
    other={'pred_none_0.0':p.copy(),'pred_noise_0.1':p.copy()}
    effect,ret=paired_effect(other,base,y,'noise_0.1')
    assert abs(effect)<1e-12 and ret>0
    print('PASS: macro-F1 matches sklearn; class-stratified bootstrap preserves counts; robustness retention contrast computed')
if __name__=='__main__':main()
