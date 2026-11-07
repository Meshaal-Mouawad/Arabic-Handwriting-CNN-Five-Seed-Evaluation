#!/usr/bin/env python3
"""Recalculate saved-prediction metrics independently; no training or GPU required.
Usage: python verification/verify_predictions.py --root . --out verification_output
Dependencies: numpy, scipy (stdlib csv is used for stored results).
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import t

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metrics(y, p):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
    if y.shape!=(13360,) or p.shape!=y.shape or min(y.min(),p.min())<0 or max(y.max(),p.max())>=38:
        raise ValueError('Expected 13,360 valid 38-class predictions.')
    cm=np.bincount(y*38+p,minlength=38*38).reshape(38,38)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(38,dtype=float),where=den>0)
    d=y<10
    return {'accuracy_all':float(np.mean(y==p)),
            'accuracy_digits':float(np.mean(y[d]==p[d])),
            'accuracy_letters':float(np.mean(y[~d]==p[~d])),
            'macro_f1_digits':float(f[:10].mean()),'macro_f1_letters':float(f[10:].mean()),
            'macro_f1_38':float(f.mean()),'worst_family_macro_f1':float(min(f[:10].mean(),f[10:].mean()))}

def interval(v):
    a=np.array(v,dtype=float);m=float(a.mean());s=float(a.std(ddof=1));e=float(t.ppf(.975,len(a)-1)*s/np.sqrt(len(a)))
    return {'mean':m,'sd':s,'ci_95':[m-e,m+e],'per_seed_differences':a.tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('verification_output'))
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((root/'record/EXPERIMENT_LOCK.json').read_text())
    code_bad=[]
    for path,h in lock['code_sha256'].items():
        if digest(root/'code'/path)!=h:code_bad.append(path)
    if code_bad:raise ValueError('Frozen-code checksum mismatch: '+str(code_bad))
    source=list(csv.DictReader((root/'results/full_210_metric_rows.csv').open()))
    reference={(r['model'],int(r['seed']),r['condition']):r for r in source}
    if len(reference)!=210:raise ValueError('Expected exactly 210 unique reference rows.')
    calculated=[];hash_checks=[];values={};max_error=0.;common_y=None
    for model in lock['configs']:
        for seed in lock['seeds']:
            folder=root/'record/runs'/f'{model}_seed{seed}'
            tr=json.loads((folder/'training_record.json').read_text());ev=json.loads((folder/'test_results.json').read_text())
            assert tr['config']==model and tr['seed']==seed and tr['epochs']==20
            assert len(tr['history'])==20 and tr['train_file_sha256']==lock['train_sha256'] and tr['val_file_sha256']==lock['validation_sha256']
            assert ev['test_sha256']==lock['test_sha256'] and ev['test_n']==13360
            assert digest(folder/'predictions.npz')==ev['predictions_sha256']
            assert digest(folder/'best.pt')==ev['checkpoint_sha256']
            best=max(tr['history'],key=lambda a:a['worst_family_macro_f1'])
            assert best['epoch']==tr['best_epoch']==ev['chosen_epoch']
            hash_checks.append({'run':folder.name,'prediction_sha256':digest(folder/'predictions.npz'),
                                'selected_checkpoint_sha256':digest(folder/'best.pt'),
                                'initial_checkpoint_sha256':digest(folder/'initial.pt'),
                                'selected_epoch':tr['best_epoch']})
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                y=z['y_true']
                if common_y is None:common_y=y.copy()
                else:assert np.array_equal(y,common_y)
                assert int((y<10).sum())==10000 and int((y>=10).sum())==3360
                for condition in ev['corruption_tests']:
                    m=metrics(y,z['pred_'+condition]);row={'model':model,'seed':seed,'condition':condition,**m}
                    r=reference[(model,seed,condition)]
                    for k,v in m.items():
                        err=abs(v-float(r[k]));max_error=max(max_error,err)
                        if err>1e-12:raise ValueError(f'Metric mismatch {model}/{seed}/{condition}/{k}: {err}')
                        if abs(v-ev['corruption_tests'][condition][k])>1e-12:raise ValueError('JSON metric mismatch')
                    calculated.append(row);values[(model,seed,condition)]=m['worst_family_macro_f1']
    stats=json.loads((root/'results/PAIRED_HELDOUT_STATISTICS.json').read_text())
    contrasts={}
    conditions=['noise_0.05','noise_0.1','blur_0.8','blur_1.3','contrast_0.75','contrast_0.5']
    for name,data in stats['contrasts'].items():
        method,control=data['method'],data['control']
        v=[values[(method,s,'none_0.0')]-values[(control,s,'none_0.0')] for s in lock['seeds']]
        est=interval(v);ref=data['clean_worst_family_F1_difference']
        for k in ['mean','sd','ci_95','per_seed_differences']:
            if not np.allclose(est[k],ref[k],atol=1e-12,rtol=0):raise ValueError('Paired contrast mismatch '+name)
        contrasts[name]=est
    v=[np.mean([(values[('P_hier_smooth_worst',s,c)]-values[('P_hier_smooth_worst',s,'none_0.0')])-
                (values[('B3_hier_balanced',s,c)]-values[('B3_hier_balanced',s,'none_0.0')]) for c in conditions]) for s in lock['seeds']]
    h3=interval(v);ref=stats['contrasts']['H3_smooth_worst']['retention_differences']['H3_prespecified_mean_across_6_corruptions']
    for k in ['mean','sd','ci_95','per_seed_differences']:
        if not np.allclose(h3[k],ref[k],atol=1e-12,rtol=0):raise ValueError('H3 retention mismatch')
    with (out/'recalculated_210_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(calculated[0]));w.writeheader();w.writerows(calculated)
    report={'status':'PASS','evaluations_recalculated':len(calculated),'models':len(hash_checks),
            'frozen_code_files_matched':len(lock['code_sha256']),'max_absolute_metric_difference':max_error,
            'training_epochs_checked':600,'checkpoint_selection_rule_checks':30,
            'prediction_and_selected_checkpoint_hashes':'30/30 each matched recorded hashes',
            'initial_checkpoint_hashes':'30 recorded for package integrity; no original hash claim',
            'clean_paired_contrasts':contrasts,'H3_equal_weight_retention_contrast':h3,
            'execution_scope':'saved predictions, frozen code identity and record consistency; no new inference or retraining',
            'run_hashes':hash_checks}
    (out/'verification_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items()if k not in ['run_hashes','clean_paired_contrasts']},indent=2))
if __name__=='__main__':main()
