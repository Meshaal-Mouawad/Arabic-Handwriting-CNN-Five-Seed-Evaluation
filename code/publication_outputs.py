"""Produce scientific results tables/figures from COMPLETE real-image test evidence.

No output is produced when fewer than 30 real runs with 7 held-out
conditions each are available. This prevents accidental placeholder results.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_real_experiment import CONFIGS,SEEDS
CONDITIONS=('none_0.0','noise_0.05','noise_0.1','blur_0.8','blur_1.3','contrast_0.75','contrast_0.5')
CAPTIONS={'none_0.0':'Clean', 'noise_0.05':'Noise .05','noise_0.1':'Noise .10','blur_0.8':'Blur .8','blur_1.3':'Blur 1.3','contrast_0.75':'Contrast .75','contrast_0.5':'Contrast .50'}
LABELS={'B0_flat_ce':'Flat CE','B1_flat_balanced':'Flat balanced',
        'B2_hier_ce':'Hierarchical CE','B3_hier_balanced':'Hierarchical balanced',
        'P_hier_smooth_worst':'Hierarchical smooth-worst','C_flat_matched':'Capacity-matched flat'}

def mean_ci(x):
    vals=np.asarray(x,dtype=float)
    m=float(vals.mean());s=float(vals.std(ddof=1))
    rad=float(tdist.ppf(.975,len(vals)-1)*s/np.sqrt(len(vals)))
    return m,s,m-rad,m+rad

def build(root):
    root=Path(root)
    stats=root/'PAIRED_HELDOUT_STATISTICS.json'
    if not stats.exists():
        raise FileNotFoundError('Publication blocked: real paired held-out predictions must be complete before exporting scientific figures')
    out=root/'publication_outputs'
    rows=[]
    expected=set(CONDITIONS)
    for config in CONFIGS:
        for seed in SEEDS:
            folder=root/'runs'/f'{config}_seed{seed}'
            t=folder/'test_results.json';p=folder/'predictions.npz';h=folder/'training_record.json'
            if not t.exists() or not p.exists() or not h.exists():
                raise FileNotFoundError(f'Publication blocked: missing real-model evidence for {config}, seed={seed}')
            trial=json.loads(t.read_text())
            if set(trial['corruption_tests'])!=expected:raise ValueError(f'Incomplete evaluation conditions in {t}')
            if trial['seed']!=seed or trial['config']!=config:raise ValueError('Metadata run mismatch')
            for c in CONDITIONS:
                metrics=trial['corruption_tests'][c]
                rows.append({'model':config,'seed':seed,'condition':c,
                    'accuracy_all':metrics['accuracy_all'],'accuracy_digits':metrics['accuracy_digits'],
                    'accuracy_letters':metrics['accuracy_letters'],'macro_f1_digits':metrics['macro_f1_digits'],
                    'macro_f1_letters':metrics['macro_f1_letters'],
                    'macro_f1_38':metrics['macro_f1_38'],
                    'worst_family_macro_f1':metrics['worst_family_macro_f1']})
    assert len(rows)==210
    out.mkdir(parents=True,exist_ok=True)
    with (out/'full_210_metric_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    clean=[]
    for config in CONFIGS:
        r=[r for r in rows if r['model']==config and r['condition']=='none_0.0']
        cur={'model':config,'label':LABELS[config]}
        for key in ['accuracy_all','accuracy_digits','accuracy_letters','macro_f1_digits','macro_f1_letters','macro_f1_38','worst_family_macro_f1']:
            m,s,lo,hi=mean_ci([r0[key] for r0 in r]);cur[key+'_mean']=m;cur[key+'_sd']=s;cur[key+'_ci_lo']=lo;cur[key+'_ci_hi']=hi
        clean.append(cur)
    with (out/'clean_model_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(clean[0]));w.writeheader();w.writerows(clean)
    # Honest latex: manuscript authors must refer to this as actual held-out test results.
    text=['% Generated exclusively from verified Stage4 30-run held-out results.',
          r'\begin{table}[t]',r'\centering',r'\caption{Held-out clean-image results over five seeds. Values are mean $\pm$ standard deviation.}',
          r'\label{tab:real_results}',r'\begin{tabular}{lrrr}',r'\toprule',
          r'Method & Overall Acc. & Digit macro-F1 & Letter macro-F1 \\',r'\midrule']
    for cur in clean:
        cells=[f'{cur[key+"_mean"]:.3f} $\\pm$ {cur[key+"_sd"]:.3f}' for key in ['accuracy_all','macro_f1_digits','macro_f1_letters']]
        text.append(cur['label'].replace('-','--')+' & '+' & '.join(cells)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (out/'paper_clean_results_table.tex').write_text('\n'.join(text)+'\n')
    # Plot A: family-aware clean scores with explicit uncertainty intervals.
    xs=np.arange(len(clean));fig,ax=plt.subplots(figsize=(9,4.8))
    metric='worst_family_macro_f1'
    vals=np.array([cur[metric+'_mean'] for cur in clean])
    errlo=vals-np.array([cur[metric+'_ci_lo'] for cur in clean]);errhi=np.array([cur[metric+'_ci_hi'] for cur in clean])-vals
    ax.errorbar(xs,vals,yerr=[errlo,errhi],fmt='o',capsize=3)
    ax.set_xticks(xs,list(LABELS.values()),rotation=27,ha='right')
    ax.set_ylabel('Worst-family macro-F1 (held-out test)')
    ax.set_title('Clean-image performance: five-seed mean and descriptive 95% interval')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'fig_heldout_clean_worst_family_f1.pdf');fig.savefig(out/'fig_heldout_clean_worst_family_f1.png',dpi=220);plt.close(fig)
    # Plot B: robustness across unchanged corruptions.
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    for config in CONFIGS:
        avg=[np.mean([r['worst_family_macro_f1'] for r in rows if r['model']==config and r['condition']==c]) for c in CONDITIONS]
        ax.plot(np.arange(len(CONDITIONS)),avg,marker='o',label=LABELS[config])
    ax.set_xticks(np.arange(len(CONDITIONS)),[CAPTIONS[c] for c in CONDITIONS],rotation=24,ha='right')
    ax.set_ylabel('Worst-family macro-F1')
    ax.set_title('Held-out accuracy under fixed image corruptions; mean across five seeds')
    ax.grid(axis='y',alpha=.25);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(out/'fig_robustness_six_conditions.pdf');fig.savefig(out/'fig_robustness_six_conditions.png',dpi=220);plt.close(fig)
    d=json.loads(stats.read_text())
    contrasts=d['contrasts'];md=['# Empirical findings generated from the full real-data experiment','',
       f'Completed held-out evaluation: {len(CONFIGS)} model conditions × {len(SEEDS)} seeds × {len(CONDITIONS)} test conditions = {len(rows)} metric rows.','',
       '| Prespecified comparison | Mean clean paired Δ worst-family macro-F1 | 95% descriptive across-seed interval |',
       '|---|---:|---|']
    for name,item in contrasts.items():
        val=item['clean_worst_family_F1_difference'];lo,hi=val['ci_95']
        md.append(f'| {name} | {val["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    md+=['','For H3, the pre-specified mean corruption-retention contrast is in the paired-statistics JSON.','',
          'The reported intervals quantify between-seed variability; image-bootstrap results are separately reported.']
    (out/'real_results_for_editor.md').write_text('\n'.join(md)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    out=build(a.root);print('REAL DATA PUBLICATION ASSETS:',out)
