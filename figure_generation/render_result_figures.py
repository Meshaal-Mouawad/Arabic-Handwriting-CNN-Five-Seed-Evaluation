"""Render manuscript Figures 2–3 without altering any experimental values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results'
OUT = ROOT / 'regenerated_figures'
OUT.mkdir(exist_ok=True)
stats=json.loads((DATA/'PAIRED_HELDOUT_STATISTICS.json').read_text())
records=pd.read_csv(DATA/'full_210_metric_rows.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'ps.fonttype':42,'axes.labelsize':10,'xtick.labelsize':9,
                     'ytick.labelsize':10,'savefig.dpi':250})
rows=[('H1_flat_balance', 'Family balancing: B1 − B0'),
      ('H2_hierarchy_CE', 'Hierarchy, pooled: B2 − B0'),
      ('H2_hierarchy_balanced', 'Hierarchy, balanced: B3 − B1'),
      ('H3_smooth_worst', 'Clean diagnostic: P − B3'),
      ('capacity_H2_vs_matched_flat', 'Parameter-count control: B2 − C')]
fig,ax=plt.subplots(figsize=(6.3,3.55))
means=[]; intervals=[]
for key,label in rows:
    v=stats['contrasts'][key]['clean_worst_family_F1_difference']
    means.append(v['mean']*100);intervals.append(np.array(v['ci_95'])*100)
means=np.array(means);intervals=np.array(intervals)
y=np.arange(5)
ax.errorbar(means,y,xerr=np.vstack([means-intervals[:,0],intervals[:,1]-means]),
            fmt='o',capsize=4,elinewidth=1.5,markersize=5)
ax.axvline(0,linestyle='--',linewidth=1)
ax.set_yticks(y, [x[1] for x in rows]);ax.invert_yaxis()
ax.set_xlabel('Paired change in worst-family macro-F1\n(percentage points; first model minus second)')
ax.set_xlim(-1.55,1.65)
ax.set_ylim(4.5,-.5)
ax.grid(axis='x',alpha=.25)
ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(left=.48,right=.98,bottom=.25,top=.97)
fig.savefig(OUT/'paired_effects.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(OUT/'paired_effects.png',bbox_inches='tight',pad_inches=.03)
plt.close(fig)

models=[('B0_flat_ce','B0 Flat / pooled','o','-'),('B1_flat_balanced','B1 Flat / balanced','s','--'),
        ('B2_hier_ce','B2 Hierarchical / pooled','^','-.'),('B3_hier_balanced','B3 Hierarchical / balanced','D',':'),
        ('P_hier_smooth_worst','P Hierarchical / smooth-worst','v','--'),('C_flat_matched','C Flat / parameter-matched','X','-.')]
conds=['none_0.0','noise_0.05','noise_0.1','blur_0.8','blur_1.3','contrast_0.75','contrast_0.5']
labels=['Clean','Noise\n$\\sigma=0.05$','Noise\n$\\sigma=0.10$',
        'Blur\n$\\sigma=0.8$ px','Blur\n$\\sigma=1.3$ px',
        'Contrast\n$\\alpha=0.75$','Contrast\n$\\alpha=0.50$']
fig,ax=plt.subplots(figsize=(6.3,4.5))
for model,label,marker,ls in models:
    z=records[records.model==model].groupby('condition')['worst_family_macro_f1'].mean().reindex(conds)
    assert z.notna().all()
    ax.plot(np.arange(7),z.to_numpy(),label=label,marker=marker,linestyle=ls,markersize=6,linewidth=1.4)
ax.set_xticks(np.arange(7),labels)
ax.set_ylabel('Worst-family macro-F1\n(mean of five seeds)')
ax.set_ylim(-.025,1.04);ax.set_xlim(-.2,6.2)
ax.set_xlabel('Noise $\\sigma$: normalized intensity; blur $\\sigma$: pixels; contrast $\\alpha$: scale',fontsize=9)
ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.26),ncol=2,frameon=False,fontsize=9.1,
          columnspacing=1.3,handlelength=2.5,handletextpad=.6,labelspacing=.7)
fig.subplots_adjust(left=.14,right=.98,top=.98,bottom=.38)
fig.savefig(OUT/'corruption_response.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(OUT/'corruption_response.png',bbox_inches='tight',pad_inches=.03)
plt.close(fig)
(OUT/'figure_numeric_inputs.json').write_text(json.dumps({'figure2':{'rows':[r[1] for r in rows],
    'means_percentage_points':means.tolist(),'intervals_percentage_points':intervals.tolist()},
    'figure3':{'conditions':conds,'metric':'worst_family_macro_f1','aggregation':'arithmetic mean over five seeds'}},indent=2))
