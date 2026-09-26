"""Compare source-capped retrain with old Qwen and the two Pangram baselines."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dashboard_v5',HERE/'report_evaluation_dashboard_v5.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
OUT=HERE.parent/'reports'
RUN='qwen3_token_repeat2_balanced_v6_20k'
OLD='qwen3_token_repeat2_v5_20k_e1_local'
COLORS=['#075383','#169c86','#735a9e','#9b68a6']
NAMES=['Qwen old','Qwen balanced','Pangram RoBERTa','Pangram Llama']


def pct(x):return f'{100*x:.1f}%'


def load():
    old_pure_thresholds,old_span_thresholds=d.thresholds()
    old_pure=d.load_pure(old_pure_thresholds,old_span_thresholds)
    old_mixed=d.load_mixed(old_span_thresholds)
    assert json.loads((d.RUNS/RUN/'v6_human_calibration.json').read_text())['threshold_source'].startswith('same-set pure-human')
    new_threshold=json.loads((d.RUNS/RUN/'v6_human_calibration.json').read_text())['threshold']
    d.MODELS[0]=('Our Qwen balanced',RUN,COLORS[1])
    pure_map={'v5_llmtrace_heldout':'v6_llmtrace_heldout',
              'v5_prior_synthetic_val':'v6_synthetic_v4_val',
              'v5_external_human_detectors':'v6_external_articles',
              'v5_human_locked_test':'v6_human_locked_test',
              'v5_aitdna':'v6_aitdna'}
    mixed_map={'v5_llmtrace_heldout':'v6_llmtrace_heldout',
               'v5_prior_synthetic_val':'v6_synthetic_v4_val',
               'v5_aitdna':'v6_aitdna','v5_coauthor':'v6_coauthor'}
    d.PURE=[(name,path,pure_map[file],baseline) for name,path,file,baseline in d.PURE]
    d.MIXED=[(name,path,mixed_map[file],baseline) for name,path,file,baseline in d.MIXED]
    new_pure=d.load_pure([new_threshold,*old_pure_thresholds[1:]],old_span_thresholds)
    new_mixed=d.load_mixed([new_threshold,*old_span_thresholds[1:]])
    return old_pure,old_mixed,new_pure,new_mixed,old_pure_thresholds[0],new_threshold


def combined(old,new,source):
    return [old[source]['models'][0],new[source]['models'][0],*old[source]['models'][1:]]


def bars(ax,labels,values,ylabel,lower=False):
    x=np.arange(len(labels));w=.18
    for i,(name,color) in enumerate(zip(NAMES,COLORS)):
        series=np.array(values[i])*100
        ax.bar(x+(i-1.5)*w,series,width=w,color=color,label=name)
    ax.set_xticks(x,labels)
    ax.set_ylim(0,105);ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    if lower:ax.text(.99,.98,'LOWER IS BETTER',transform=ax.transAxes,ha='right',va='top',fontsize=8)


def main():
    old_pure,old_mixed,new_pure,new_mixed,old_threshold,new_threshold=load()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    output=OUT/'balanced_retrain_v6.pdf'
    with PdfPages(output) as pdf:
        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Did the source-balanced retrain fix false alarms?',fontsize=19,weight='bold',y=.97)
        fig.text(.055,.91,'Each Qwen model uses a 5% false-alarm calibration on the same independent human set.',fontsize=10)
        sources=['LLMTrace test','External articles','Locked human test','AITDNA human controls']
        labels=['LLMTrace','Published articles','Other human','AITDNA controls']
        models=[combined(old_pure,new_pure,s) for s in sources]
        bars(axes[0],labels,[[m[i]['fp']/m[i]['human'] for m in models] for i in range(4)],
             'Human document false alarms (%)',True)
        sources=['LLMTrace test','Synthetic v4 validation','External articles']
        models=[combined(old_pure,new_pure,s) for s in sources]
        bars(axes[1],['LLMTrace AI','Synthetic AI','Article AI'],
             [[m[i]['ai_recall'] for m in models] for i in range(4)],'AI documents caught (%)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Mixed documents: AI found versus human text falsely marked',fontsize=18,weight='bold',y=.97)
        sources=['LLMTrace test','Synthetic v4 validation','AITDNA collaboration','CoAuthor collaboration']
        models=[combined(old_mixed,new_mixed,s) for s in sources]
        labels=['LLMTrace','Synthetic','AITDNA','CoAuthor']
        bars(axes[0],labels,[[m[i]['ai_recall'] for m in models] for i in range(4)],
             'AI tokens highlighted (%)')
        bars(axes[1],labels,[[m[i]['human_fpr'] for m in models] for i in range(4)],
             'Human tokens falsely marked (%)',True)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pdf.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.7,8.3))
        fig.suptitle('Published article ROC: ranking quality across thresholds',fontsize=18,weight='bold',y=.97)
        rows=old_pure['External articles']['rows'];y=np.array([r['kind']=='ai' for r in rows])
        for i,m in enumerate(combined(old_pure,new_pure,'External articles')):
            fpr,tpr,_=roc_curve(y,m['score'])
            ax.plot(fpr,tpr,color=COLORS[i],lw=2,label=f"{NAMES[i]} · AUC {m['auroc']:.3f}")
            ax.scatter(m['fp']/m['human'],m['tp']/m['ai'],s=60,color=COLORS[i],zorder=4)
        ax.plot([0,1],[0,1],'--',color='#94a3b8')
        ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='False-positive rate on human articles',
               ylabel='Recall on AI articles')
        ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.9));pdf.savefig(fig);plt.close(fig)

    old=old_pure['External articles']['models'][0];new=new_pure['External articles']['models'][0]
    lines=['# Source-balanced retrain evaluation','',f'Old Qwen threshold: `{old_threshold:.4f}`; '
           f'new Qwen threshold: `{new_threshold:.4f}`. Both are independently calibrated to 5% document '
           'false alarms on the same 1,120 human controls. The external article set was not used to select either threshold.','',
           '## Fully human and fully AI documents','',
           '| Dataset | Model | Human false alarms | AI documents caught | Document AUROC |',
           '| --- | --- | ---: | ---: | ---: |']
    for source in old_pure:
        for name,m in zip(NAMES,combined(old_pure,new_pure,source)):
            lines.append(f"| {source} | {name} | {m['fp']}/{m['human']} ({pct(m['fp']/m['human'])}) | "
                         f"{m['tp']}/{m['ai']} ({pct(m['ai_recall'])}) | {m['auroc']:.4f} |" if m['ai'] else
                         f"| {source} | {name} | {m['fp']}/{m['human']} ({pct(m['fp']/m['human'])}) | — | — |")
    lines+=['','## Mixed documents','',
            '| Dataset | Model | AI token recall | Human token false alarms | AI spans half covered |',
            '| --- | --- | ---: | ---: | ---: |']
    for source in old_mixed:
        for name,m in zip(NAMES,combined(old_mixed,new_mixed,source)):
            lines.append(f"| {source} | {name} | {pct(m['ai_recall'])} | "
                         f"{pct(m['human_fpr'])} | {pct(m['span_recall_half_covered'])} |")
    lines+=['','## Verdict','',
            f"Published human articles: old Qwen {old['fp']}/{old['human']} false alarms; "
            f"balanced Qwen {new['fp']}/{new['human']}. AI article recall: "
            f"{old['tp']}/{old['ai']} old; {new['tp']}/{new['ai']} balanced.",'',
            'The 150 external human articles have attributed human authors but their production workflows were not independently verified as AI-free. EditLens baseline span scores are coarse window broadcasts, not native token heads.']
    (OUT/'balanced_retrain_v6.md').write_text('\n'.join(lines)+'\n')
    print(output)


if __name__=='__main__':main()
