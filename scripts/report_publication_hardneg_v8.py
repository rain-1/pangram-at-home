"""Compare publication pilot v8 with balanced v6 and Pangram baselines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/f/pangram-at-home')
OUT=HERE.parent/'reports'
RUN='qwen3_token_repeat2_publication_v8_20k'
COLORS=['#169c86','#d46b28','#735a9e','#9b68a6']
NAMES=['Qwen balanced v6','Qwen publication v8','Pangram RoBERTa','Pangram Llama']


def import_comparison():
    spec=importlib.util.spec_from_file_location('compare_v6',HERE/'report_balanced_retrain_v6.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load():
    m=import_comparison();d=m.d
    baseline_pure,baseline_span=d.thresholds()
    old_pure,old_mixed,v6_pure,v6_mixed,_,v6_threshold=m.load()
    v8_threshold=json.loads((ROOT/'runs'/RUN/'v8_human_calibration.json').read_text())['threshold']
    d.MODELS[0]=('Our Qwen publication v8',RUN,COLORS[1])
    pure_map={'v6_llmtrace_heldout':'v8_llmtrace_heldout',
              'v6_synthetic_v4_val':'v8_synthetic_v4_val',
              'v6_external_articles':'v8_external_articles',
              'v6_human_locked_test':'v8_human_locked_test',
              'v6_aitdna':'v8_aitdna'}
    mixed_map={'v6_llmtrace_heldout':'v8_llmtrace_heldout',
               'v6_synthetic_v4_val':'v8_synthetic_v4_val',
               'v6_aitdna':'v8_aitdna','v6_coauthor':'v8_coauthor'}
    d.PURE=[(name,path,pure_map[file],baseline) for name,path,file,baseline in d.PURE]
    d.MIXED=[(name,path,mixed_map[file],baseline) for name,path,file,baseline in d.MIXED]
    v8_pure=d.load_pure([v8_threshold,*baseline_pure[1:]],baseline_span)
    v8_mixed=d.load_mixed([v8_threshold,*baseline_span[1:]])
    return d,old_pure,old_mixed,v6_pure,v6_mixed,v8_pure,v8_mixed,v6_threshold,v8_threshold


def models(v6,v8,source):
    return [v6[source]['models'][0],v8[source]['models'][0],*v6[source]['models'][1:]]


def bars(ax,categories,values,ylabel,lower=False):
    x=np.arange(len(categories));w=.18
    for i in range(4):ax.bar(x+(i-1.5)*w,np.array(values[i])*100,w,color=COLORS[i],label=NAMES[i])
    ax.set_xticks(x,categories);ax.set_ylim(0,105);ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    if lower:ax.text(.99,.98,'LOWER IS BETTER',transform=ax.transAxes,ha='right',va='top',fontsize=8)


def doc_scores(name):
    with np.load(ROOT/'runs'/RUN/f'{name}_scores.npz') as data:
        scores=data['score'];offsets=data['document_offsets']
        return np.array([scores[offsets[i]:offsets[i+1]].max() for i in range(len(offsets)-1)])


def source_aware(threshold):
    cal=doc_scores('v8_commonpile_calibration')
    own=float(np.nextafter(np.sort(cal)[::-1][int(.05*len(cal))],np.inf))
    combined=max(threshold,own)
    external=doc_scores('v8_external_articles')
    rows=[json.loads(line) for line in (ROOT/'data/span_ai_eval_candidate_v1/test.jsonl').open()]
    labels=np.array([row['kind']=='ai' for row in rows])
    heldout=doc_scores('v8_commonpile_locked_test')
    return {'original_threshold':threshold,'commonpile_5pct_threshold':own,
            'source_aware_threshold':combined,'at_source_aware':{
                'external_human_false_alarms':int(((external>=combined)&~labels).sum()),
                'external_human_total':int((~labels).sum()),
                'external_ai_caught':int(((external>=combined)&labels).sum()),
                'external_ai_total':int(labels.sum()),
                'commonpile_human_false_alarms':int((heldout>=combined).sum()),
                'commonpile_human_total':len(heldout)},
            'at_original':{'commonpile_human_false_alarms':int((heldout>=threshold).sum()),
                           'commonpile_human_total':len(heldout)}}


def main():
    d,old_pure,old_mix,v6_pure,v6_mix,v8_pure,v8_mix,v6_thr,v8_thr=load()
    adjusted=source_aware(v8_thr)
    (ROOT/'runs'/RUN/'v8_source_aware_threshold_analysis.json').write_text(json.dumps(adjusted,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    pdf=OUT/'publication_hardneg_v8.pdf'
    with PdfPages(pdf) as pages:
        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Did publication hard negatives lower false alarms?',fontsize=19,weight='bold',y=.97)
        fig.text(.055,.91,'Each Qwen threshold was calibrated to 5% false alarms on the same separate 1,120 human controls.',fontsize=9)
        sources=['LLMTrace test','External articles','Locked human test','AITDNA human controls']
        bundles=[models(v6_pure,v8_pure,s) for s in sources]
        bars(axes[0],['LLMTrace','Published articles','Other human','AITDNA controls'],
             [[b[i]['fp']/b[i]['human'] for b in bundles] for i in range(4)],
             'Human document false alarms (%)',True)
        sources=['LLMTrace test','Synthetic v4 validation','External articles']
        bundles=[models(v6_pure,v8_pure,s) for s in sources]
        bars(axes[1],['LLMTrace AI','Synthetic AI','Article AI'],
             [[b[i]['ai_recall'] for b in bundles] for i in range(4)],'AI documents caught (%)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pages.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Mixed documents: passage detection and false marks',fontsize=18,weight='bold',y=.97)
        sources=['LLMTrace test','Synthetic v4 validation','AITDNA collaboration','CoAuthor collaboration']
        bundles=[models(v6_mix,v8_mix,s) for s in sources]
        labels=['LLMTrace','Synthetic','AITDNA','CoAuthor']
        bars(axes[0],labels,[[b[i]['ai_recall'] for b in bundles] for i in range(4)],
             'AI tokens highlighted (%)')
        bars(axes[1],labels,[[b[i]['human_fpr'] for b in bundles] for i in range(4)],
             'Human tokens falsely marked (%)',True)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pages.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.7,8.3))
        fig.suptitle('Published article ROC: ranking across thresholds',fontsize=18,weight='bold',y=.97)
        rows=v6_pure['External articles']['rows'];y=np.array([row['kind']=='ai' for row in rows])
        for i,m in enumerate(models(v6_pure,v8_pure,'External articles')):
            fpr,tpr,_=roc_curve(y,m['score'])
            ax.plot(fpr,tpr,color=COLORS[i],lw=2,label=f'{NAMES[i]} · AUC {m["auroc"]:.3f}')
            ax.scatter(m['fp']/m['human'],m['tp']/m['ai'],s=60,color=COLORS[i],zorder=4)
        ax.plot([0,1],[0,1],'--',color='#94a3b8')
        ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='False-positive rate on human articles',
               ylabel='Recall on AI articles')
        ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.9));pages.savefig(fig);plt.close(fig)

    lines=['# Publication hard-negative pilot v8','',
           'The v8 model changes 500 of 20,000 training documents from DAMASHA mixed examples to dated, attributed human publication prose from five CC BY publishers. The architecture, tuned hyperparameters, and initialization match v6. Original article stress results have guided this experiment and are now development evidence, not an untouched final test.','',
           f'Qwen v6 threshold: `{v6_thr:.4f}`; v8 threshold: `{v8_thr:.4f}`. Both use the same independent 1,120-document calibration protocol.','',
           '## Fully human and fully AI documents','',
           '| Dataset | Model | Human false alarms | AI caught | Document AUROC |',
           '| --- | --- | ---: | ---: | ---: |']
    for source in v6_pure:
        for name,m in zip(NAMES,models(v6_pure,v8_pure,source)):
            ai=f'{m["tp"]}/{m["ai"]}' if m['ai'] else '—'
            auc=f'{m["auroc"]:.4f}' if m['auroc'] is not None else '—'
            lines.append(f'| {source} | {name} | {m["fp"]}/{m["human"]} | {ai} | {auc} |')
    lines+=['','## Mixed documents','',
            '| Dataset | Model | AI-token recall | Human-token false marks | AI spans half covered |',
            '| --- | --- | ---: | ---: | ---: |']
    for source in v6_mix:
        for name,m in zip(NAMES,models(v6_mix,v8_mix,source)):
            lines.append(f'| {source} | {name} | {100*m["ai_recall"]:.1f}% | '
                         f'{100*m["human_fpr"]:.2f}% | {100*m["span_recall_half_covered"]:.1f}% |')
    lines+=['','## Separate publication controls','',
            '| Human-only source | Qwen v6 false alarms | Qwen v8 false alarms |',
            '| --- | ---: | ---: |']
    for label,v6_file,v8_file,total in (
        ('PMC pre-2023','v6_pmc_article_test','v8_pmc_article_test',346),
        ('CNN/Daily Mail historical','v6_cnn_article_test','v8_cnn_article_test',500)):
        v6=json.loads((ROOT/'runs/qwen3_token_repeat2_balanced_v6_20k'/f'{v6_file}.json').read_text())
        v8=json.loads((ROOT/'runs'/RUN/f'{v8_file}.json').read_text())
        lines.append(f'| {label} | {v6["overall"]["pure_human_documents_with_false_highlight"]}/{total} | '
                     f'{v8["overall"]["pure_human_documents_with_false_highlight"]}/{total} |')
    common=json.loads((ROOT/'runs'/RUN/'v8_commonpile_locked_test.json').read_text())
    lines.append(f'| Common Pile four unseen publishers | — | {common["overall"]["pure_human_documents_with_false_highlight"]}/200 |')
    adj=adjusted['at_source_aware']
    lines+=['','## Independent publication threshold check','',
            f'Common Pile human calibration threshold for 5% FPR: `{adjusted["commonpile_5pct_threshold"]:.4f}`; '
            f'source-aware maximum with original controls: `{adjusted["source_aware_threshold"]:.4f}`.','',
            f'At that source-aware threshold: external human false alarms {adj["external_human_false_alarms"]}/150; '
            f'external AI caught {adj["external_ai_caught"]}/150; '
            f'new held-out Common Pile human false alarms {adj["commonpile_human_false_alarms"]}/200.','',
            'Common Pile calibration and locked-test articles come from publishers absent in v8 training. Dates/bylines provide strong but not absolute authorship evidence. The external Human Detectors set has attributed authors but AI-free workflows were not independently verified. EditLens has coarse broadcast window scores for mixed text.']
    (OUT/'publication_hardneg_v8.md').write_text('\n'.join(lines)+'\n')
    print(pdf)


if __name__=='__main__':main()
