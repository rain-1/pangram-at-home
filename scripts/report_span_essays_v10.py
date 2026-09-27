"""Compare the essay-paired pilot at calibration-only operating points."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path('/mnt/f/pangram-at-home')
RUNS = ROOT/'runs'
REPORTS = Path(__file__).resolve().parents[1]/'reports'
MODELS = {
    'v8': ('qwen3_token_repeat2_publication_v8_20k', '#d5873b'),
    'v9': ('qwen3_token_repeat2_science_paired_v9_20k', '#168f83'),
    'v10': ('qwen3_token_repeat2_essay_paired_v10_20k', '#435acb'),
}
STEMS = {
    'calibration': {'v8':'v8_human_calibration','v9':'v9_human_calibration','v10':'v10_human_calibration'},
    'generic': {'v8':'v8_human_locked_test','v9':'v9_human_locked_test','v10':'v10_human_locked_test'},
    'external': {'v8':'v8_external_articles','v9':'v9_external_articles','v10':'v10_external_articles'},
    'cnn': {'v8':'v8_cnn_article_test','v9':'v9_cnn_article_test','v10':'v10_cnn_article_test'},
    'pmc': {'v8':'v8_pmc_article_test','v9':'v9_pmc_article_test','v10':'v10_pmc_article_test'},
    'epa': {'v8':'v9_archived_epa_human_v8','v9':'v9_archived_epa_human','v10':'v10_archived_epa_human'},
    'magazine': {'v8':'v10_magazine_test_v8','v9':'v10_magazine_test_v9','v10':'v10_magazine_locked_test'},
    'asap': {'v8':'v10_asap2_locked_test_v8','v9':'v10_asap2_locked_test_v9','v10':'v10_asap2_locked_test'},
    'llmtrace': {'v8':'v8_llmtrace_heldout','v9':'v9_llmtrace_heldout','v10':'v10_llmtrace_heldout'},
    'aitdna': {'v8':'v8_aitdna','v9':'v9_aitdna','v10':'v10_aitdna'},
}
HUMAN_LABELS = {
    'external':'External articles', 'persuade':'PERSUADE essays',
    'writers':'Writers Stack Exchange', 'cnn':'CNN articles', 'pmc':'PMC papers',
    'epa':'Archived EPA', 'magazine':'Archived Smithsonian', 'asap':'ASAP 2.0 essays',
}
MIXED_LABELS = {'llmtrace':'LLMTrace', 'aitdna':'AITDNA'}


def load(tag, key):
    run = RUNS/MODELS[tag][0]
    path = run/(STEMS[key][tag]+'_scores.npz')
    with np.load(path) as data:
        return {name:data[name] for name in ('score','label','document_offsets')}


def threshold(calibration, target):
    score, offsets = calibration['score'], calibration['document_offsets']
    maxima = np.array([score[offsets[i]:offsets[i+1]].max()
                       for i in range(len(offsets)-1)])
    ordered = np.sort(maxima)[::-1]
    return float(np.nextafter(ordered[int(np.floor(target*len(ordered)))], np.inf))


def result(data, cutoff, doc_indices=None):
    score, labels, offsets = (data[name] for name in ('score','label','document_offsets'))
    indices = range(len(offsets)-1) if doc_indices is None else doc_indices
    human_docs = ai_docs = human_flagged = ai_flagged = 0
    human_tokens = ai_tokens = false_tokens = true_tokens = 0
    for i in indices:
        a,b = offsets[i:i+2]
        if a == b:
            continue
        y = labels[a:b]
        marked = score[a:b] >= cutoff
        human = y == 0
        ai = y == 1
        human_tokens += int(human.sum())
        ai_tokens += int(ai.sum())
        false_tokens += int((marked & human).sum())
        true_tokens += int((marked & ai).sum())
        if human.all():
            human_docs += 1
            human_flagged += bool(marked.any())
        elif ai.all():
            ai_docs += 1
            ai_flagged += bool(marked.any())
    return {'human_docs':human_docs, 'human_flagged':human_flagged,
            'ai_docs':ai_docs, 'ai_flagged':ai_flagged,
            'human_token_fpr':false_tokens/human_tokens if human_tokens else None,
            'ai_token_recall':true_tokens/ai_tokens if ai_tokens else None}


def external_roc(data):
    score, labels, offsets = (data[name] for name in ('score','label','document_offsets'))
    values = np.array([score[offsets[i]:offsets[i+1]].max() for i in range(len(offsets)-1)])
    y = np.array([int(labels[offsets[i]:offsets[i+1]].mean() > .5)
                  for i in range(len(offsets)-1)])
    return y, values


def main():
    REPORTS.mkdir(exist_ok=True)
    generic_rows = [json.loads(line) for line in
                    (ROOT/'data/span_human_eval_v2/test.jsonl').open()]
    groups = {
        'persuade':[i for i,row in enumerate(generic_rows) if row['source']=='persuade_2.0'],
        'writers':[i for i,row in enumerate(generic_rows) if row['source']=='writers.stackexchange.com'],
    }
    assert len(groups['persuade']) == 3000 and len(groups['writers']) == 579
    loaded = {tag:{key:load(tag,key) for key in STEMS} for tag in MODELS}
    summary = {}
    for tag in MODELS:
        cal = loaded[tag]['calibration']
        summary[tag] = {'thresholds':{}, 'human':{}, 'mixed':{}}
        for target in (.005,.01,.02,.05):
            cutoff = threshold(cal,target)
            generic = loaded[tag]['generic']
            human = {key:result(loaded[tag][key],cutoff)
                     for key in ('external','cnn','pmc','epa','magazine','asap')}
            human.update({key:result(generic,cutoff,indices) for key,indices in groups.items()})
            mixed = {key:result(loaded[tag][key],cutoff) for key in MIXED_LABELS}
            summary[tag]['thresholds'][f'{target:.1%}'] = {
                'cutoff':cutoff, 'human':human, 'mixed':mixed,
                'external_ai':result(loaded[tag]['external'],cutoff)}
        y, score = external_roc(loaded[tag]['external'])
        summary[tag]['external_document_auroc'] = float(roc_auc_score(y,score))
    (REPORTS/'essay_paired_v10_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

    target = '2.0%'
    pdf = REPORTS/'essay_paired_v10_comparison.pdf'
    with PdfPages(pdf) as pages:
        fig,ax = plt.subplots(figsize=(12,7))
        keys = list(HUMAN_LABELS)
        y = np.arange(len(keys))
        width = .25
        for position,(tag,(_,color)) in enumerate(MODELS.items()):
            metrics = summary[tag]['thresholds'][target]['human']
            values = [100*metrics[key]['human_flagged']/metrics[key]['human_docs'] for key in keys]
            ax.barh(y+(position-1)*width,values,width,color=color,label=tag)
            for yi,value,key in zip(y+(position-1)*width,values,keys):
                row = metrics[key]
                ax.text(value+.35,yi,f'{row["human_flagged"]}/{row["human_docs"]}',
                        va='center',fontsize=8)
        ax.set_yticks(y,[HUMAN_LABELS[key] for key in keys]);ax.invert_yaxis()
        ax.set_xlabel('Documents with any false highlight (%)')
        ax.set_xlim(0,max(40,ax.get_xlim()[1]*1.12))
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend()
        ax.set_title('Human false alarms at 2% generic calibration target',loc='left',fontsize=15)
        fig.text(.09,.015,'PERSUADE and Writers are disjoint slices of the generic held-out human test. '
                 'ASAP test prompts are excluded from v10 training.',fontsize=8)
        fig.tight_layout(rect=[0,.035,1,1]);pages.savefig(fig);plt.close(fig)

        fig,axes = plt.subplots(1,2,figsize=(12,5.7))
        keys=list(MIXED_LABELS);x=np.arange(len(keys));width=.25
        for position,(tag,(_,color)) in enumerate(MODELS.items()):
            metrics=summary[tag]['thresholds'][target]['mixed']
            for ax,field,title in ((axes[0],'ai_token_recall','AI tokens highlighted'),
                                   (axes[1],'human_token_fpr','Human tokens falsely highlighted')):
                vals=[100*metrics[key][field] for key in keys]
                ax.bar(x+(position-1)*width,vals,width,color=color,label=tag)
                ax.set_xticks(x,[MIXED_LABELS[key] for key in keys])
                ax.set_ylabel('Token rate (%)');ax.set_title(title,loc='left')
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[1].legend();fig.suptitle('Mixed-authorship localization at 2% generic calibration target',
                                      x=.06,ha='left',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.92]);pages.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        for tag,(_,color) in MODELS.items():
            y,score=external_roc(loaded[tag]['external'])
            fpr,tpr,_=roc_curve(y,score)
            for ax in axes:ax.plot(fpr,tpr,label=f'{tag} ({roc_auc_score(y,score):.4f})',color=color,lw=2)
        for name,run,color in [('Pangram RoBERTa','open_pangram_editlens_roberta_v5','#9156a8'),
                               ('Pangram Llama','open_pangram_editlens_llama_v5','#b95f88')]:
            rows=[json.loads(line) for line in (RUNS/run/'external_human_detectors.jsonl').open()]
            y=np.array([int(row['kind']=='ai') for row in rows])
            score=np.array([row['score'] for row in rows])
            fpr,tpr,_=roc_curve(y,score)
            for ax in axes:ax.plot(fpr,tpr,label=f'{name} ({roc_auc_score(y,score):.4f})',color=color,lw=2)
        axes[0].set(xlim=(0,1),ylim=(0,1),title='Full ROC')
        axes[1].set(xlim=(0,.2),ylim=(.75,1),title='Low false-positive region')
        for ax in axes:
            ax.set_xlabel('Human article false-positive rate')
            ax.set_ylabel('AI article recall')
            ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
        fig.suptitle('External article ranking across thresholds',x=.06,ha='left',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.93]);pages.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        for tag,(_,color) in MODELS.items():
            points=[]
            for target_key,row in summary[tag]['thresholds'].items():
                h=row['human'];a=row['external_ai']
                points.append((h['persuade']['human_flagged'],h['external']['human_flagged'],
                               100*a['ai_token_recall'],target_key))
            axes[0].plot([p[0] for p in points],[p[2] for p in points],marker='o',lw=2,color=color,label=tag)
            axes[1].plot([p[1] for p in points],[p[2] for p in points],marker='o',lw=2,color=color,label=tag)
            for p in points:
                axes[0].annotate(p[3],(p[0],p[2]),xytext=(3,3),textcoords='offset points',fontsize=7)
                axes[1].annotate(p[3],(p[1],p[2]),xytext=(3,3),textcoords='offset points',fontsize=7)
        axes[0].set(xlabel='PERSUADE human essays flagged (of 3,000)',title='Student essay transfer')
        axes[1].set(xlabel='External human articles flagged (of 150)',title='Publication transfer')
        for ax in axes:
            ax.set_ylabel('External AI-token recall (%)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Decision thresholds selected on separate human calibration',x=.06,ha='left',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.93]);pages.savefig(fig);plt.close(fig)

    markdown=['# Paired student essay v10 comparison','',
              'All rows below use a threshold selected for 2% document false alarms on the separate '
              'generic-human calibration split. v8 and v9 were rescored from saved token scores; '
              'v10 was evaluated directly at its frozen threshold. The external article and PERSUADE '
              'sets have informed development and are now development tests rather than blind tests.','',
              '## Human documents with any false highlight','',
              '| Source | v8 | v9 | v10 |','|---|---:|---:|---:|']
    for key,label in HUMAN_LABELS.items():
        values=[]
        for tag in MODELS:
            row=summary[tag]['thresholds'][target]['human'][key]
            values.append(f'{row["human_flagged"]}/{row["human_docs"]}')
        markdown.append(f'| {label} | '+' | '.join(values)+' |')
    markdown+=['','## AI-token recall on mixed documents','',
               '| Source | v8 | v9 | v10 |','|---|---:|---:|---:|']
    for key,label in MIXED_LABELS.items():
        values=[]
        for tag in MODELS:
            row=summary[tag]['thresholds'][target]['mixed'][key]
            values.append(f'{row["ai_token_recall"]:.1%} recall / {row["human_token_fpr"]:.1%} FPR')
        markdown.append(f'| {label} | '+' | '.join(values)+' |')
    markdown+=['','## External AI articles','',
               '| Model | AI-token recall | AI articles with any highlight | Document AUROC |',
               '|---|---:|---:|---:|']
    for tag in MODELS:
        row=summary[tag]['thresholds'][target]['external_ai']
        markdown.append(f'| {tag} | {row["ai_token_recall"]:.1%} | '
                        f'{row["ai_flagged"]}/{row["ai_docs"]} | '
                        f'{summary[tag]["external_document_auroc"]:.4f} |')
    markdown+=['','The two open Pangram baselines appear on the external article ROC chart. '
               'They make whole-document decisions, so their article alarm counts are not directly '
               'equivalent to any-token highlights from these span models. V8/v9 saved-score '
               'threshold sweeps may differ by one document at a score tie because those older '
               'exports were float32; v10 exports retain float64.','',
               f'[Download charts]({pdf.name})','']
    (REPORTS/'essay_paired_v10_comparison.md').write_text('\n'.join(markdown))
    print(pdf)


if __name__ == '__main__':
    main()
