"""Compare the v13 GRADTEX-dose follow-up with v10/v12 and Pangram baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/mnt/f/pangram-at-home')
REPO=Path(__file__).resolve().parents[1]
OLD=json.loads((REPO/'reports/essay_paired_v10_comparison.json').read_text())
RUNS={
    'v10':ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k',
    'v12':ROOT/'runs/qwen3_token_repeat2_new_sources_v12_20k',
    'v13':ROOT/'runs/qwen3_token_repeat2_new_sources_v13_20k',
}
BASELINES={
    'Pangram RoBERTa':ROOT/'runs/open_pangram_editlens_roberta_new_sources_v12',
    'Pangram Llama':ROOT/'runs/open_pangram_editlens_llama_new_sources_v12',
}
COLORS={'v10':'#888888','v12':'#087e8b','v13':'#c34d25',
        'Pangram RoBERTa':'#d98628','Pangram Llama':'#735ca2'}


def pct(value: float | None) -> str:
    if value is None:return '—'
    return f'{100*value:.2f}%' if value<.01 else f'{100*value:.1f}%'


def load(run: Path,stem: str) -> dict:
    return json.loads((run/(stem+'.json')).read_text())


def caught(run: Path,stem: str,dataset: Path,threshold: float) -> tuple[int,int]:
    with np.load(run/(stem+'_scores.npz')) as data:
        scores=data['score'];offsets=data['document_offsets'];ids=data['document_ids']
    rows=[json.loads(line) for line in dataset.open()]
    assert len(rows)==len(ids)==len(offsets)-1
    assert all(row['id']==str(identity) for row,identity in zip(rows,ids))
    chosen=[i for i,row in enumerate(rows) if row['kind']=='ai']
    return sum(bool(np.any(scores[offsets[i]:offsets[i+1]]>=threshold))
               for i in chosen),len(chosen)


def gather() -> dict:
    data={}
    for name,run in RUNS.items():
        h=load(run,f'{name}_human_locked_test')
        e=load(run,f'{name}_external_articles')
        l=load(run,f'{name}_llmtrace_heldout')
        a=load(run,f'{name}_aitdna')
        n=load(run,f'{name}_new_source_holdout')
        c=load(run,f'{name}_cnn_article_test')
        p=load(run,f'{name}_pmc_article_test')
        s=load(run,f'{name}_asap2_locked_test')
        data[name]={
            'human':h['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'external_human':e['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'external_ai':caught(run,f'{name}_external_articles',
                ROOT/'data/span_ai_eval_candidate_v1/test.jsonl',e['threshold']),
            'llmtrace_human':l['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'llmtrace_ai':caught(run,f'{name}_llmtrace_heldout',
                ROOT/'data/span_size_curve_v5/size_20000/test_llmtrace.jsonl',l['threshold']),
            'llmtrace_mixed':l['by_kind']['mixed'],
            'aitdna_mixed':a['by_kind']['mixed'],
            'new_human':n['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'new_mixed':n['by_kind']['mixed'],
            'new_human_sources':{source:row['pure_human_documents_with_false_highlight']
                                 for source,row in n['by_source_family'].items()},
            'cnn':c['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'pmc':p['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'asap':s['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'external_auroc':e['overall']['roc_auc'],
            'llmtrace_auroc':l['overall']['roc_auc'],
            'new_mixed_auroc':n['by_kind']['mixed']['roc_auc'],
        }
    for name,run in BASELINES.items():
        old=OLD[name]['thresholds']['2.0%']
        n=load(run,'new_source_holdout')
        data[name]={
            'human':old['human']['persuade']['human_flagged']+
                     old['human']['writers']['human_flagged'],
            'external_human':old['human']['external']['human_flagged'],
            'external_ai':(old['external_ai']['ai_flagged'],150),
            'llmtrace_human':old['llmtrace_ai']['human_flagged'],
            'llmtrace_ai':(old['llmtrace_ai']['ai_flagged'],516),
            'llmtrace_mixed':{'ai_recall':old['mixed']['llmtrace']['ai_token_recall'],
                              'fpr':old['mixed']['llmtrace']['human_token_fpr']},
            'aitdna_mixed':{'ai_recall':old['mixed']['aitdna']['ai_token_recall'],
                            'fpr':old['mixed']['aitdna']['human_token_fpr']},
            'new_human':n['by_kind']['human']['pure_human_documents_with_false_highlight'],
            'new_mixed':n['by_kind']['mixed'],
            'new_human_sources':{},
            'cnn':old['human']['cnn']['human_flagged'],
            'pmc':old['human']['pmc']['human_flagged'],
            'asap':old['human']['asap']['human_flagged'],
            'external_auroc':OLD[name]['external_document_auroc'],
            'llmtrace_auroc':OLD[name]['llmtrace_pure_document_auroc'],
            'new_mixed_auroc':n['by_kind']['mixed']['roc_auc'],
        }
    return data


def chart(data: dict,path: Path) -> None:
    names=list(data)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    fig.suptitle('GRADTEX dose: v10, v12, v13 and Pangram baselines',fontsize=15)
    ax=axes[0,0]
    x=np.arange(len(names))
    ax.bar(x,[data[name]['human'] for name in names],color=[COLORS[name] for name in names])
    ax.set_xticks(x,names,rotation=20,ha='right')
    ax.set_ylabel('False alarms of 3,579 human documents')
    ax.set_title('Broad human evaluation: lower is better')
    ax.grid(axis='y',alpha=.25)
    ax=axes[0,1]
    ax.bar(x,[data[name]['external_human'] for name in names],
           color=[COLORS[name] for name in names])
    ax.set_xticks(x,names,rotation=20,ha='right')
    ax.set_ylabel('False alarms of 150 human articles')
    ax.set_title('External articles: lower is better')
    ax.grid(axis='y',alpha=.25)
    for ax,key,title in ((axes[1,0],'llmtrace_mixed','LLMTrace mixed spans'),
                         (axes[1,1],'new_mixed','New GRADTEX mixed spans')):
        for name in names:
            row=data[name][key]
            ax.scatter(100*row['fpr'],100*row['ai_recall'],
                       color=COLORS[name],s=75,label=name)
            ax.annotate(name,(100*row['fpr'],100*row['ai_recall']),
                        xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Human-token FPR (%)')
        ax.set_ylabel('AI-token recall (%)')
        ax.set_title(title)
        ax.grid(alpha=.25)
    fig.text(.5,.015,'All thresholds use the same separate human calibration set at 2% document-any false highlights. '
             'EditLens mixed scores are coarse window broadcasts.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    status=json.loads((ROOT/'span_new_sources_v13_status.json').read_text())
    assert status['phase'] in {'reporting','complete'}
    manifest=json.loads((ROOT/'data/span_new_sources_v13/manifest.json').read_text())
    audit=json.loads((ROOT/'data/span_new_sources_v13/exposure_audit.json').read_text())
    assert manifest['documents']==20000
    data=gather()
    names=list(data)
    lines=['# Half-dose GRADTEX follow-up (v13)','',
           'V13 keeps the v12 human additions and restores 500 DAMASHA mixed examples '
           'in place of 500 GRADTEX mixed examples. It retains the 20,000-document '
           'budget, v10/v12 validation and new-source holdout files, Qwen3-1.7B '
           'Repeat2 architecture, initialization adapter, and tuned hyperparameters. '
           f'AI-labeled tokens are {audit["ai_supervised_token_fraction"]:.1%} of '
           'supervised positions. Each Qwen threshold is independently chosen on '
           'the same separate human calibration set for 2% document-any false '
           'highlights; the Pangram EditLens scores use the same calibration text.','',
           '## Human and AI documents','',
           '| Model | Broad human alarms | External human alarms | CNN human alarms | '
           'PMC human alarms | External AI caught | LLMTrace pure-AI caught |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        row=data[name]
        lines.append(f'| {name} | {row["human"]}/3,579 | '
                     f'{row["external_human"]}/150 | {row["cnn"]}/500 | '
                     f'{row["pmc"]}/346 | {row["external_ai"][0]}/150 | '
                     f'{row["llmtrace_ai"][0]}/516 |')
    lines+=['','## Mixed-document localization','',
            '| Model | LLMTrace AI-token recall / human-token FPR | '
            'AITDNA AI-token recall / human-token FPR | '
            'New GRADTEX AI-token recall / human-token FPR | '
            'New-source human alarms |',
            '|---|---:|---:|---:|---:|']
    for name in names:
        row=data[name];llm=row['llmtrace_mixed'];ait=row['aitdna_mixed'];new=row['new_mixed']
        lines.append(f'| {name} | {pct(llm["ai_recall"])} / {pct(llm["fpr"])} | '
                     f'{pct(ait["ai_recall"])} / {pct(ait["fpr"])} | '
                     f'{pct(new["ai_recall"])} / {pct(new["fpr"])} | '
                     f'{row["new_human"]}/590 |')
    lines+=['','## New human-source false alarms','',
            '| Qwen model | Dolly (150) | Historical fiction (100) | '
            'Named pre-LLM essays (90) | GRADTEX human (250) |',
            '|---|---:|---:|---:|---:|']
    for name in RUNS:
        src=data[name]['new_human_sources']
        lines.append(f'| {name} | {src["databricks/databricks-dolly-15k"]}/150 | '
                     f'{src["Travis-ML/ShortStory-SFT-jsonl"]}/100 | '
                     f'{src["craphound.com"]+src["aaronsw.com/weblog"]}/90 | '
                     f'{src["elisabeth-pl-pl/GRADTEX"]}/250 |')
    summary=json.loads((RUNS['v13']/'train_summary.json').read_text())
    lines+=['','## Training and interpretation','',
            f'- Completed {summary["global_step"]:,} steps in '
            f'{summary["train_runtime_seconds"]/3600:.2f} hours. '
            f'Best validation partial AUROC: {summary["best_metric"]:.4f}.',
            f'- [Weights & Biases run]({summary["wandb_run_url"]}).',
            '- V13 trades some of v12\'s mixed-span recall for fewer broad-human '
            'false alarms: 19/3,579 versus 45/3,579. V10 still has the fewest '
            'broad-human alarms among our checkpoints (9/3,579), while v13 '
            'improves LLMTrace mixed AI-token recall from 38.6% to 48.7%. '
            'Keep v10 as the conservative default until a calibration-only '
            'threshold sweep shows whether v13 can retain that gain at an '
            'acceptable false-positive rate.',
            '- The Pangram baselines produce fewer pure-human false alarms '
            'on these sets, but their mixed-span recall is lower. Their '
            'window-broadcast localization and our token-level output '
            'represent different resolution levels.',
            '- These evaluations have informed development and are not untouched '
            'final tests. The new-source holdout is disjoint by work/group, not by author. '
            'GRADTEX span boundaries are inferred from preserved context.', '',
            '[Download comparison charts](span_new_sources_v13.pdf)', '']
    (REPO/'reports/span_new_sources_v13.md').write_text('\n'.join(lines))
    (REPO/'reports/span_new_sources_v13.json').write_text(json.dumps(data,indent=2)+'\n')
    chart(data,REPO/'reports/span_new_sources_v13.pdf')
    print(REPO/'reports/span_new_sources_v13.md')


if __name__=='__main__':
    main()
