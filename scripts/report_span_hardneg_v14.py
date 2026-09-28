"""Summarize the v14 pilot against v13 and the two open Pangram baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/mnt/f/pangram-at-home')
REPO = Path(__file__).resolve().parents[1]
RUNS = {
    'v13': ROOT/'runs/qwen3_token_repeat2_new_sources_v13_20k',
    'v14': ROOT/'runs/qwen3_token_repeat2_hardneg_v14_21k',
}
BASELINES = json.loads((REPO/'reports/essay_paired_v10_comparison.json').read_text())
BASELINE_RUNS = {
    'Pangram RoBERTa': ROOT/'runs/open_pangram_editlens_roberta_new_sources_v12',
    'Pangram Llama': ROOT/'runs/open_pangram_editlens_llama_new_sources_v12',
}


def load(run: Path, stem: str) -> dict:
    return json.loads((run/(stem+'.json')).read_text())


def caught(run: Path, stem: str, dataset: Path, threshold: float) -> tuple[int, int]:
    with np.load(run/(stem+'_scores.npz')) as data:
        scores = data['score']
        offsets = data['document_offsets']
        ids = data['document_ids']
    rows = [json.loads(line) for line in dataset.open()]
    assert len(rows) == len(ids) == len(offsets)-1
    assert all(row['id'] == str(identity) for row, identity in zip(rows, ids))
    selected = [i for i, row in enumerate(rows) if row['kind'] == 'ai']
    return (sum(bool(np.any(scores[offsets[i]:offsets[i+1]] >= threshold))
                for i in selected), len(selected))


def row(run: Path, prefix: str) -> dict:
    h = load(run, prefix+'_human_locked_test')
    e = load(run, prefix+'_external_articles')
    l = load(run, prefix+'_llmtrace_heldout')
    a = load(run, prefix+'_aitdna')
    n = load(run, prefix+'_new_source_holdout')
    c = load(run, prefix+'_cnn_article_test')
    p = load(run, prefix+'_pmc_article_test')
    s = load(run, prefix+'_asap2_locked_test')
    return {
        'broad_human': h['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'external_human': e['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'external_ai': caught(run, prefix+'_external_articles',
            ROOT/'data/span_ai_eval_candidate_v1/test.jsonl', e['threshold']),
        'llmtrace_human': l['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'llmtrace_ai': caught(run, prefix+'_llmtrace_heldout',
            ROOT/'data/span_size_curve_v5/size_20000/test_llmtrace.jsonl', l['threshold']),
        'llmtrace_mixed': l['by_kind']['mixed'],
        'aitdna_mixed': a['by_kind']['mixed'],
        'new_human': n['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'new_mixed': n['by_kind']['mixed'],
        'new_human_sources': {key: value['pure_human_documents_with_false_highlight']
                              for key, value in n['by_source_family'].items()},
        'cnn': c['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'pmc': p['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'asap': s['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'external_auroc': e['overall']['roc_auc'],
        'llmtrace_auroc': l['overall']['roc_auc'],
    }


def baseline(name: str, run: Path) -> dict:
    old = BASELINES[name]['thresholds']['2.0%']
    n = load(run, 'new_source_holdout')
    return {
        'broad_human': old['human']['persuade']['human_flagged']+
                       old['human']['writers']['human_flagged'],
        'external_human': old['human']['external']['human_flagged'],
        'external_ai': (old['external_ai']['ai_flagged'], 150),
        'llmtrace_human': old['llmtrace_ai']['human_flagged'],
        'llmtrace_ai': (old['llmtrace_ai']['ai_flagged'], 516),
        'llmtrace_mixed': {'ai_recall': old['mixed']['llmtrace']['ai_token_recall'],
                           'fpr': old['mixed']['llmtrace']['human_token_fpr']},
        'aitdna_mixed': {'ai_recall': old['mixed']['aitdna']['ai_token_recall'],
                         'fpr': old['mixed']['aitdna']['human_token_fpr']},
        'new_human': n['by_kind']['human']['pure_human_documents_with_false_highlight'],
        'new_mixed': n['by_kind']['mixed'],
        'cnn': old['human']['cnn']['human_flagged'],
        'pmc': old['human']['pmc']['human_flagged'],
        'asap': old['human']['asap']['human_flagged'],
        'external_auroc': BASELINES[name]['external_document_auroc'],
        'llmtrace_auroc': BASELINES[name]['llmtrace_pure_document_auroc'],
    }


def pct(x: float) -> str:
    return f'{100*x:.1f}%'


def main() -> None:
    data = {name: row(run, name) for name, run in RUNS.items()}
    data.update({name: baseline(name, run) for name, run in BASELINE_RUNS.items()})
    summary = json.loads((RUNS['v14']/'train_summary.json').read_text())
    status = json.loads((ROOT/'span_hardneg_v14_status.json').read_text())
    assert summary['global_step'] == 3352 and status['phase'] == 'complete'
    lines = ['# Essay/forum hard-negative pilot (v14): results', '',
             'V14 adds 1,200 balanced human and AI essay/forum documents to the '
             '20,000-document v13 mix. The Qwen3-1.7B Repeat2 architecture and '
             'fine-tuning settings are unchanged. Every threshold is calibrated '
             'independently on the same separate human set to target 2% of documents '
             'receiving any false AI highlight. The two Pangram EditLens baselines '
             'use their separately calibrated thresholds. All rows below use the '
             'same locked evaluations.', '',
             '## Pure human and AI documents', '',
             '| Model | Broad human false alarms | External article human false alarms | '
             'CNN human false alarms | PMC human false alarms | External AI caught | '
             'LLMTrace AI caught |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, x in data.items():
        lines.append(f'| {name} | {x["broad_human"]}/3,579 | '
                     f'{x["external_human"]}/150 | {x["cnn"]}/500 | '
                     f'{x["pmc"]}/346 | {x["external_ai"][0]}/150 | '
                     f'{x["llmtrace_ai"][0]}/516 |')
    lines += ['', '## Mixed documents', '',
              '| Model | LLMTrace AI recall / human FPR | AITDNA AI recall / human FPR | '
              'New-source AI recall / human FPR | New-source human false alarms |',
              '|---|---:|---:|---:|---:|']
    for name, x in data.items():
        mixed = [f'{pct(x[key]["ai_recall"])} / {pct(x[key]["fpr"])}'
                 for key in ('llmtrace_mixed', 'aitdna_mixed', 'new_mixed')]
        lines.append(f'| {name} | {" | ".join(mixed)} | {x["new_human"]}/590 |')
    lines += ['', '## Additional held-out human checks', '',
              '| Model | LLMTrace pure human false alarms | ASAP student essays false alarms |',
              '|---|---:|---:|']
    for name, x in data.items():
        lines.append(f'| {name} | {x["llmtrace_human"]}/720 | {x["asap"]}/200 |')
    lines += ['', '## Interpretation', '',
              f'- V14 finished {summary["global_step"]:,} steps in '
              f'{summary["train_runtime_seconds"]/3600:.2f} training hours. '
              f'[W&B run]({summary["wandb_run_url"]}).',
              '- V14 improves mixed AI-token recall, especially LLMTrace (48.7% to '
              '58.9%) and new-source spans (64.0% to 73.8%), but also raises '
              'human-token false positives on those sets (2.0% to 3.4% and '
              '0.3% to 1.1%, respectively).',
              '- The broad human alarm count rises from 19 to 25 of 3,579; CNN '
              'rises from 8 to 25 of 500; LLMTrace pure human rises from 5 to 28 '
              'of 720. This is not an across-the-board win.',
              '- The Pangram baselines retain much lower false-positive counts '
              'on the pure-human sets. Their mixed-span scores come from coarse '
              'window broadcasts and should be read with that limitation.',
              '- Pure-document detection and mixed-span localization are different '
              'operating conditions; report both rather than averaging them into '
              'one score.', '']
    (REPO/'reports/span_hardneg_v14_results.md').write_text('\n'.join(lines))
    (REPO/'reports/span_hardneg_v14_results.json').write_text(json.dumps(data, indent=2)+'\n')
    names=list(data)
    colors=['#087e8b','#c34d25','#d98628','#735ca2']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,total,title in [
            (axes[0,0],'broad_human',3579,'Broad human documents: false alarms'),
            (axes[0,1],'cnn',500,'CNN human articles: false alarms')]:
        ax.bar(names,[data[n][key]/total*100 for n in names],color=colors)
        ax.set_xticks(range(len(names)),names,rotation=18,ha='right')
        ax.set_ylabel('Documents with any false AI highlight (%)')
        ax.set_title(title);ax.grid(axis='y',alpha=.25)
    for ax,key,title in [(axes[1,0],'llmtrace_mixed','LLMTrace mixed spans'),
                         (axes[1,1],'new_mixed','New-source mixed spans')]:
        for n,color in zip(names,colors):
            x=data[n][key]
            ax.scatter(100*x['fpr'],100*x['ai_recall'],s=80,color=color,label=n)
            ax.annotate(n,(100*x['fpr'],100*x['ai_recall']),xytext=(5,4),
                        textcoords='offset points',fontsize=8)
        ax.set_xlabel('Human-token FPR (%)');ax.set_ylabel('AI-token recall (%)')
        ax.set_title(title);ax.grid(alpha=.25)
    fig.suptitle('V14 pilot versus v13 and open Pangram baselines')
    fig.tight_layout()
    fig.savefig(REPO/'reports/span_hardneg_v14_results.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
