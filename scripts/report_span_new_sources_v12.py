"""Compare v12, v10, and both open Pangram models at held-out sources."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from span_metrics import summarize_scores

ROOT = Path('/mnt/f/pangram-at-home')
REPO = Path(__file__).resolve().parents[1]
V10 = ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k'
V12 = ROOT/'runs/qwen3_token_repeat2_new_sources_v12_20k'
DATA = ROOT/'data/span_new_sources_v12/new_source_holdout.jsonl'
OLD = json.loads((REPO/'reports/essay_paired_v10_comparison.json').read_text())
MODELS = {
    'v10': (V10, 'v10_new_source_holdout'),
    'v12': (V12, 'v12_new_source_holdout'),
    'Pangram RoBERTa': (ROOT/'runs/open_pangram_editlens_roberta_new_sources_v12',
                        'new_source_holdout'),
    'Pangram Llama': (ROOT/'runs/open_pangram_editlens_llama_new_sources_v12',
                      'new_source_holdout'),
}


def pct(value) -> str:
    return '—' if value is None else f'{100*value:.1f}%'


def loaded(name: str) -> tuple[dict, list[dict]]:
    folder, stem = MODELS[name]
    report = json.loads((folder/f'{stem}.json').read_text())
    with np.load(folder/f'{stem}_scores.npz') as cache:
        scores = cache['score']
        labels = cache['label']
        offsets = cache['document_offsets']
        ids = cache['document_ids']
    rows = [json.loads(line) for line in DATA.open()]
    assert len(rows) == len(ids) == len(offsets)-1
    assert all(row['id'] == str(identity) for row, identity in zip(rows, ids))
    grouped = [{'row': row, 'score': scores[offsets[i]:offsets[i+1]],
                'label': labels[offsets[i]:offsets[i+1]]}
               for i, row in enumerate(rows)]
    return report, grouped


def doc_any(path: Path, stem: str, dataset: Path, threshold: float, kind: str) -> tuple[int, int]:
    with np.load(path/f'{stem}_scores.npz') as cache:
        scores, offsets, ids = (cache[key] for key in ('score','document_offsets','document_ids'))
    rows = [json.loads(line) for line in dataset.open()]
    assert len(rows) == len(ids)
    assert all(row['id'] == str(identity) for row, identity in zip(rows, ids))
    selected = [i for i, row in enumerate(rows) if row['kind'] == kind]
    count = sum(bool(np.any(scores[offsets[i]:offsets[i+1]] >= threshold)) for i in selected)
    return count, len(selected)


def new_group(row: dict) -> str:
    if row['source'] == 'elisabeth-pl-pl/GRADTEX':
        return f'GRADTEX {row["kind"]}'
    if row['source'] == 'databricks/databricks-dolly-15k':
        return 'Dolly employee responses'
    if row['source'] == 'Travis-ML/ShortStory-SFT-jsonl':
        return 'Historical fiction'
    return 'Pre-LLM author essays'


def chart(values: dict, v12_reports: dict) -> Path:
    names = list(MODELS)
    colors = {'v10':'#888888', 'v12':'#087e8b',
              'Pangram RoBERTa':'#d98628', 'Pangram Llama':'#735ca2'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('New-source v12: held-out comparisons at 2% calibration FPR',
                 fontsize=15, fontweight='bold')
    groups = ['Dolly employee responses', 'Historical fiction',
              'Pre-LLM author essays', 'GRADTEX human']
    labels = ['Dolly', 'Historical fiction', 'Pre-LLM essays', 'GRADTEX human']
    x = np.arange(len(groups))
    width = .19
    ax = axes[0, 0]
    for i, name in enumerate(names):
        heights = [100*values[name][group]['pure_human_document_any_false_highlight_rate']
                   for group in groups]
        ax.bar(x+(i-1.5)*width, heights, width, label=name, color=colors[name])
    ax.set_xticks(x, labels, rotation=22, ha='right')
    ax.set_ylabel('Human documents falsely highlighted (%)')
    ax.set_title('New human holdouts: lower is better')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    x = np.arange(len(names))
    recall = [100*values[name]['GRADTEX mixed']['ai_recall'] for name in names]
    fpr = [100*values[name]['GRADTEX mixed']['fpr'] for name in names]
    ax.bar(x-.17, recall, .34, color='#087e8b', label='AI-token recall')
    ax.bar(x+.17, fpr, .34, color='#d98628', label='Human-token FPR')
    ax.set_xticks(x, names, rotation=22, ha='right')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Tokens (%)')
    ax.set_title('GRADTEX mixed spans: recall up, FPR down')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in names:
        if name == 'v12':
            m = v12_reports['llmtrace']['by_kind']['mixed']
            human_fpr, ai_recall = m['fpr'], m['ai_recall']
        else:
            m = OLD[name]['thresholds']['2.0%']['mixed']['llmtrace']
            human_fpr, ai_recall = m['human_token_fpr'], m['ai_token_recall']
        ax.scatter(100*human_fpr, 100*ai_recall, s=85, color=colors[name], label=name)
        ax.annotate(name, (100*human_fpr, 100*ai_recall), xytext=(4, 4),
                    textcoords='offset points', fontsize=8)
    ax.set_xlabel('Human-token FPR (%)')
    ax.set_ylabel('AI-token recall (%)')
    ax.set_title('LLMTrace mixed spans')
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    for name in names:
        if name == 'v12':
            ext = v12_reports['external']
            alarms = ext['by_kind']['human']['pure_human_documents_with_false_highlight']
            caught, total = doc_any(V12, 'v12_external_articles',
                ROOT/'data/span_ai_eval_candidate_v1/test.jsonl', ext['threshold'], 'ai')
        else:
            old = OLD[name]['thresholds']['2.0%']
            alarms = old['human']['external']['human_flagged']
            caught, total = old['external_ai']['ai_flagged'], old['external_ai']['ai_docs']
        ax.scatter(100*alarms/150, 100*caught/total, s=85,
                   color=colors[name], label=name)
        ax.annotate(name, (100*alarms/150, 100*caught/total), xytext=(4, 4),
                    textcoords='offset points', fontsize=8)
    ax.set_xlabel('Human articles falsely highlighted (%)')
    ax.set_ylabel('AI articles detected (%)')
    ax.set_title('External articles')
    ax.grid(alpha=.25)
    fig.text(.5, .015,
             'EditLens broadcasts window scores for mixed spans; Qwen predicts token scores directly. '
             'New holdout is work/group-disjoint development data.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .96))
    output = REPO/'reports/span_new_sources_v12.pdf'
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    assert json.loads((ROOT/'span_new_sources_v12_status.json').read_text())['phase'] in {
        'comparing_models', 'complete'}
    reports = {name: loaded(name) for name in MODELS}
    lines = [
        '# New-source v12 span retrain', '',
        'The model uses the v10 Qwen3-1.7B Repeat2 architecture, initialization adapter,',
        'LoRA settings, effective batch size, validation set, and 20,000-document budget.',
        'Exactly 1,860 documents were replaced. New sources are 600 Dolly responses,',
        '200 historical fiction excerpts, 60 pre-LLM author essay excerpts, and',
        '1,000 GRADTEX mixed completions. Training exposure is 42.4% AI-labeled tokens;',
        'the largest source is DAMASHA at 21.4%.', '',
        'All four models below use thresholds chosen on the same separate human',
        'calibration set for 2% document-any false highlights. EditLens span scores',
        'broadcast window-level decisions; the Qwen scores are native token predictions.',
        'The new-source holdout is a development evaluation, disjoint by source group',
        'from this training mix. GRADTEX boundaries are inferred from exact preserved',
        'context, so they are weaker labels than independently logged provenance.', '',
        '## New-source holdout: human false alarms', '',
        '| Source | Documents | v10 | v12 | Pangram RoBERTa | Pangram Llama |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    groups = ['Dolly employee responses', 'Historical fiction',
              'Pre-LLM author essays', 'GRADTEX human']
    values = {}
    for name, (report, prepared) in reports.items():
        threshold = report['threshold']
        by_group = defaultdict(list)
        for row in prepared:
            by_group[new_group(row['row'])].append(row)
        values[name] = {group: summarize_scores(rows, threshold) for group, rows in by_group.items()}
    for group in groups:
        count = values['v12'][group]['pure_human_documents']
        cells = []
        for name in MODELS:
            v = values[name][group]
            cells.append(f'{v["pure_human_documents_with_false_highlight"]}/{count} '
                         f'({pct(v["pure_human_document_any_false_highlight_rate"])})')
        lines.append(f'| {group} | {count} | '+ ' | '.join(cells)+' |')
    lines += ['', '## New-source holdout: mixed GRADTEX token localization', '',
              '| Model | AI-token recall | Human-token FPR | Token AUROC |',
              '|---|---:|---:|---:|']
    for name in MODELS:
        v = values[name]['GRADTEX mixed']
        lines.append(f'| {name} | {pct(v["ai_recall"])} | {pct(v["fpr"])} | '
                     f'{v["roc_auc"]:.3f} |')
    lines += ['', '## Established held-out evaluations', '',
              '| Model | External human alarms | External AI documents caught | '
              'LLMTrace AI documents caught | Locked human alarms | '
              'LLMTrace mixed AI-token recall / human-token FPR | '
              'AITDNA mixed AI-token recall / human-token FPR |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        if name == 'v12':
            external = json.loads((V12/'v12_external_articles.json').read_text())
            llm = json.loads((V12/'v12_llmtrace_heldout.json').read_text())
            human = json.loads((V12/'v12_human_locked_test.json').read_text())
            ait = json.loads((V12/'v12_aitdna.json').read_text())
            ext_h = external['by_kind']['human']
            locked = human['by_kind']['human']
            ext_ai = doc_any(V12, 'v12_external_articles',
                ROOT/'data/span_ai_eval_candidate_v1/test.jsonl', external['threshold'], 'ai')
            llm_ai = doc_any(V12, 'v12_llmtrace_heldout',
                ROOT/'data/span_size_curve_v5/size_20000/test_llmtrace.jsonl', llm['threshold'], 'ai')
            llm_m = llm['by_kind']['mixed']; ait_m = ait['by_kind']['mixed']
            cells = [f'{ext_h["pure_human_documents_with_false_highlight"]}/150',
                     f'{ext_ai[0]}/{ext_ai[1]}', f'{llm_ai[0]}/{llm_ai[1]}',
                     f'{locked["pure_human_documents_with_false_highlight"]}/3579',
                     f'{pct(llm_m["ai_recall"])} / {pct(llm_m["fpr"])}',
                     f'{pct(ait_m["ai_recall"])} / {pct(ait_m["fpr"])}']
        else:
            old = OLD[name]['thresholds']['2.0%']
            h = old['human']
            lm = old['mixed']['llmtrace']; am = old['mixed']['aitdna']
            # This 3,579-document set contains 3,000 PERSUADE essays and
            # 579 Writers Stack Exchange documents.
            locked_count = h['persuade']['human_flagged']+h['writers']['human_flagged']
            locked_total = h['persuade']['human_docs']+h['writers']['human_docs']
            cells = [f'{h["external"]["human_flagged"]}/150',
                     f'{old["external_ai"]["ai_flagged"]}/150',
                     f'{old["llmtrace_ai"]["ai_flagged"]}/516',
                     f'{locked_count}/{locked_total}',
                     f'{pct(lm["ai_token_recall"])} / {pct(lm["human_token_fpr"])}',
                     f'{pct(am["ai_token_recall"])} / {pct(am["human_token_fpr"])}']
        lines.append(f'| {name} | '+' | '.join(cells)+' |')
    summary = json.loads((V12/'train_summary.json').read_text())
    lines += ['', '## Training and limits', '',
              f'- Completed {summary["global_step"]:,} optimizer steps in '
              f'{summary["train_runtime_seconds"]/3600:.2f} hours; '
              f'best validation partial AUROC: {summary["best_metric"]:.4f}.',
              f'- [Weights & Biases run]({summary["wandb_run_url"]}).',
        '- New-source holdout labels have different evidence levels: Dolly employees were '
              'instructed not to use AI; historical fiction and pre-LLM essays have '
              'publication-era evidence; GRADTEX mixed boundaries are reconstructed.',
              '- The new holdout is work/group-disjoint, not author-disjoint: writers may '
              'appear on both sides. Its author rows test held-out prose, not unknown authors.',
              '- External articles and other established evaluations have informed model '
              'development. They are useful comparisons, not untouched final tests.', '']
    out = REPO/'reports/span_new_sources_v12.md'
    chart_path = chart(values, {
        'llmtrace': json.loads((V12/'v12_llmtrace_heldout.json').read_text()),
        'external': json.loads((V12/'v12_external_articles.json').read_text())})
    lines += [f'[Download comparison charts]({chart_path.name})', '']
    out.write_text('\n'.join(lines))
    print(out)


if __name__ == '__main__':
    main()
