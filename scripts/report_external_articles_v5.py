"""Summarize frozen-threshold transfer to Human Detectors articles."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path('/mnt/f/pangram-at-home')
RUNS = ROOT / 'runs'
OUT = Path(__file__).resolve().parents[1] / 'reports'
DATA = ROOT / 'data/span_ai_eval_candidate_v1/test.jsonl'
QWEN = 'qwen3_token_repeat2_v5_20k_e1_local'
MODELS = [
    ('Our Qwen 20k', '#075383'),
    ('EditLens RoBERTa', '#735a9e'),
    ('EditLens Llama', '#9b68a6'),
]


def load_rows():
    return [json.loads(line) for line in DATA.open()]


def load_results(rows):
    records = []
    folder = RUNS / QWEN
    threshold = json.loads((folder / 'v5_llmtrace_heldout.json').read_text())['threshold']
    with np.load(folder / 'v5_external_human_detectors_scores.npz') as data:
        values, offsets, ids = data['score'], data['document_offsets'], data['document_ids']
        assert len(ids) == len(rows)
        assert all(str(id_) == row['id'] for id_, row in zip(ids, rows))
        scores = [float(values[offsets[i]:offsets[i + 1]].max())
                  for i in range(len(ids))]
    records.append({'name': MODELS[0][0], 'color': MODELS[0][1], 'score': scores,
                    'threshold': threshold, 'aggregation': 'maximum token logit'})
    for key, (name, color) in zip(('roberta', 'llama'), MODELS[1:]):
        folder = RUNS / f'open_pangram_editlens_{key}_v5'
        threshold = json.loads((folder / 'summary.json').read_text())['threshold']
        scored = [json.loads(line) for line in
                  (folder / 'external_human_detectors.jsonl').open()]
        assert len(scored) == len(rows)
        assert all(s['id'] == row['id'] for s, row in zip(scored, rows))
        scores = [s['score'] for s in scored]
        records.append({'name': name, 'color': color, 'score': scores,
                        'threshold': threshold, 'aggregation': 'mean native-window score'})
    for result in records:
        pred = [score >= result['threshold'] for score in result['score']]
        result['human_false'] = sum(hit for hit, row in zip(pred, rows) if row['kind'] == 'human')
        result['ai_detected'] = sum(hit for hit, row in zip(pred, rows) if row['kind'] == 'ai')
        result['auroc'] = roc_auc_score([r['kind'] == 'ai' for r in rows],
                                      result['score'])
        result['by_generator'] = {
            g: sum(hit for hit, row in zip(pred, rows)
                   if row['kind'] == 'ai' and row['generator'] == g)
            for g in sorted({r['generator'] for r in rows if r['kind'] == 'ai'})}
    return records


def render(rows, results):
    OUT.mkdir(exist_ok=True)
    figs = []
    names = [r['name'] for r in results]
    colors = [r['color'] for r in results]
    y = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, key, title, denominator in [
        (axes[0], 'ai_detected', 'AI articles detected', 150),
        (axes[1], 'human_false', 'Human articles falsely flagged', 150),
    ]:
        vals = [100 * r[key] / denominator for r in results]
        ax.barh(y, vals, color=colors, height=.65)
        for i, value in enumerate(vals):
            ax.text(value + 1, i, f"{results[i][key]}/150", va='center')
        ax.set_xlim(0, max(105 if key == 'ai_detected' else 30, max(vals) + 13))
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel('Articles (%)')
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('External article stress test: frozen thresholds', fontsize=15)
    fig.text(.5, .02, '150 AI + 150 attributed-human articles from Human Detectors. No threshold tuning on these articles.\n'
             'Human source articles have named authors, but AI-free writing workflows are not independently verified.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .91))
    figs.append(('comparison', fig))

    generators = sorted({r['generator'] for r in rows if r['kind'] == 'ai'})
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(generators))
    for i, result in enumerate(results):
        ax.bar(x + (i - 1) * .25,
               [100 * result['by_generator'][g] / 30 for g in generators],
               width=.24, label=result['name'], color=result['color'])
    ax.set_xticks(x, [g.replace('-', '-\n') for g in generators])
    ax.set_ylim(0, 105)
    ax.set_ylabel('AI articles detected (%)')
    ax.set_title('Recall by generator or rewrite method (30 articles each)')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    fig.tight_layout()
    figs.append(('generators', fig))

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = np.array([r['kind'] == 'ai' for r in rows])
    for result in results:
        scores = np.array(result['score'])
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(100*fpr, 100*tpr, label=f"{result['name']} (AUROC {result['auroc']:.3f})",
                color=result['color'], linewidth=2)
    ax.plot([0, 100], [0, 100], color='#888888', linestyle=':', label='Chance ranking')
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Human false-positive rate (%)',
           ylabel='AI recall (%)', title='External article ROC')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    figs.append(('roc', fig))
    with PdfPages(OUT / 'external_articles_v5.pdf') as pdf:
        for stem, fig in figs:
            pdf.savefig(fig)
            fig.savefig(OUT / f'external_articles_v5_{stem}.png', dpi=180)
            plt.close(fig)


def write_markdown(results):
    lines = ['# External article stress test', '',
             'The [Human Detectors dataset](https://github.com/jenna-russell/human_detectors) provides 150 human and 150 AI nonfiction articles. Our 20k model trained on LLMTrace and earlier synthetic sources; these articles are from a separate dataset. The local audit found no exact text or normalized 24-word shingle overlap with training. These are whole-article labels, so they do not test mixed-span localization.', '',
             'Each model uses its previously fixed threshold, calibrated to at most 5% document false alarms on a separate set of 1,120 human documents. Our Qwen article score is its highest token logit; EditLens uses the mean of overlapping native-window scores. The thresholds and score aggregation were chosen before looking at this stress set. No thresholds were retuned here.', '',
             '**Provenance limit:** Source articles have named authors and publication links, but their writing workflows were not independently verified as AI-free. Treat them as attributed-human articles, rather than a gold-standard known-human corpus.', '',
             '| Model | AI detected / 150 | Human false alarms / 150 | Article AUROC |',
             '| --- | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['name']} | {r['ai_detected']}/150 ({100*r['ai_detected']/150:.1f}%) | {r['human_false']}/150 ({100*r['human_false']/150:.1f}%) | {r['auroc']:.3f} |")
    lines += ['', '## AI detection by generator', '',
              '| Generator | Our Qwen 20k | EditLens RoBERTa | EditLens Llama |',
              '| --- | ---: | ---: | ---: |']
    for g in sorted(results[0]['by_generator']):
        lines.append('| ' + g + ' | ' + ' | '.join(f"{r['by_generator'][g]}/30" for r in results) + ' |')
    lines += ['', 'This is a stress test of source transfer, not a representative estimate of deployment accuracy. It contains one domain (nonfiction news articles) and five AI generation/rewrite modes.', '']
    (OUT / 'external_articles_v5.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    docs = load_rows()
    models = load_results(docs)
    render(docs, models)
    write_markdown(models)
