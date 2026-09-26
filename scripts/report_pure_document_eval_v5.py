"""Explain fully human and fully AI document evaluation with matched baselines."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path('/mnt/f/pangram-at-home/runs')
DATA = Path('/mnt/f/pangram-at-home/data/span_size_curve_v5/size_20000/test_llmtrace.jsonl')
OUT = Path(__file__).resolve().parents[1] / 'reports'
RUNS = [
    ('Earlier v4', 'qwen3_token_repeat2_v4_pilot1', '#a44d38'),
    ('5k × 1', 'qwen3_token_repeat2_v5_5k_e1', '#72a9c9'),
    ('10k × 1', 'qwen3_token_repeat2_v5_10k_e1', '#337eae'),
    ('20k × 1', 'qwen3_token_repeat2_v5_20k_e1_local', '#075383'),
    ('5k × 4', 'qwen3_token_repeat2_v5_5k_e4_local', '#9c7a3c'),
]


def read(run, file):
    return json.loads((ROOT / run / (file + '.json')).read_text())


def doc_hits(run: str, threshold: float, kind: dict[str, str]) -> int:
    with np.load(ROOT / run / 'v5_llmtrace_heldout_scores.npz') as values:
        scores = values['score']
        offsets = values['document_offsets']
        ids = values['document_ids']
    return sum(bool((scores[offsets[i]:offsets[i + 1]] >= threshold).any())
               for i, id_ in enumerate(ids) if kind.get(str(id_)) == 'ai')


def heldout_figure() -> tuple[plt.Figure, list[dict]]:
    kind = {row['id']: row['kind'] for line in DATA.open() if (row := json.loads(line))['kind'] in ('human', 'ai')}
    rows = []
    for name, run, color in RUNS:
        report = read(run, 'v5_llmtrace_heldout')
        rows.append({'name': name, 'color': color,
                     'token_recall': 100 * report['by_kind']['ai']['ai_recall'],
                     'ai_doc_hits': doc_hits(run, report['threshold'], kind),
                     'human_doc_false': report['by_kind']['human']['pure_human_documents_with_false_highlight']})
    names = [row['name'] for row in rows]
    colors = [row['color'] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    values = [row['token_recall'] for row in rows]
    axes[0].bar(range(len(rows)), values, color=colors)
    for i, value in enumerate(values):
        axes[0].text(i, value + 1.2, f'{value:.1f}%', ha='center', fontsize=9)
    axes[0].set(title='Fully AI: tokens highlighted', ylim=(0, 105), ylabel='AI-token recall (%)')
    values = [row['ai_doc_hits'] / 516 * 100 for row in rows]
    axes[1].bar(range(len(rows)), values, color=colors)
    for i, (value, row) in enumerate(zip(values, rows)):
        axes[1].text(i, value + 1.2, f"{row['ai_doc_hits']}/516", ha='center', fontsize=9)
    axes[1].set(title='Fully AI: documents detected', ylim=(0, 112), ylabel='Documents with ≥1 AI highlight (%)')
    values = [row['human_doc_false'] / 720 * 100 for row in rows]
    axes[2].bar(range(len(rows)), values, color=colors)
    for i, (value, row) in enumerate(zip(values, rows)):
        axes[2].text(i, value + .018, f"{row['human_doc_false']}/720", ha='center', fontsize=9)
    axes[2].set(title='Fully human: false alarms', ylim=(0, .48), ylabel='Documents with ≥1 false highlight (%)')
    for ax in axes:
        ax.set_xticks(range(len(rows)), names, rotation=30, ha='right')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Held-out LLMTrace: separate the fully human and fully AI documents', fontsize=16, fontweight='bold')
    fig.text(.5, .025, '516 fully AI and 720 fully human documents; nine domains. Each threshold was set on a separate 1,120-document human calibration set.\n'
             'Detecting ≥1 token is a permissive document rule; token recall measures how much AI text was actually highlighted.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .9), w_pad=2.3)
    return fig, rows


def baseline_figure() -> tuple[plt.Figure, list[dict]]:
    entries = []
    for name, run, color in RUNS:
        d = read(run, 'v5_prior_synthetic_val') if name != 'Earlier v4' else read(run, 'v4_synthetic_val')
        entries.append((name, d, color))
    entries += [
        ('Token v3', read('qwen3_token_repeat2_v3_pilot1', 'v4_synthetic_val'), '#91999e'),
        ('Qwen passage', read('vast_hpo_selected_v3', 'span_v5_synthetic_v4_val'), '#91999e'),
    ]
    for label, run in [('Char TF-IDF', 'span_char_window_baseline_v5'),
                       ('Word TF-IDF', 'span_word_window_baseline_v5')]:
        d = json.loads((ROOT / run / 'report.json').read_text())['sets']['synthetic_v4_val']
        entries.append((label, d, '#91999e'))
    rows = [{'name': label, 'color': color,
             'ai_recall': 100 * d['by_kind']['ai']['ai_recall'],
             'human_false': d['by_kind']['human']['pure_human_documents_with_false_highlight']}
            for label, d, color in entries]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.4), sharey=True)
    rec = [r['ai_recall'] for r in rows]
    false = [100 * r['human_false'] / 152 for r in rows]
    axes[0].barh(y, rec, color=[r['color'] for r in rows], height=.68)
    axes[1].barh(y, false, color=[r['color'] for r in rows], height=.68)
    for i, (value, row) in enumerate(zip(rec, rows)):
        axes[0].text(value + 1.2, i, f'{value:.1f}%', va='center', fontsize=9)
    for i, (value, row) in enumerate(zip(false, rows)):
        axes[1].text(value + .12, i, f"{row['human_false']}/152", va='center', fontsize=9)
    axes[0].set(xlim=(0, 110), xlabel='Share of AI tokens highlighted (%)', title='Fully AI: token recall')
    axes[1].set(xlim=(0, 7), xlabel='Fully human docs with any false highlight (%)', title='Fully human: false alarms')
    axes[0].set_yticks(y, [r['name'] for r in rows])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Shared v4 synthetic evaluation: size sweep versus baselines', fontsize=16, fontweight='bold')
    fig.text(.5, .03, 'Same 148 fully AI and 152 fully human documents for every model. Each model uses its own threshold\n'
             'calibrated to 5% document-any false highlight on the same separate human calibration set.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .92), w_pad=2.4)
    return fig, rows


def main():
    OUT.mkdir(exist_ok=True)
    figures = [heldout_figure(), baseline_figure()]
    names = ['heldout', 'baselines']
    with PdfPages(OUT / 'pure_document_eval_v5.pdf') as pdf:
        for (fig, _), name in zip(figures, names):
            fig.savefig(OUT / f'pure_document_eval_v5_{name}.png', dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    lines = ['# Fully human and fully AI evaluation', '',
             'AI-token recall means the fraction of tokens known to be AI that receive an AI highlight. It is not balanced accuracy or document accuracy. A random model that highlights half the AI tokens would also highlight about half the human tokens, while these models are calibrated to highlight few human tokens.', '',
             '## Held-out LLMTrace pure documents', '',
             '| Model | AI tokens highlighted, 516 pure AI docs | AI docs with any highlight | Human docs with any false highlight |',
             '| --- | ---: | ---: | ---: |']
    for r in figures[0][1]:
        lines.append(f"| {r['name']} | {r['token_recall']:.1f}% | {r['ai_doc_hits']}/516 ({100*r['ai_doc_hits']/516:.1f}%) | {r['human_doc_false']}/720 |")
    lines += ['', 'The any-highlight rule is deliberately permissive. The token-recall column shows whether the model highlights most of an AI document rather than a stray word. The older v4 used a different training mixture; the 5k, 10k, and 20k tiers are nested. The 5k × 4 trial has the same optimizer-step budget as 20k × 1.', '',
              '## Shared synthetic v4 evaluation: models and baselines', '',
              '| Model | AI-token recall, 148 pure AI docs | Human docs with any false highlight, 152 pure human docs |',
              '| --- | ---: | ---: |']
    for r in figures[1][1]:
        lines.append(f"| {r['name']} | {r['ai_recall']:.1f}% | {r['human_false']}/152 |")
    lines += ['', 'The Qwen passage baseline has high pure-AI recall, but its previously measured human-token FPR on mixed documents is much larger. Pure-document performance alone is therefore insufficient for the span-localization goal. The held-out LLMTrace test has no TF-IDF or Qwen passage baseline scores yet, so those comparisons use the shared synthetic evaluation.', '']
    (OUT / 'pure_document_eval_v5.md').write_text('\n'.join(lines))
    print(OUT / 'pure_document_eval_v5.pdf')


if __name__ == '__main__':
    main()
