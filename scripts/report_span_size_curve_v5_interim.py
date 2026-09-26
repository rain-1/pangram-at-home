"""Readable interim plots for complete v4, 5k, 10k and 20k span runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path('/mnt/f/pangram-at-home/runs')
OUT = Path(__file__).resolve().parents[1] / 'reports'
RUNS = [
    ('Earlier v4', 'qwen3_token_repeat2_v4_pilot1', '#a44d38'),
    ('5k', 'qwen3_token_repeat2_v5_5k_e1', '#6ea6c9'),
    ('10k', 'qwen3_token_repeat2_v5_10k_e1', '#317caf'),
    ('20k', 'qwen3_token_repeat2_v5_20k_e1_local', '#075383'),
]


def read(run: str, name: str) -> dict:
    return json.loads((ROOT / run / f'{name}.json').read_text())


def pct(value: float) -> float:
    return 100 * value


def label_bars(ax, values: list[float], fmt: str, suffix: str) -> None:
    for i, value in enumerate(values):
        ax.annotate(f'{value:{fmt}}{suffix}', (i, value), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10)


def heldout() -> plt.Figure:
    reports = [read(run, 'v5_llmtrace_heldout') for _, run, _ in RUNS]
    names = [name for name, _, _ in RUNS]
    colors = [color for _, _, color in RUNS]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    panels = [
        (axes[0, 0], 'AI-token recall, all documents',
         [pct(r['overall']['ai_recall']) for r in reports], '.1f', '%', (0, 100)),
        (axes[0, 1], 'AI-token recall, mixed documents only',
         [pct(r['by_kind']['mixed']['ai_recall']) for r in reports], '.1f', '%', (0, 70)),
        (axes[1, 0], 'Human-token false-positive rate',
         [pct(r['overall']['fpr']) for r in reports], '.4f', '%', (0, 0.8)),
        (axes[1, 1], 'Token AUROC (threshold independent)',
         [r['overall']['roc_auc'] for r in reports], '.3f', '', (0.5, 1.02)),
    ]
    for ax, title, values, fmt, suffix, ylim in panels:
        ax.bar(range(4), values, color=colors, width=.68)
        ax.set_xticks(range(4), names)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        label_bars(ax, values, fmt, suffix)
    axes[0, 0].set_ylabel('Percent of AI tokens caught')
    axes[0, 1].set_ylabel('Percent of AI tokens caught')
    axes[1, 0].set_ylabel('Percent of human tokens flagged')
    axes[1, 1].set_ylabel('AUROC')
    fig.suptitle('Held-out LLMTrace: 2,000 documents / nine domains', fontsize=17, fontweight='bold')
    fig.text(.5, .025, 'Frozen threshold per model: calibrated on the same separate 1,120-document human set.\n'
             'Earlier v4 used a different 5k training mixture; 5k/10k/20k are nested and comparable.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .94), h_pad=2.6)
    return fig


def realism() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    points = []
    for name, run, color in RUNS:
        if name == 'Earlier v4':
            r = read(run, 'aitdna_locked_v1')
        else:
            r = read(run, 'v5_aitdna')
        points.append((name, pct(r['by_kind']['mixed']['fpr']),
                       pct(r['by_kind']['mixed']['ai_recall']), color, 'o'))
    other = [
        ('Token v3', read('qwen3_token_repeat2_v3_pilot1', 'aitdna_locked_v1')['by_kind']['mixed']),
        ('Qwen passage', read('vast_hpo_selected_v3', 'span_v5_aitdna')['by_kind']['mixed']),
        ('Char TF-IDF', json.loads((ROOT / 'span_char_window_baseline_v5/report.json').read_text())['sets']['aitdna']['by_kind']['mixed']),
        ('Word TF-IDF', json.loads((ROOT / 'span_word_window_baseline_v5/report.json').read_text())['sets']['aitdna']['by_kind']['mixed']),
    ]
    for name, result in other:
        points.append((name, pct(result['fpr']), pct(result['ai_recall']), '#7d858a', '^'))
    for name, fpr, recall, color, marker in points:
        size = 145 if name in ('20k', 'Earlier v4') else 105
        ax.scatter(fpr, recall, s=size, c=color, marker=marker, edgecolor='white', linewidth=1.2, zorder=3)
        offsets = {'Earlier v4': (6, -16), '5k': (6, -17), '10k': (6, 6),
                   '20k': (6, -14), 'Token v3': (5, 5), 'Qwen passage': (-98, -14),
                   'Char TF-IDF': (-112, 5), 'Word TF-IDF': (6, -15)}
        ax.annotate(name, (fpr, recall), xytext=offsets.get(name, (6, 6)),
                    textcoords='offset points', fontsize=10, color=color, fontweight='bold' if name == '20k' else 'normal')
    v5 = [p for p in points if p[0] in ('5k', '10k', '20k')]
    ax.plot([p[1] for p in v5], [p[2] for p in v5], color='#317caf', alpha=.5, linewidth=1.5)
    ax.annotate('Better', (2, 96), fontsize=11, color='#267648', fontweight='bold')
    ax.annotate('', xy=(2, 94), xytext=(11, 83), arrowprops={'arrowstyle': '->', 'color': '#267648', 'lw': 1.5})
    ax.set(xlim=(0, 82), ylim=(30, 103), xlabel='Human-token false-positive rate in mixed documents (%)',
           ylabel='AI-token recall in mixed documents (%)')
    ax.set_title('AITDNA: real human–AI collaborative writing (258 mixed documents)',
                 loc='left', fontsize=16, fontweight='bold')
    ax.grid(alpha=.2)
    fig.text(.5, .025, 'Each model uses its own threshold calibrated on the same separate pure-human set.\n'
             'At these thresholds, CoAuthor AI-token recall is 0.0% (5k), 0.6% (10k), and 0.0% (20k).',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    return fig


def roc() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4))
    for name, run, color in RUNS:
        values = np.load(ROOT / run / 'v5_llmtrace_heldout_scores.npz')
        fpr, tpr, _ = roc_curve(values['label'], values['score'])
        for ax in axes:
            ax.plot(100 * fpr, 100 * tpr, label=name, color=color, lw=2.2)
    axes[0].plot([0, 100], [0, 100], ':', color='#8a8a8a', label='Random ranking')
    axes[0].set(xlim=(0, 100), ylim=(0, 100), title='Full range')
    axes[1].set(xlim=(0, 2), ylim=(0, 100), title='Zoom: low false-positive rates')
    for ax in axes:
        ax.set_xlabel('Human-token false-positive rate (%)')
        ax.set_ylabel('AI-token recall (%)')
        ax.grid(alpha=.2)
    axes[1].legend(loc='lower right', frameon=True)
    fig.suptitle('LLMTrace ROC: available recall / false-positive tradeoff', fontsize=16, fontweight='bold')
    fig.text(.5, .02, 'These curves use test-set scores to show possible tradeoffs; threshold choices on this plot are diagnostic, not deployable.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .93))
    return fig


def main() -> None:
    OUT.mkdir(exist_ok=True)
    figures = [('heldout', heldout()), ('realism', realism()), ('roc', roc())]
    with PdfPages(OUT / 'span_size_curve_v5_interim.pdf') as pdf:
        for name, fig in figures:
            fig.savefig(OUT / f'span_size_curve_v5_interim_{name}.png', dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
    print(OUT / 'span_size_curve_v5_interim.pdf')


if __name__ == '__main__':
    main()
