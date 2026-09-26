"""Report coarse mixed-document EditLens transfer beside native Qwen span scores."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

RUNS = Path('/mnt/f/pangram-at-home/runs')
OUT = Path(__file__).resolve().parents[1] / 'reports'
MODELS = [
    ('Our 20k token model', 'qwen3_token_repeat2_v5_20k_e1_local', '#075383', True),
    ('EditLens RoBERTa windows', 'open_pangram_editlens_roberta_span_v5', '#735a9e', False),
    ('EditLens Llama windows', 'open_pangram_editlens_llama_span_v5', '#9b68a6', False),
]
SETS = [('LLMTrace mixed', 'v5_llmtrace_heldout', 'llmtrace_heldout'),
        ('AITDNA mixed', 'v5_aitdna', 'aitdna'),
        ('CoAuthor mixed', 'v5_coauthor', 'coauthor')]


def gather():
    rows = []
    for label, run, color, ours in MODELS:
        for dataset, our_file, pangram_file in SETS:
            path = RUNS / run
            report = (json.loads((path / f'{our_file}.json').read_text()) if ours
                      else json.loads((path / 'summary.json').read_text())['sets'][pangram_file])
            mixed = report['by_kind']['mixed']
            rows.append({'model': label, 'dataset': dataset, 'color': color,
                         'recall': mixed['ai_recall'], 'fpr': mixed['fpr'],
                         'auroc': mixed['roc_auc'], 'ai_tokens': mixed['ai_tokens'],
                         'human_tokens': mixed['human_tokens']})
    return rows


def render(rows):
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.8), sharex='col')
    names = [m[0] for m in MODELS]
    recall_limit = max(105, max(100 * r['recall'] for r in rows) + 12)
    fpr_limit = max(55, max(100 * r['fpr'] for r in rows) + 12)
    for i, (dataset, _, _) in enumerate(SETS):
        group = [r for r in rows if r['dataset'] == dataset]
        y = np.arange(3)
        for j, key in enumerate(('recall', 'fpr')):
            ax = axes[i, j]
            values = [100 * r[key] for r in group]
            ax.barh(y, values, color=[r['color'] for r in group], height=.64)
            for k, value in enumerate(values):
                display = f'{value:.1f}%' if j == 0 else f'{value:.2f}%'
                ax.text(value + 1, k, display, va='center', fontsize=9)
            ax.set_xlim(0, recall_limit if j == 0 else fpr_limit)
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.set_title(f'{dataset}: ' + ('AI token recall' if j == 0 else 'human token FPR'))
    axes[2, 0].set_xlabel('AI tokens highlighted (%)')
    axes[2, 1].set_xlabel('Human tokens falsely highlighted (%)')
    fig.suptitle('Mixed-document transfer: native token model versus EditLens windows', fontsize=14)
    fig.text(.5, .015, 'EditLens window scores are broadcast across each window; these are coarse spans, not native EditLens token predictions.\n'
             'Our 20k model trained on other LLMTrace records. All thresholds use the separate human calibration set.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=1.3, w_pad=2)
    with PdfPages(OUT / 'open_pangram_mixed_v5.pdf') as pdf:
        pdf.savefig(fig)
    fig.savefig(OUT / 'open_pangram_mixed_v5.png', dpi=180)
    plt.close(fig)


def write_markdown(rows):
    lines = ['# Mixed-document comparison with Open Pangram EditLens', '',
             'EditLens is a document/window classifier. Its scores are broadcast across overlapping source-token windows and averaged where windows overlap. This gives a coarse span baseline, not a native EditLens token prediction. Our Qwen model directly predicts token labels. A threshold for each model was frozen using the same separate 1,120 pure-human calibration documents at no more than 5% document-any false highlights.', '',
             '**Interpretation limit:** Our 20k model trained on 15,036 LLMTrace train records; LLMTrace mixed test is from the same corpus, although train/test texts and group IDs are disjoint. AITDNA and CoAuthor are different sources. CoAuthor AI insertions are often very short and every model misses them at these thresholds.', '',
             '| Set | Model | AI-token recall | Human-token FPR | Token AUROC |',
             '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {100*r['recall']:.1f}% | {100*r['fpr']:.2f}% | {r['auroc']:.3f} |")
    lines += ['', 'The two EditLens checkpoints are [RoBERTa-large](https://huggingface.co/pangram/editlens_roberta-large) and [Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B).', '']
    (OUT / 'open_pangram_mixed_v5.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    results = gather()
    render(results)
    write_markdown(results)
