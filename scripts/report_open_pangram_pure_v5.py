"""Compare our span models and both open Pangram EditLens models on pure docs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path('/mnt/f/pangram-at-home/runs')
DATA = Path('/mnt/f/pangram-at-home/data/span_size_curve_v5/size_20000/test_llmtrace.jsonl')
OUT = Path(__file__).resolve().parents[1] / 'reports'
OUR = [
    ('Earlier v4', 'qwen3_token_repeat2_v4_pilot1', '#a44d38'),
    ('5k × 1', 'qwen3_token_repeat2_v5_5k_e1', '#72a9c9'),
    ('10k × 1', 'qwen3_token_repeat2_v5_10k_e1', '#337eae'),
    ('20k × 1', 'qwen3_token_repeat2_v5_20k_e1_local', '#075383'),
    ('5k × 4', 'qwen3_token_repeat2_v5_5k_e4_local', '#9c7a3c'),
]
PANGRAM = [
    ('EditLens RoBERTa', 'open_pangram_editlens_roberta_v5', '#735a9e'),
    ('EditLens Llama', 'open_pangram_editlens_llama_v5', '#9b68a6'),
]


def our_results(kinds: dict[str, str]) -> list[dict]:
    result = []
    for label, run, color in OUR:
        folder = ROOT / run
        report = json.loads((folder / 'v5_llmtrace_heldout.json').read_text())
        with np.load(folder / 'v5_llmtrace_heldout_scores.npz') as scores:
            values = scores['score']
            offsets = scores['document_offsets']
            ids = scores['document_ids']
        pure = [(kinds[str(id_)], float(values[offsets[i]:offsets[i + 1]].max()))
                for i, id_ in enumerate(ids) if str(id_) in kinds]
        labels = np.array([kind == 'ai' for kind, _ in pure], dtype=bool)
        values = np.array([score for _, score in pure])
        threshold = float(np.float32(report['threshold']))
        locked_file = 'v4_human_locked_test.json' if label == 'Earlier v4' else 'v5_human_locked_test.json'
        result.append({'name': label, 'group': 'Our token model', 'color': color,
                       'ai_detected': int(np.sum(labels & (values >= threshold))),
                       'human_false': report['by_kind']['human']['pure_human_documents_with_false_highlight'],
                       'auroc': float(roc_auc_score(labels, values)),
                       'threshold_rule': 'any token exceeds separately calibrated threshold',
                       'locked_human_false': json.loads((folder / locked_file).read_text())['overall']['pure_human_documents_with_false_highlight']})
    return result


def pangram_results() -> list[dict]:
    result = []
    for label, run, color in PANGRAM:
        summary = json.loads((ROOT / run / 'summary.json').read_text())
        heldout = summary['sets']['llmtrace_pure']
        result.append({'name': label, 'group': 'Open Pangram EditLens', 'color': color,
                       'ai_detected': heldout['ai_detected'],
                       'human_false': heldout['human_false_alarms'],
                       'auroc': heldout['auroc'],
                       'threshold_rule': 'mean native-window score exceeds separately calibrated threshold',
                       'locked_human_false': summary['sets']['locked_human']['human_false_alarms'],
                       'synthetic_ai_detected': summary['sets']['synthetic_v4_pure']['ai_detected'],
                       'synthetic_human_false': summary['sets']['synthetic_v4_pure']['human_false_alarms'],
                       'revision': summary['revision']})
    return result


def render(rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    y = np.arange(len(rows))
    colors = [r['color'] for r in rows]
    names = [r['name'] for r in rows]
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
    recall = [100 * r['ai_detected'] / 516 for r in rows]
    fpr = [100 * r['human_false'] / 720 for r in rows]
    axes[0].barh(y, recall, color=colors, height=.68)
    axes[1].barh(y, fpr, color=colors, height=.68)
    for i, row in enumerate(rows):
        axes[0].text(recall[i] + 1, i, f"{row['ai_detected']}/516", va='center', fontsize=9)
        axes[1].text(fpr[i] + .08, i, f"{row['human_false']}/720", va='center', fontsize=9)
    axes[0].set(xlim=(0, 110), xlabel='Fully AI documents detected (%)', title='AI-document recall')
    axes[1].set(xlim=(0, max(2.5, max(fpr) + .8)), xlabel='Fully human documents falsely flagged (%)',
                title='Human-document false-positive rate')
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Pure-document detection: our size sweep and Open Pangram', fontsize=16, fontweight='bold')
    fig.text(.5, .025, 'Held-out LLMTrace: 516 AI + 720 human documents. All thresholds set on the same separate 1,120 human documents\n'
             'with ≤5% false-alarm allowance. Our detector flags a document if any token is highlighted; EditLens averages native-window scores.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .92), w_pad=2)
    figures.append(('comparison', fig))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    auc = [r['auroc'] for r in rows]
    locked = [100 * r['locked_human_false'] / 3579 for r in rows]
    axes[0].barh(y, auc, color=colors, height=.68)
    axes[1].barh(y, locked, color=colors, height=.68)
    for i, row in enumerate(rows):
        axes[0].text(auc[i] + .005, i, f"{auc[i]:.3f}", va='center', fontsize=9)
        axes[1].text(locked[i] + .025, i, f"{row['locked_human_false']}/3579", va='center', fontsize=9)
    axes[0].set(xlim=(.5, 1.06), xlabel='Document AUROC (higher is better)', title='Ranking quality on the pure held-out set')
    axes[1].set(xlim=(0, max(.7, max(locked) + .2)),
                xlabel='Locked human documents falsely flagged (%)', title='Additional human-only stress test')
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Threshold-independent ranking and held-out human false alarms', fontsize=15, fontweight='bold')
    fig.text(.5, .025, 'AUROC uses each model’s document score: maximum token score for ours, mean window score for EditLens.\n'
             'Zero observed false alarms is an estimate on these datasets, not a guarantee for arbitrary human text.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .9), w_pad=2)
    figures.append(('ranking', fig))

    with PdfPages(OUT / 'open_pangram_comparison_v5.pdf') as pdf:
        for stem, fig in figures:
            fig.savefig(OUT / f'open_pangram_comparison_v5_{stem}.png', dpi=180)
            pdf.savefig(fig)
            plt.close(fig)


def write_markdown(rows: list[dict]) -> None:
    lines = ['# Open Pangram pure-document comparison', '',
             'Open Pangram EditLens RoBERTa-large and Llama-3.2-3B are four-bucket models that score the extent of AI intervention. Their score is the expected bucket divided by three, following Pangram’s inference code. We average scores across overlapping native-token windows (512 for RoBERTa; 1024 for Llama). The checkpoints are licensed CC BY-NC-SA 4.0 for noncommercial use.', '',
             'Every model’s threshold is chosen on the same separate 1,120-document pure-human calibration set to allow at most 5% document-level false alarms. On our token model, a document is positive if any token is highlighted; on EditLens, its mean window score must cross the threshold. Both are then evaluated without retuning.', '',
             '| Model | AI detected (516) | Human falsely flagged (720) | Pure-document AUROC | Locked human falsely flagged (3,579) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['name']} | {row['ai_detected']}/516 ({100*row['ai_detected']/516:.1f}%) | {row['human_false']}/720 ({100*row['human_false']/720:.2f}%) | {row['auroc']:.3f} | {row['locked_human_false']}/3,579 |")
    lines += ['', 'The LLMTrace pure set is a different named corpus from the published EditLens training dataset. Source-level overlap has not been fully audited. The older synthetic evaluation may be closer to EditLens training data, so LLMTrace is the primary comparison here. The pure-document task is easier than localizing AI spans within mixed documents.', '',
              'EditLens has no native token labels. This report does not compare EditLens against our token recall; that would require explicitly broadcasting its window scores to tokens and evaluating the resulting coarse spans.', '',
              '## Additional shared synthetic pure-document result for EditLens', '',
              '| Model | AI detected (148) | Human falsely flagged (152) | AUROC |',
              '| --- | ---: | ---: | ---: |']
    for row in rows:
        if 'synthetic_ai_detected' in row:
            d = json.loads((ROOT / ('open_pangram_editlens_roberta_v5' if 'RoBERTa' in row['name'] else 'open_pangram_editlens_llama_v5') / 'summary.json').read_text())['sets']['synthetic_v4_pure']
            lines.append(f"| {row['name']} | {row['synthetic_ai_detected']}/148 | {row['synthetic_human_false']}/152 | {d['auroc']:.3f} |")
    lines += ['', 'Checkpoint sources: [RoBERTa-large](https://huggingface.co/pangram/editlens_roberta-large), [Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B), [upstream inference](https://github.com/pangramlabs/EditLens/blob/05a588f15d792330ccaf91be8ee4fdb54ce26835/scripts/inference.py).', '']
    (OUT / 'open_pangram_comparison_v5.md').write_text('\n'.join(lines))


def main():
    kinds = {row['id']: row['kind'] for line in DATA.open() if (row := json.loads(line))['kind'] in ('human', 'ai')}
    rows = our_results(kinds) + pangram_results()
    render(rows)
    write_markdown(rows)
    print(OUT / 'open_pangram_comparison_v5.pdf')


if __name__ == '__main__':
    main()
