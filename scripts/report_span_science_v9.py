"""Readable v9 comparison against older Qwen and open Pangram article baselines."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path('/mnt/f/pangram-at-home')
REPORTS = Path(__file__).resolve().parents[1]/'reports'
RUNS = ROOT/'runs'
QWEN = (
    ('Qwen balanced v6', 'qwen3_token_repeat2_balanced_v6_20k', 'v6_external_articles', '#5d7285'),
    ('Qwen publication v8', 'qwen3_token_repeat2_publication_v8_20k', 'v8_external_articles', '#d5873b'),
    ('Qwen paired science v9', 'qwen3_token_repeat2_science_paired_v9_20k', 'v9_external_articles', '#168f83'),
)
OPEN = (
    ('Pangram RoBERTa', 'open_pangram_editlens_roberta_v5', '#9156a8'),
    ('Pangram Llama', 'open_pangram_editlens_llama_v5', '#b95f88'),
)


def qwen_document_scores(run, stem):
    with np.load(RUNS/run/(stem+'_scores.npz')) as data:
        scores, labels, offsets = (data[k] for k in ('score', 'label', 'document_offsets'))
        values = np.array([scores[offsets[i]:offsets[i+1]].max()
                           for i in range(len(offsets)-1)])
        y = np.array([int(labels[offsets[i]:offsets[i+1]].mean() > .5)
                      for i in range(len(offsets)-1)])
        if len(y) != 300 or int(y.sum()) != 150:
            raise ValueError('Expected 150 human and 150 AI external articles')
        return y, values


def open_document_scores(run):
    rows = [json.loads(line) for line in (RUNS/run/'external_human_detectors.jsonl').open()]
    y = np.array([int(row['kind'] == 'ai') for row in rows])
    score = np.array([row['score'] for row in rows])
    if len(y) != 300 or int(y.sum()) != 150:
        raise ValueError('Expected 150 human and 150 AI external articles')
    return y, score


def qwen_external_metrics(run, stem):
    report = json.loads((RUNS/run/(stem+'.json')).read_text())
    threshold = report['threshold']
    with np.load(RUNS/run/(stem+'_scores.npz')) as data:
        score, label, offsets = (data[k] for k in ('score', 'label', 'document_offsets'))
        human_any = human_substantial = ai_any = 0
        for i in range(len(offsets)-1):
            a, b = offsets[i:i+2]
            selected = score[a:b] >= threshold
            if np.all(label[a:b] == 0):
                human_any += selected.any()
                human_substantial += selected.mean() >= .10
            elif np.all(label[a:b] == 1):
                ai_any += selected.any()
        return {'human_any': int(human_any), 'human_10pct': int(human_substantial),
                'ai_any': int(ai_any), 'human_token_fpr': report['overall']['fpr'],
                'ai_token_recall': report['overall']['ai_recall'],
                'threshold': threshold}


def report_json(run, stem):
    return json.loads((RUNS/run/(stem+'.json')).read_text())


def publisher_false_alarms(run, stem):
    rows = [json.loads(line) for line in (ROOT/'data/span_ai_eval_candidate_v1/test.jsonl').open()]
    threshold = report_json(run, stem)['threshold']
    counts = {}
    with np.load(RUNS/run/(stem+'_scores.npz')) as data:
        scores, offsets = data['score'], data['document_offsets']
        assert len(rows) == len(offsets)-1
        for i, row in enumerate(rows):
            if row['kind'] != 'human':
                continue
            publisher = row['publication'].replace('Readers Digest', "Reader's Digest")
            flagged = bool((scores[offsets[i]:offsets[i+1]] >= threshold).any())
            counts.setdefault(publisher, [0,0])
            counts[publisher][0] += int(flagged)
            counts[publisher][1] += 1
    return counts


def main():
    REPORTS.mkdir(exist_ok=True)
    qwen = []
    for name, run, stem, color in QWEN:
        metrics = qwen_external_metrics(run, stem)
        y, score = qwen_document_scores(run, stem)
        metrics['roc_auc'] = roc_auc_score(y, score)
        qwen.append((name, metrics, y, score, color))
    opened = [(name, *open_document_scores(run), color) for name, run, color in OPEN]
    open_reports = [json.loads((RUNS/run/'external_human_detectors_summary.json').read_text())
                    for _, run, _ in OPEN]
    model_labels = [name for name, *_ in qwen]+[name for name, *_ in opened]
    false_alarms = [m['human_any'] for _, m, *_ in qwen]+[
        report['overall']['human_false_alarms'] for report in open_reports]
    ai_caught = [m['ai_any'] for _, m, *_ in qwen]+[
        report['overall']['ai_detected'] for report in open_reports]
    colors = [row[-1] for row in qwen+opened]
    v8_run = QWEN[1][1]
    v9_run = QWEN[2][1]
    human_sets = [
        ('External articles', 'v8_external_articles', 'v9_external_articles'),
        ('Archived EPA', 'v9_archived_epa_human_v8', 'v9_archived_epa_human'),
        ('Archived magazine', 'v10_magazine_test_v8', 'v10_magazine_test_v9'),
        ('AITDNA collaboration', 'v8_aitdna', 'v9_aitdna'),
    ]
    fpr = []
    for label, old, new in human_sets:
        fpr.append((label, 100*report_json(v8_run, old)['overall']['fpr'],
                    100*report_json(v9_run, new)['overall']['fpr']))
    mixed_sets = [
        ('LLMTrace', 'v8_llmtrace_heldout', 'v9_llmtrace_heldout'),
        ('AITDNA', 'v8_aitdna', 'v9_aitdna'),
        ('CoAuthor', 'v8_coauthor', 'v9_coauthor'),
    ]
    recall = [(label, 100*report_json(v8_run, old)['overall']['ai_recall'],
               100*report_json(v9_run, new)['overall']['ai_recall'])
              for label, old, new in mixed_sets]
    publishers_v8 = publisher_false_alarms(v8_run, 'v8_external_articles')
    publishers_v9 = publisher_false_alarms(v9_run, 'v9_external_articles')
    assert publishers_v8.keys() == publishers_v9.keys()
    pdf = REPORTS/'science_paired_v9_comparison.pdf'
    with PdfPages(pdf) as pages:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
        x = np.arange(len(model_labels))
        for ax, vals, title, target in (
            (axes[0], false_alarms, 'Human articles flagged (of 150)', 15),
            (axes[1], ai_caught, 'AI articles caught (of 150)', 142.5)):
            ax.bar(x, vals, color=colors, width=.7)
            ax.axhline(target, ls='--', color='#333333', lw=1)
            ax.set_xticks(x, model_labels, rotation=30, ha='right')
            ax.set_ylim(0, 160)
            ax.set_ylabel('Articles')
            ax.set_title(title, loc='left')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            for i, value in enumerate(vals):
                ax.text(i, value+3, str(value), ha='center', fontsize=9)
        fig.suptitle('External article stress test: frozen operating points', fontsize=15, x=.06, ha='left')
        fig.text(.06, .01, 'Dashed goals: ≤10% human false alarms and ≥95% AI article recall. '
                 'Qwen flags an article if any token is highlighted; Pangram baselines make whole-article decisions.',
                 fontsize=8)
        fig.tight_layout(rect=[0,.055,1,.94]);pages.savefig(fig);plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for name, metrics, y, score, color in qwen:
            curve = roc_curve(y, score)
            axes[0].plot(curve[0], curve[1], label=f'{name} ({roc_auc_score(y,score):.4f})',
                         color=color, lw=2)
            axes[1].plot(curve[0], curve[1], label=name, color=color, lw=2)
        for name, y, score, color in opened:
            curve = roc_curve(y, score)
            axes[0].plot(curve[0], curve[1], label=f'{name} ({roc_auc_score(y,score):.4f})',
                         color=color, lw=2)
            axes[1].plot(curve[0], curve[1], label=name, color=color, lw=2)
        axes[0].set(xlim=(0,1), ylim=(0,1), xlabel='Human article false-positive rate',
                    ylabel='AI article recall', title='Full ROC')
        axes[1].set(xlim=(0,.2), ylim=(.75,1), xlabel='Human article false-positive rate',
                    ylabel='AI article recall', title='Low false-positive region')
        for ax in axes:
            ax.grid(alpha=.2);ax.legend(fontsize=7, loc='lower right')
        fig.suptitle('External article discrimination across decision thresholds', fontsize=15, x=.06, ha='left')
        fig.tight_layout(rect=[0,0,1,.94]);pages.savefig(fig);plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, values, title, ylabel in ((axes[0], fpr, 'Human tokens falsely highlighted', 'False-positive rate (%)'),
                                           (axes[1], recall, 'AI tokens highlighted', 'AI-token recall (%)')):
            pos = np.arange(len(values));width=.36
            ax.bar(pos-width/2, [r[1] for r in values], width, color=QWEN[1][-1], label='v8')
            ax.bar(pos+width/2, [r[2] for r in values], width, color=QWEN[2][-1], label='v9')
            ax.set_xticks(pos, [r[0] for r in values], rotation=20, ha='right')
            ax.set_ylabel(ylabel);ax.set_title(title, loc='left');ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True);ax.legend()
        fig.suptitle('Frozen-threshold transfer across domains', fontsize=15, x=.06, ha='left')
        fig.text(.06,.01,'Archived magazine test authors were excluded from its training candidates. '
                 'AITDNA and CoAuthor contain genuinely mixed authorship; their AI passages differ in length.',
                 fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.94]);pages.savefig(fig);plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        names = sorted(publishers_v8)
        pos = np.arange(len(names));width=.36
        old = [100*publishers_v8[name][0]/publishers_v8[name][1] for name in names]
        new = [100*publishers_v9[name][0]/publishers_v9[name][1] for name in names]
        ax.barh(pos-width/2, old, width, color=QWEN[1][-1], label='v8')
        ax.barh(pos+width/2, new, width, color=QWEN[2][-1], label='v9')
        ax.axvline(10, color='#333333', lw=1, ls='--', label='10% goal')
        ax.set_yticks(pos, names);ax.invert_yaxis()
        ax.set_xlim(0, 100);ax.set_xlabel('Human articles with any false highlight (%)')
        ax.set_title('False alarms by publisher on external articles', loc='left')
        ax.legend();ax.grid(axis='x', alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout();pages.savefig(fig);plt.close(fig)

    markdown = [
        '# Paired science v9 comparison', '',
        'The v9 run changes paired science article windows while retaining the v8 architecture, '
        'tuned hyperparameters, 20k-document scale, and existing mixed examples. Thresholds for '
        'each Qwen model were selected on the same separate human-calibration protocol, not these '
        'article tests.', '',
        '| Model | External human articles flagged | External AI articles caught | Document AUROC | '
        'External human-token FPR |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, metrics, *_ in qwen:
        markdown.append(f'| {name} | {metrics["human_any"]}/150 | {metrics["ai_any"]}/150 | '
                        f'{metrics["roc_auc"]:.4f} | {metrics["human_token_fpr"]:.2%} |')
    for (name, _, _), baseline in zip(OPEN, open_reports):
        item = baseline['overall']
        markdown.append(f'| {name} | {item["human_false_alarms"]}/150 | '
                        f'{item["ai_detected"]}/150 | {item["auroc"]:.4f} | — |')
    markdown += ['',
        'An article is counted as falsely flagged if Qwen highlights any human token. '
        'This is sensitive to isolated errors; Pangram makes a whole-article decision. '
        'The AI-generated article count is analogously any highlighted token.', '',
        '## Human false-highlight rate by source', '',
        '| Source | v8 | v9 |', '|---|---:|---:|',
    ]
    markdown += [f'| {label} | {old:.2f}% | {new:.2f}% |' for label, old, new in fpr]
    markdown += ['', '## Mixed-document AI-token recall', '',
                 '| Source | v8 | v9 |', '|---|---:|---:|']
    markdown += [f'| {label} | {old:.1f}% | {new:.1f}% |' for label, old, new in recall]
    markdown += ['', '## External human false alarms by publisher', '',
                 '| Publisher | v8 | v9 |', '|---|---:|---:|']
    for publisher in sorted(publishers_v8):
        old, n_old = publishers_v8[publisher]
        new, n_new = publishers_v9[publisher]
        assert n_old == n_new
        markdown.append(f'| {publisher} | {old}/{n_old} | {new}/{n_new} |')
    markdown += ['', 'The external Human Detectors human labels have attributed bylines but no independently '
                 'verified AI-free workflow. The archived EPA and magazine extracts were captured before 2023. '
                 'External article results have informed development and are no longer a pristine blind test. '
                 'The magazine test is author-exclusive from its candidate training pool. '
                 'The original Human Detectors source IDs repeat, although all 300 text hashes are distinct '
                 'across 150 source article URLs; numeric results use row order, and new prediction exports include row index and '
                 'text hash for unambiguous case review.', '',
                 f'[Download the comparison charts]({pdf.name})', '']
    (REPORTS/'science_paired_v9_comparison.md').write_text('\n'.join(markdown))
    print(pdf)


if __name__ == '__main__':
    main()
