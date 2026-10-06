"""Held-out operating points: fit sentence cutoffs on calibration windows, apply them to the sweep test half.

Inputs (fetched from the hosts into results/calibration/<tag>/): <ckpt>-calibration.npz and <ckpt>-sentences.npz.
For target FPRs 0.5/1/2%, the cutoff is the calibration human-sentence quantile; we report the realized human
FPR on test-half paper sentences and recall on edits (all/small/one/para) and paper_v3 sentences.

Document "any highlight" operating point (from the span-detection v1-v14 work in research/span-detection-20260928):
a human document is a false alarm if *any* of its sentences crosses the cutoff, which is what a reader of a
highlighted report sees. That cutoff is fit on dev-half pure-human rows (max sentence score per row) and applied to
the group-disjoint test half. The calibration windows are not used for it because they are mostly easy human text.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SIZES = {'all': ('sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3'), 'small': ('sentence', 'two_sentence'),
         'one': ('sentence',), 'para': ('paragraph_concise', 'paragraph_v3')}


def cutoff(human_scores, fpr):
    s = np.sort(human_scores)[::-1]; k = int(np.floor(fpr * len(s)))
    return float(s[k]) if k < len(s) else -np.inf


def doc_maxima(rows, sents, split):
    """Max sentence score of each pure-human row with scored sentences in one split."""
    return np.asarray([max(s for s, _ in sl) for r, sl in zip(rows, sents)
                       if r['split'] == split and r['label'] == 'human' and len(sl)], dtype=np.float64)


def collect(rows, sents, split='test'):
    neg, pos, v3p, v3n = [], {k: [] for k in SIZES}, [], []
    for r, sl in zip(rows, sents):
        if r['split'] != split:
            continue
        if r['slot'] in ('edit_context', 'human_untouched', 'human_paper'):
            neg += [s for s, l in sl if l == 0]
        if r['slot'] == 'edit_context':
            for k, conds in SIZES.items():
                if r['condition'] in conds:
                    pos[k] += [s for s, l in sl if l == 1]
        if r['slot'] == 'v3':
            v3p += [s for s, l in sl if l == 1]; v3n += [s for s, l in sl if l == 0]
    return np.asarray(neg), {k: np.asarray(v) for k, v in pos.items()}, np.asarray(v3p), np.asarray(v3n)


def operating_point(t, neg, pos, v3p, v3n, docs):
    return {'cutoff': t, 'test_human_fpr': float(np.mean(neg > t)),
            **{f'recall_{k}': float(np.mean(v > t)) for k, v in pos.items()},
            'v3_recall': float(np.mean(v3p > t)), 'v3_human_fpr': float(np.mean(v3n > t)),
            'test_human_docs_any_highlight': float(np.mean(docs > t)) if len(docs) else None}


def analyze_rows(hum_cal, rows, sents):
    neg, pos, v3p, v3n = collect(rows, sents)
    dev_docs, test_docs = doc_maxima(rows, sents, 'dev'), doc_maxima(rows, sents, 'test')
    out = {}
    for f in (0.005, 0.01, 0.02):
        out[f'{f:.3f}'] = operating_point(cutoff(hum_cal, f), neg, pos, v3p, v3n, test_docs)
        if len(dev_docs):
            out[f'{f:.3f}']['document_any'] = {**operating_point(cutoff(dev_docs, f), neg, pos, v3p, v3n, test_docs),
                                               'dev_human_docs': len(dev_docs), 'test_human_docs': len(test_docs)}
    return out


def analyze(cal_path, sent_path, rows):
    c = np.load(cal_path); z = np.load(sent_path)
    return analyze_rows(c['scores'][c['labels'] == 0], rows, [z[str(i)] for i in range(len(rows))])


if __name__ == '__main__':
    rows = [json.loads(l) for l in gzip.open(HERE / 'sweep-eval-rows.jsonl.gz', 'rt')]
    base = HERE / 'results' / 'calibration'; report = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for cal in sorted(d.glob('*-calibration.npz')):
            ck = cal.name.replace('-calibration.npz', ''); sent = d / f'{ck}-sentences.npz'
            if sent.exists():
                report[f'{d.name}/{ck}'] = analyze(cal, sent, rows)
    (HERE / 'results' / 'calibrated-operating-points.json').write_text(json.dumps(report, indent=1))
    print(f"{'run':32s} {'cut':>6s} {'testFPR':>8s} {'all':>6s} {'small':>6s} {'one':>6s} {'para':>6s} {'v3':>6s} {'v3FPR':>6s} {'docFP':>6s}   (target 1% FPR)")
    for k, v in report.items():
        for name, m in [('sentence', v['0.010']), ('doc-any', v['0.010'].get('document_any'))]:
            if m:
                print(f"{k + ' ' + name:32s} {m['cutoff']:6.3f} {m['test_human_fpr']:8.4f} {m['recall_all']:6.3f} {m['recall_small']:6.3f} {m['recall_one']:6.3f} {m['recall_para']:6.3f} {m['v3_recall']:6.3f} {m['v3_human_fpr']:6.3f} {m['test_human_docs_any_highlight']:6.3f}")
