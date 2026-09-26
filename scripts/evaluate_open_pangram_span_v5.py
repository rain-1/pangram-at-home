"""Broadcast Open Pangram EditLens window scores for a coarse span baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate_open_pangram_pure_v5 import MODELS, ROOT, clean_text, load_model
from span_data import encode_document, window_starts
from span_metrics import calibrate_threshold, summarize_scores

SETS = {
    'calibration': ('span_human_eval_v2', 'calibration.jsonl'),
    'llmtrace_heldout': ('span_size_curve_v5/size_20000', 'test_llmtrace.jsonl'),
    'synthetic_v4_val': ('span_training_v4', 'val.jsonl'),
    'locked_human': ('span_human_eval_v2', 'test.jsonl'),
    'aitdna': ('span_sources_v5/normalized_aitdna_real', 'locked_test.jsonl'),
    'coauthor': ('span_realistic_eval_v1', 'test.jsonl'),
}
WINDOW = {'roberta': (384, 192), 'llama': (512, 256)}


def score_rows(path: Path, qwen_tokenizer, tokenizer, model, which: str):
    rows = [json.loads(line) for line in path.open()]
    width, stride = WINDOW[which]
    prepared = []
    passages = []
    assignments = []
    for index, row in enumerate(rows):
        ids, offsets, labels = encode_document(row, qwen_tokenizer)
        prepared.append({'row': row, 'label': np.asarray(labels, dtype=np.int16),
                         'score': np.zeros(len(ids), dtype=np.float64),
                         'counts': np.zeros(len(ids), dtype=np.int16)})
        for start in window_starts(len(ids), width, stride):
            end = min(start + width, len(ids))
            passages.append(clean_text(row['text'][offsets[start][0]:offsets[end - 1][1]]))
            assignments.append((index, start, end))
    batch_size = MODELS[which]['batch_size']
    max_length = MODELS[which]['max_length']
    truncated = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for base in range(0, len(passages), batch_size):
            source = passages[base:base + batch_size]
            native_lengths = tokenizer(source, add_special_tokens=True, truncation=False)['input_ids']
            truncated += sum(len(ids) > max_length for ids in native_lengths)
            encoded = tokenizer(source, truncation=True, max_length=max_length,
                                padding=True, return_tensors='pt').to(device)
            logits = model(**encoded).logits.float()
            probs = logits.softmax(-1)
            positions = torch.arange(4, device=probs.device, dtype=probs.dtype) / 3
            scores = (probs @ positions).cpu().numpy()
            for (index, start, end), score in zip(assignments[base:base + batch_size], scores):
                prepared[index]['score'][start:end] += float(score)
                prepared[index]['counts'][start:end] += 1
            if base // batch_size % 100 == 0:
                print(f'{path.name}: {min(base + batch_size, len(passages))}/{len(passages)} windows', flush=True)
    for result in prepared:
        if not np.all(result['counts'] > 0):
            raise RuntimeError('Unscored source tokens')
        result['score'] /= result.pop('counts')
    return prepared, len(passages), truncated


def save_scores(path: Path, results) -> None:
    labels = [r['label'][r['label'] != -100] for r in results]
    scores = [r['score'][r['label'] != -100] for r in results]
    offsets = np.r_[0, np.cumsum([len(x) for x in labels])]
    np.savez_compressed(path, label=np.concatenate(labels).astype(np.int8),
                        score=np.concatenate(scores).astype(np.float32),
                        document_offsets=offsets,
                        document_ids=np.asarray([r['row']['id'] for r in results]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--sets', nargs='+', choices=SETS, default=list(SETS))
    args = p.parse_args()
    if 'calibration' not in args.sets:
        raise ValueError('Calibration set is required')
    out = ROOT / 'runs' / f'open_pangram_editlens_{args.model}_span_v5'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model)
    qwen = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3-1.7B')
    reports = {}
    threshold = None
    for name in args.sets:
        folder, filename = SETS[name]
        rows, windows, truncated = score_rows(ROOT / 'data' / folder / filename,
                                              qwen, tokenizer, model, args.model)
        if threshold is None:
            threshold = calibrate_threshold(rows, .05, 'document')
        overall = summarize_scores(rows, threshold)
        by_kind = {kind: summarize_scores([r for r in rows if r['row']['kind'] == kind], threshold)
                   for kind in ('human', 'ai', 'mixed')}
        report = {'set': name, 'file': f'{folder}/{filename}', 'threshold': threshold,
                  'windows': windows, 'truncated_native_windows': truncated,
                  'truncated_native_window_fraction': truncated / windows,
                  'overall': overall, 'by_kind': by_kind,
                  'by_domain': {domain: summarize_scores([r for r in rows if r['row']['domain'] == domain], threshold)
                                for domain in sorted({r['row']['domain'] for r in rows})}}
        reports[name] = report
        (out / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
        save_scores(out / f'{name}_scores.npz', rows)
        print(name, overall['ai_recall'], overall['fpr'], flush=True)
    summary = {'model': MODELS[args.model]['source'], 'revision': MODELS[args.model]['revision'],
               'task': 'coarse token scores from native EditLens window classification',
               'window_qwen_tokens': WINDOW[args.model][0],
               'stride_qwen_tokens': WINDOW[args.model][1],
               'max_native_tokens': MODELS[args.model]['max_length'],
               'threshold_source': '5% document-any false highlight on shared pure-human calibration',
               'threshold': threshold, 'sets': reports}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
