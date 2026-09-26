"""Score the independent Human Detectors article set with frozen EditLens thresholds.

This research-only test is never used for calibration. Human article authorship
is credited by the source, but AI-free writing workflows are not verified.
"""
from __future__ import annotations

import argparse
import json

from evaluate_open_pangram_pure_v5 import MODELS, ROOT, load_model, metrics, read_rows, score_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    spec = MODELS[args.model]
    source = ROOT / 'data/span_ai_eval_candidate_v1/test.jsonl'
    rows = read_rows(source, {'human', 'ai'}, None)
    output = ROOT / 'runs' / f'open_pangram_editlens_{args.model}_v5'
    threshold = json.loads((output / 'summary.json').read_text())['threshold']
    tokenizer, model = load_model(args.model)
    scores = score_dataset('external_human_detectors', rows, tokenizer, model,
                           spec['max_length'], spec['batch_size'],
                           output / 'external_human_detectors.jsonl')
    # This source reuses numeric article IDs across generator variants, so
    # align scores by the preserved input row order rather than by ID.
    assert len(scores) == len(rows)
    assert all(score['id'] == row['id'] for score, row in zip(scores, rows))
    report = {
        'model': spec['source'],
        'source': 'Russell et al., Human Detectors (2025); 150 human and 150 AI articles',
        'threshold': threshold,
        'threshold_source': 'frozen separate pure-human calibration; no tuning on this set',
        'overall': metrics(scores, threshold),
        'by_generator': {
            generator: metrics([score for score, row in zip(scores, rows)
                                if row['kind'] == 'ai' and row.get('generator') == generator], threshold)
            for generator in sorted({row['generator'] for row in rows if row['kind'] == 'ai'})
        },
        'human_provenance_caveat': 'Named published authors; AI-free writing workflow not independently verified',
    }
    (output / 'external_human_detectors_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
