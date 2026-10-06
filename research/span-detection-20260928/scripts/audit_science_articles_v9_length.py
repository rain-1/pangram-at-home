#!/usr/bin/env python3
"""Check whether document length alone separates the paired pilot classes."""
import json
import statistics
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def main():
    result = {}
    splits = ['calibration', 'locked_test']
    if (ROOT/'paired_archived_locked_test_pilot.jsonl').exists():
        splits.append('archived_locked_test')
    for split in splits:
        rows = [json.loads(x) for x in (ROOT/f'paired_{split}_pilot.jsonl').open()]
        humans = [r['words'] for r in rows if r['label'] == 0]
        ai = [r['words'] for r in rows if r['label'] == 1]
        if not humans or not ai:
            continue
        y = [r['label'] for r in rows]
        scores = [-r['words'] for r in rows]
        result[split] = {'rows': len(rows), 'human_median_words': statistics.median(humans),
                         'ai_median_words': statistics.median(ai),
                         'length_only_auroc_ai_shorter': roc_auc_score(y, scores)}
    (ROOT/'length_confound_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
