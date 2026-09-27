#!/usr/bin/env python3
"""Build paired 1:1 human/AI pilot files without altering full human pools."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def read(path):
    return [json.loads(x) for x in path.open()]


def main():
    ai = read(ROOT/'matched_ai_pilot_accepted.jsonl')
    humans = {r['id']: r for name in ('calibration_human', 'locked_test_human')
              for r in read(ROOT/f'{name}.jsonl')}
    outputs = {'calibration': [], 'locked_test': []}
    for row in ai:
        h = humans[row['human_id']]
        split = 'calibration' if row['split'] == 'calibration_ai_candidate' else 'locked_test'
        assert row['source'] == h['source']
        assert row['human_text_sha256'] == h['text_sha256']
        outputs[split].extend((h, row))
    manifest = {'purpose': 'topic-matched 1:1 pilot evaluation; full human pools remain separate',
                'splits': {}, 'model': 'Qwen/Qwen2.5-3B-Instruct'}
    for split, rows in outputs.items():
        rows.sort(key=lambda r: (r.get('human_id', r['id']), r['label']))
        path = ROOT/f'paired_{split}_pilot.jsonl'
        with path.open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        manifest['splits'][split] = {'rows': len(rows), 'labels': dict(Counter(str(r['label']) for r in rows)),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    archived_path = ROOT/'archived_epa_science_matters_human.jsonl'
    if archived_path.exists():
        archived = {r['id'].split(':archive:')[0]: r for r in read(archived_path)}
        archived_pairs = []
        for row in ai:
            if row['split'] != 'locked_test_ai_candidate' or row['human_id'] not in archived:
                continue
            h = archived[row['human_id']]
            assert h['source'] == row['source']
            archived_pairs.extend((h, row))
        archived_pairs.sort(key=lambda r: (r.get('human_id', r['id'].split(':archive:')[0]), r['label']))
        path = ROOT/'paired_archived_locked_test_pilot.jsonl'
        with path.open('w') as f:
            for row in archived_pairs:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        manifest['splits']['archived_locked_test'] = {
            'rows': len(archived_pairs),
            'labels': dict(Counter(str(r['label']) for r in archived_pairs)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (ROOT/'paired_pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
