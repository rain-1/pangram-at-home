#!/usr/bin/env python3
"""Create equal-token opening/ending windows from the balanced article pilot."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
TOKENIZER = Path('/mnt/f/pangram-at-home/models/Qwen3-1.7B')
WINDOW = 512


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    manifest = {'tokenizer': str(TOKENIZER), 'source_tokens_per_window': WINDOW,
                'positions': ['opening', 'ending'], 'splits': {}}
    splits = ['calibration', 'locked_test']
    if (ROOT/'paired_archived_locked_test_pilot.jsonl').exists():
        splits.append('archived_locked_test')
    for split in splits:
        parents = [json.loads(x) for x in (ROOT/f'paired_{split}_pilot.jsonl').open()]
        encoded = [(row, tok.encode(row['text'], add_special_tokens=False)) for row in parents]
        short_pairs = {row.get('human_id', row['id'].split(':archive:')[0])
                       for row, ids in encoded if len(ids) < WINDOW}
        if short_pairs and split != 'archived_locked_test':
            raise SystemExit(f'{split}: {len(short_pairs)} pairs shorter than {WINDOW} tokens')
        windows = []
        for row, ids in encoded:
            pair_id = row.get('human_id', row['id'].split(':archive:')[0])
            if pair_id in short_pairs:
                continue
            for position, start in (('opening', 0), ('ending', len(ids)-WINDOW)):
                subset = ids[start:start+WINDOW]
                text = tok.decode(subset, skip_special_tokens=True)
                assert len(subset) == WINDOW
                windows.append({
                    'id': row['id']+':'+position,
                    'parent_id': row['id'],
                    'human_id': row.get('human_id', row['id'].split(':archive:')[0]),
                    'source': row['source'], 'label': row['label'],
                    'kind': row['kind'], 'text': text,
                    'spans': [{'start': 0, 'end': len(text), 'label': row['label']}],
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'parent_text_sha256': row['text_sha256'],
                    'window_position': position, 'window_start_token': start,
                    'window_source_tokens': WINDOW,
                })
        path = ROOT/f'paired_{split}_windows_pilot.jsonl'
        with path.open('w') as f:
            for row in windows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        manifest['splits'][split] = {'rows': len(windows),
                                     'labels': dict(Counter(str(r['label']) for r in windows)),
                                     'excluded_short_pairs': len(short_pairs),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT/'window_pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
