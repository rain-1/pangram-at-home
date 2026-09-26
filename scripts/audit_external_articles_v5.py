"""Audit train overlap and row identity for the external article stress set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/mnt/f/pangram-at-home/data')
EXTERNAL = ROOT / 'span_ai_eval_candidate_v1/test.jsonl'
TRAIN = ROOT / 'span_size_curve_v5/size_20000/train.jsonl'


def words(text: str):
    return re.findall(r'\b\w+\b', text.casefold())


def read_rows(path: Path):
    return [json.loads(line) for line in path.open()]


def main():
    external = read_rows(EXTERNAL)
    hashes = {hashlib.sha256(row['text'].encode()).digest() for row in external}
    shingles = set()
    for row in external:
        tokens = words(row['text'])
        shingles.update(hash(tuple(tokens[i:i + 24])) for i in range(len(tokens) - 23))
    exact = shared = train_count = 0
    for line in TRAIN.open():
        row = json.loads(line)
        train_count += 1
        exact += hashlib.sha256(row['text'].encode()).digest() in hashes
        tokens = words(row['text'])
        shared += any(hash(tuple(tokens[i:i + 24])) in shingles
                      for i in range(len(tokens) - 23))
    result = {
        'external_rows': len(external),
        'external_unique_texts': len(hashes),
        'external_unique_ids': len({row['id'] for row in external}),
        'external_unique_source_groups': len({tuple(row['source_groups']) for row in external}),
        'train_rows': train_count,
        'train_exact_text_overlap_rows': exact,
        'train_rows_with_normalized_24_word_overlap': shared,
        'note': 'External numeric IDs repeat across variants; align scores by row order.',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
