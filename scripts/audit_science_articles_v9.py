#!/usr/bin/env python3
"""Audit science article pool against prior protected partitions and itself."""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_span_balanced_v6 import phrase_fingerprints

DATA = Path('/mnt/f/pangram-at-home/data')
ROOT = DATA/'science_articles_v9'


def main():
    rows = [json.loads(x) for x in (ROOT/'human_articles.jsonl').open()]
    rows += [json.loads(x) for x in (ROOT/'epa_science_matters.jsonl').open()]
    archive_path = ROOT/'archived_epa_science_matters_human.jsonl'
    archived_rows = [json.loads(x) for x in archive_path.open()] if archive_path.exists() else []
    sources = {}
    for row in rows:
        sources.setdefault(row['source'], []).append(row)
    result = {'rows': len(rows), 'archived_rows': len(archived_rows),
              'protected_overlap': {}, 'archived_protected_overlap': {}, 'cross_source_overlap': {},
              'exact_duplicate_count': len(rows)-len({x['text_sha256'] for x in rows})}
    protected = [
        DATA/'span_balanced_v6/train.jsonl',
        DATA/'span_publication_hardneg_v8/train.jsonl',
        DATA/'span_realistic_eval_v1/eval.jsonl',
        DATA/'span_realistic_eval_v1/test.jsonl',
        DATA/'span_human_eval_v2/calibration.jsonl',
        DATA/'span_human_eval_v2/test.jsonl',
        DATA/'span_training_v4/val.jsonl',
        DATA/'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
        DATA/'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
        DATA/'span_ai_eval_candidate_v1/test.jsonl',
        DATA/'diverse_pyramid_v1/val_full.parquet',
        DATA/'diverse_pyramid_v1/test_full.parquet',
    ]
    fingerprints = {r['id']: phrase_fingerprints(r['text']) for r in rows}
    archived_fingerprints = {r['id']: phrase_fingerprints(r['text']) for r in archived_rows}
    for path in protected:
        if not path.exists():
            continue
        if path.suffix == '.parquet':
            iterator = (r['text'] for r in pq.read_table(path, columns=['text']).to_pylist())
        else:
            iterator = (json.loads(line)['text'] for line in path.open())
        phrases = set()
        for text in iterator:
            phrases.update(phrase_fingerprints(text))
        hits = [r['id'] for r in rows if fingerprints[r['id']] & phrases]
        result['protected_overlap'][str(path.relative_to(DATA))] = hits
        archived_hits = [r['id'] for r in archived_rows if archived_fingerprints[r['id']] & phrases]
        result['archived_protected_overlap'][str(path.relative_to(DATA))] = archived_hits
    for left in sorted(sources):
        for right in sorted(sources):
            if left >= right:
                continue
            other = set().union(*(fingerprints[r['id']] for r in sources[right]))
            hits = [r['id'] for r in sources[left] if fingerprints[r['id']] & other]
            result['cross_source_overlap'][left+'__'+right] = hits
    (ROOT/'overlap_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if (result['exact_duplicate_count'] or any(result['protected_overlap'].values())
            or any(result['archived_protected_overlap'].values())
            or any(result['cross_source_overlap'].values())):
        raise SystemExit('Overlap detected; remove affected rows before use')


if __name__ == '__main__':
    main()
