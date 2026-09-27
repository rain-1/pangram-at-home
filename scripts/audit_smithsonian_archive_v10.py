"""Verify archived magazine provenance, overlap, and author-exclusive split."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_span_balanced_v6 import phrase_fingerprints

DATA = Path('/mnt/f/pangram-at-home/data')
ROOT = DATA/'smithsonian_archive_v10'
TEST_AUTHORS = {'Livia Gershon', 'Rasha Aridi', 'Jane Recker'}
PROTECTED = (
    'span_science_paired_v9/train.jsonl',
    'span_science_paired_v9/val.jsonl',
    'span_human_eval_v2/calibration.jsonl',
    'span_human_eval_v2/test.jsonl',
    'span_ai_eval_candidate_v1/test.jsonl',
    'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
    'span_realistic_eval_v1/test.jsonl',
    'science_articles_v9/archived_epa_science_matters_human.jsonl',
    'science_articles_v9/archived_noaa_fisheries_human.jsonl',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [json.loads(line) for line in (ROOT/'articles.jsonl').open()]
    assert len(rows) == 100
    assert len({r['id'] for r in rows}) == len({r['text_sha256'] for r in rows}) == 100
    assert all(r['published_at'] < '2023-01-01'
               and r['capture_timestamp'] < '20230101'
               and r['capture_timestamp'][:8] >= r['published_at'].replace('-', '') for r in rows)
    fp = {r['id']: phrase_fingerprints(r['text']) for r in rows}
    internal = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if fp[a['id']] & fp[b['id']]:
                internal.append([a['id'], b['id']])
    external = {}
    for name in PROTECTED:
        phrases = set()
        path = DATA/name
        for line in path.open():
            phrases.update(phrase_fingerprints(json.loads(line)['text']))
        external[name] = [r['id'] for r in rows if fp[r['id']] & phrases]
    overlap = {'internal_pairs': internal, 'protected_hits': external,
               'protected_sha256': {name: sha(DATA/name) for name in PROTECTED}}
    (ROOT/'overlap_audit.json').write_text(json.dumps(overlap, indent=2)+'\n')
    if internal or any(external.values()):
        raise ValueError('Magazine pool overlaps protected data; inspect audit before splitting')
    train = [r for r in rows if r['author'] not in TEST_AUTHORS]
    test = [r for r in rows if r['author'] in TEST_AUTHORS]
    assert len(train) == 79 and len(test) == 21
    assert not ({r['author'] for r in train} & {r['author'] for r in test})
    for name, group in (('train_candidates', train), ('locked_test_human', test)):
        (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in group))
    manifest = {'role': 'pre-2023 archive-verified magazine hard-negative candidates',
                'documents': len(rows), 'train_candidates': len(train),
                'locked_test_human': len(test), 'test_authors': sorted(TEST_AUTHORS),
                'train_authors': dict(Counter(r['author'] for r in train)),
                'test_authors_count': dict(Counter(r['author'] for r in test)),
                'train_sha256': sha(ROOT/'train_candidates.jsonl'),
                'test_sha256': sha(ROOT/'locked_test_human.jsonl'),
                'overlap_audit_sha256': sha(ROOT/'overlap_audit.json'),
                'rights_note': 'Copyrighted magazine prose stays in local research storage; code and metadata only in Git.'}
    (ROOT/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
