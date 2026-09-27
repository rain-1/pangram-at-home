#!/usr/bin/env python3
"""Quality gate for locally generated topic-matched science articles."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_span_balanced_v6 import phrase_fingerprints

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def main():
    generated = ROOT/'generated_qwen2_5_3b_pilot.jsonl'
    rows = [json.loads(x) for x in generated.open()]
    humans = {r['id']: r for name in ('calibration_human', 'locked_test_human')
              for r in (json.loads(x) for x in (ROOT/f'{name}.jsonl').open())}
    reasons = Counter()
    accepted = []
    seen = set()
    for row in rows:
        h = humans[row['human_id']]
        text = row['text'].strip()
        if row['text_sha256'] in seen:
            reasons['duplicate_ai_text'] += 1
        elif not (250 <= len(text.split()) <= 1000):
            reasons['bad_word_count'] += 1
        elif not text.endswith(('.', '!', '?', '”', '"')):
            reasons['incomplete_ending'] += 1
        elif phrase_fingerprints(text) & phrase_fingerprints(h['text']):
            reasons['verbatim_24_word_overlap_with_human'] += 1
        elif text.count('\n\n') < 2:
            reasons['too_few_paragraphs'] += 1
        else:
            seen.add(row['text_sha256'])
            accepted.append(row)
    out = ROOT/'matched_ai_pilot_accepted.jsonl'
    with out.open('w') as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'generated': len(rows), 'accepted': len(accepted), 'rejected': dict(reasons),
                'accepted_by_split': dict(Counter(r['split'] for r in accepted)),
                'accepted_by_model': dict(Counter(r['model'] for r in accepted)),
                'accepted_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                'note': 'This is a topic-matched pilot from one local open-weight model, not a broad multi-model AI evaluation.'}
    (ROOT/'matched_ai_pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
