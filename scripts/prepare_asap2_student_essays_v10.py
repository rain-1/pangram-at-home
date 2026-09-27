"""Build a source-protected human-essay pool from the original ASAP 2.0 release.

The local outputs intentionally omit student demographic fields. Raw essay text
stays on the external disk and is never checked into Git.
"""
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path('/mnt/f/pangram-at-home')
OUTPUT = ROOT/'data/asap2_student_essays_v10'
PERSUADE = ROOT/'data/persuade_essays_v1/human_eval.parquet'
PARENT = ROOT/'data/span_science_paired_v9'
TRAIN_PROMPTS = {'Car-free cities', 'Does the electoral college work?'}
TEST_PROMPTS = {'Driverless cars', 'Exploring Venus'}
LICENSE = 'https://github.com/scrosseye/ASAP_2.0#readme'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shingles(text):
    words = re.findall(r'\b\w+\b', text.casefold())
    return {hashlib.blake2b(' '.join(words[i:i+24]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words)-23))}


def rank(row, purpose):
    return sha(f'asap2-v10:{purpose}:{row["id"]}'.encode())


def main():
    zip_path = OUTPUT/'ASAP_2_Final_github_train.zip'
    with zipfile.ZipFile(zip_path) as archive:
        frame = pd.read_csv(archive.open('ASAP_2_Final_github_train.csv'),
                            usecols=['essay_id', 'full_text', 'prompt_name', 'essay_word_count'])
    assert frame.essay_id.is_unique
    protected = set()
    persuade = pq.read_table(PERSUADE, columns=['text']).column('text').to_pylist()
    for text in persuade:
        protected.update(shingles(text))
    persuade_shingles = len(protected)
    for name in ('train.jsonl', 'val.jsonl'):
        for line in (PARENT/name).open():
            protected.update(shingles(json.loads(line)['text']))
    parent_shingles = len(protected)-persuade_shingles
    candidates = []
    reasons = Counter()
    for row in frame.itertuples(index=False):
        if not isinstance(row.full_text, str) or len(row.full_text.split()) < 300:
            reasons['short'] += 1
            continue
        if row.prompt_name not in TRAIN_PROMPTS | TEST_PROMPTS:
            reasons['other_prompt'] += 1
            continue
        if shingles(row.full_text) & protected:
            reasons['protected_24_word_overlap'] += 1
            continue
        text = row.full_text.strip()
        record = {'id': 'asap2:'+str(row.essay_id), 'text': text,
                  'spans': [{'start': 0, 'end': len(text), 'label': 0}],
                  'kind': 'human', 'label': 0,
                  'construction': 'source_based_standardized_student_essay',
                  'source': 'asap2_student_essay', 'domain': 'student_argumentative_essay',
                  'prompt_name': row.prompt_name, 'words': len(text.split()),
                  'text_sha256': sha(text.encode()), 'license_url': LICENSE}
        candidates.append(record)
    training = sorted([row for row in candidates if row['prompt_name'] in TRAIN_PROMPTS],
                      key=lambda row: rank(row, 'train'))
    heldout = sorted([row for row in candidates if row['prompt_name'] in TEST_PROMPTS],
                     key=lambda row: rank(row, 'test'))
    # Source-exclusive prompts make this an independent transfer check.
    test = []
    for prompt in sorted(TEST_PROMPTS):
        test.extend([row for row in heldout if row['prompt_name'] == prompt][:100])
    assert len(training) >= 500 and len(test) == 200
    assert not ({row['id'] for row in training} & {row['id'] for row in test})
    assert not ({row['prompt_name'] for row in training} & {row['prompt_name'] for row in test})
    for name, rows in [('train_candidates.jsonl', training), ('locked_test_human.jsonl', test)]:
        path = OUTPUT/name
        if path.exists():
            raise SystemExit(f'Refusing to overwrite {path}')
        with path.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'source': 'original ASAP 2.0 GitHub release',
                'source_url': 'https://github.com/scrosseye/ASAP_2.0',
                'source_zip_sha256': sha(zip_path.read_bytes()),
                'license_url': LICENSE,
                'source_provenance': 'state standardized writing tests in grades 6, 8, 9, and 10',
                'source_rows': len(frame), 'protected_persuade_rows': len(persuade),
                'protected_persuade_shingles': persuade_shingles,
                'additional_parent_shingles': parent_shingles,
                'rejections': dict(reasons),
                'training_candidates': len(training), 'locked_test': len(test),
                'train_prompts': sorted(TRAIN_PROMPTS), 'test_prompts': sorted(TEST_PROMPTS),
                'train_sha256': sha((OUTPUT/'train_candidates.jsonl').read_bytes()),
                'test_sha256': sha((OUTPUT/'locked_test_human.jsonl').read_bytes()),
                'privacy': 'only essay text, prompt name, and opaque essay ID retained; demographics discarded; raw text local only'}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('training_candidates','locked_test','rejections')}, indent=2))


if __name__ == '__main__':
    main()
