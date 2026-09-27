"""Audit source-matched AI student essays before adding them to training."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_asap2_ai_prompts_v10 import source_texts
from prepare_asap2_student_essays_v10 import shingles

ROOT = Path('/mnt/f/pangram-at-home/data/asap2_student_essays_v10')
TOKENIZER = Path('/mnt/f/pangram-at-home/models/Qwen3-1.7B')
MODELS = ('qwen2_5_3b', 'smollm2_1_7b')


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    humans = {row['id']: row for row in (json.loads(line) for line in
                                          (ROOT/'selected_train_human.jsonl').open())}
    protected = set()
    for name in ('train_candidates', 'locked_test_human'):
        for line in (ROOT/f'{name}.jsonl').open():
            protected.update(shingles(json.loads(line)['text']))
    persuade = Path('/mnt/f/pangram-at-home/data/persuade_essays_v1/human_eval.parquet')
    for text in pq.read_table(persuade, columns=['text']).column('text').to_pylist():
        protected.update(shingles(text))
    protected_readings = {name: shingles(text)
                          for name, text in source_texts().items()}
    rejected = Counter()
    cleaned = Counter()
    accepted_ids = set()
    counts = {}
    for model in MODELS:
        rows = [json.loads(line) for line in (ROOT/f'generated_{model}_train.jsonl').open()]
        accepted = []
        for row in rows:
            text = row['text'].strip()
            first, sep, rest = text.partition('\n\n')
            if sep and len(first.split()) <= 14 and len(rest.split()) >= 250 \
                    and not first.rstrip().endswith(('.', '!', '?', '”', '"')):
                text = rest.strip()
                cleaned['removed_heading'] += 1
            if not text.endswith(('.', '!', '?', '”', '"')):
                last = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last > 0 and len(text)-last <= 300:
                    text = text[:last+1].strip()
                    cleaned['trimmed_incomplete_tail'] += 1
            human = humans.get(row['human_id'])
            if human is None or row['human_text_sha256'] != human['text_sha256']:
                rejected['bad_pair'] += 1
            elif row['human_id'] in accepted_ids:
                rejected['duplicate_pair'] += 1
            elif len(text.split()) < 260 or len(tokenizer.encode(text, add_special_tokens=False)) < 300:
                rejected['too_short'] += 1
            elif re.match(r'(?i)^(?:title:|here(?:\s+is|\u2019s)|certainly|as an ai|i cannot)\b', text):
                rejected['meta_response'] += 1
            elif text.count('\n\n') < 1:
                rejected['no_paragraphs'] += 1
            elif not text.endswith(('.', '!', '?', '”', '"')):
                rejected['incomplete_ending'] += 1
            elif shingles(text) & protected:
                rejected['human_24_word_overlap'] += 1
            elif shingles(text) & protected_readings[row['topic_title']]:
                rejected['source_reading_24_word_overlap'] += 1
            else:
                accepted_ids.add(row['human_id'])
                row.update(text=text, words=len(text.split()),
                           text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           spans=[{'start': 0, 'end': len(text), 'label': 1}])
                accepted.append(row)
        output = ROOT/f'accepted_{model}_train.jsonl'
        output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in accepted))
        counts[model] = {'generated': len(rows), 'accepted': len(accepted),
                         'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    manifest = {'models': counts, 'accepted_pairs': len(accepted_ids),
                'rejected': dict(rejected), 'cleaned': dict(cleaned),
                'protected_human_24_word_shingles': len(protected)}
    (ROOT/'ai_audit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
