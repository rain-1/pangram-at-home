"""Audit paired science training generations before building the span corpus."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from build_span_balanced_v6 import phrase_fingerprints

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
TOKENIZER = Path('/mnt/f/pangram-at-home/models/Qwen3-1.7B')
MODELS = ('qwen2_5_3b', 'smollm2_1_7b')


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    humans = {r['id']: r for r in (json.loads(line) for line in (ROOT/'train_candidates.jsonl').open())}
    protected_phrases = set()
    for name in ('train_candidates', 'calibration_human', 'locked_test_human',
                 'archived_epa_science_matters_human'):
        for row in (json.loads(line) for line in (ROOT/f'{name}.jsonl').open()):
            protected_phrases.update(phrase_fingerprints(row['text']))
    reasons = Counter()
    salvaged = Counter()
    all_accepted = []
    for model in MODELS:
        source = ROOT/f'generated_{model}_train.jsonl'
        rows = [json.loads(line) for line in source.open()]
        accepted = []
        seen = set()
        for row in rows:
            human = humans.get(row['human_id'])
            text = row['text'].strip()
            if not text.endswith(('.', '!', '?', '”', '"')):
                last = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last > 0 and len(text)-last <= 350:
                    text = text[:last+1].strip()
                    salvaged['trimmed_incomplete_last_sentence'] += 1
            if human is None or row['human_text_sha256'] != human['text_sha256']:
                reasons['bad_pair'] += 1
            elif row['human_id'] in seen:
                reasons['duplicate_human_id'] += 1
            elif len(text.split()) < 350 or len(tokenizer.encode(text, add_special_tokens=False)) < 512:
                reasons['short'] += 1
            elif re.match(r'(?i)^(?:title:|here(?:\s+is|\u2019s)|certainly|as an ai|i cannot)\b', text):
                reasons['meta_response_or_title'] += 1
            elif text.count('\n\n') < 2:
                reasons['few_paragraphs'] += 1
            elif not text.endswith(('.', '!', '?', '”', '"')):
                reasons['incomplete_ending'] += 1
            elif phrase_fingerprints(text) & protected_phrases:
                reasons['24_word_overlap_with_human'] += 1
            else:
                seen.add(row['human_id'])
                row.update(text=text, words=len(text.split()),
                           text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           spans=[{'start': 0, 'end': len(text), 'label': 1}])
                accepted.append(row)
                all_accepted.append(row)
        output = ROOT/f'accepted_{model}_train.jsonl'
        output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in accepted))
    manifest = {'generated': sum(1 for model in MODELS for _ in (ROOT/f'generated_{model}_train.jsonl').open()),
                'accepted': len(all_accepted), 'rejected': dict(reasons), 'salvaged': dict(salvaged),
                'accepted_by_model': dict(Counter(r['model'] for r in all_accepted)),
                'accepted_human_ids': len({r['human_id'] for r in all_accepted}),
                'accepted_sha256': {model: hashlib.sha256((ROOT/f'accepted_{model}_train.jsonl').read_bytes()).hexdigest()
                                    for model in MODELS}}
    (ROOT/'train_ai_audit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
