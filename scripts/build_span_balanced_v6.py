"""Build a source-capped span training set without touching frozen evaluations.

The 3% LLMTrace cap applies to documents and 512/256 training windows. DAMASHA
is treated as one aggregate source because its upstream IDs are unavailable.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from span_data import window_starts

ROOT = Path('/mnt/f/pangram-at-home')
DATA = ROOT / 'data'
OUT = DATA / 'span_balanced_v6'
SEED = 20260926


def read_jsonl(path):
    with path.open() as file:
        return [json.loads(line) for line in file]


def norm_words(text):
    return re.findall(r'\w+', text.casefold())


def phrase_fingerprints(text):
    words = norm_words(text)
    result = set()
    for i in range(max(0, len(words)-23)):
        h = hashlib.blake2b(' '.join(words[i:i+24]).encode(), digest_size=8).digest()
        if h[0] < 16:
            result.add(h)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUT}')
    rng = random.Random(SEED)
    inputs = {
        'diverse': DATA/'diverse_pyramid_v1/train_full.parquet',
        'v4': DATA/'span_training_v4/train.jsonl',
        'damasha': DATA/'span_sources_v5/normalized_damasha_clean/candidate_only.jsonl',
        'llmtrace': DATA/'span_sources_v5/normalized_llmtrace_en/train.jsonl',
        'audit': DATA/'span_sources_v5/overlap_audit.json',
    }
    protected_files = [
        DATA/'span_training_v4/val.jsonl',
        DATA/'span_human_eval_v2/calibration.jsonl',
        DATA/'span_human_eval_v2/test.jsonl',
        DATA/'span_realistic_eval_v1/test.jsonl',
        DATA/'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
        DATA/'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
        DATA/'span_ai_eval_candidate_v1/test.jsonl',
    ]
    protected_hashes = set()
    protected_phrases = set()
    for path in protected_files:
        for row in read_jsonl(path):
            protected_hashes.add(hashlib.sha256(' '.join(norm_words(row['text'])).encode()).digest())
            protected_phrases.update(phrase_fingerprints(row['text']))
    for row in pq.read_table(DATA/'diverse_pyramid_v1/test_full.parquet', columns=['text']).to_pylist():
        protected_hashes.add(hashlib.sha256(' '.join(norm_words(row['text'])).encode()).digest())
        protected_phrases.update(phrase_fingerprints(row['text']))
    print('Protected sampled phrases:', len(protected_phrases), flush=True)

    selected = []
    seen_text = set()
    def add(row):
        key = hashlib.sha256(row['text'].encode()).hexdigest()
        if key in seen_text:
            return False
        seen_text.add(key)
        selected.append(row)
        return True

    pure = pq.read_table(inputs['diverse']).to_pylist()
    assert len(pure) == 10000
    for row in pure:
        label = int(row['label'])
        add({'id': 'diverse:'+row['text_id'], 'text': row['text'],
             'spans': [{'start': 0, 'end': len(row['text']), 'label': label}],
             'kind': 'ai' if label else 'human', 'construction': 'unaltered_source',
             'source': row['source'], 'domain': row['domain'],
             'source_ids': [row['text_id']], 'source_groups': [row['group_id']],
             'text_sha256': row['text_sha256']})
    assert len(selected) == 10000
    v4 = read_jsonl(inputs['v4'])
    rng.shuffle(v4)
    n_v4 = 0
    for row in v4:
        if add(row):
            n_v4 += 1
        if n_v4 == 4000:
            break
    assert n_v4 == 4000

    # DAMASHA has no parent document/prompt IDs. Filter selected rows against
    # frozen evaluations using exact normalized text and sampled 24-word runs.
    damasha = read_jsonl(inputs['damasha'])
    rng.shuffle(damasha)
    rejected = Counter()
    n_damasha = 0
    for row in damasha:
        if len(row['spans']) < 2:
            continue
        words = norm_words(row['text'])
        if len(words) < 80:
            continue
        if hashlib.sha256(' '.join(words).encode()).digest() in protected_hashes:
            rejected['exact'] += 1
            continue
        if phrase_fingerprints(row['text']) & protected_phrases:
            rejected['phrase'] += 1
            continue
        row['kind'] = 'mixed'
        row['domain'] = 'unknown_aggregate'
        row['construction'] = 'published_mixed'
        if add(row):
            n_damasha += 1
        if n_damasha == 5400:
            break
    assert n_damasha == 5400

    audit = json.loads(inputs['audit'].read_text())
    assert audit['sources']['llmtrace_train']['file_sha256'] == sha(inputs['llmtrace'])
    blocked = set()
    hits = audit['sources']['llmtrace_train']['first_match_ids']
    llm = read_jsonl(inputs['llmtrace'])
    by_id = {row['id']: row for row in llm}
    for ref in ('old_span_val', 'diverse_frozen_test'):
        found = audit['sources']['llmtrace_train']['matches'].get(ref, {}).get('rows_with_any_match', 0)
        assert found <= len(hits.get(ref, [])), f'Truncated overlap list for {ref}'
        blocked.update(by_id[item['id']]['group_id'] for item in hits.get(ref, []))
    llm = [row for row in llm if row['group_id'] not in blocked and len(row['text'].split()) >= 80]
    rng.shuffle(llm)
    llm = sorted(llm[:6000], key=lambda row: len(row['text']))
    # Prefer shorter LLMTrace records so its training-window share stays <=3%.
    n_llm = 0
    for row in llm:
        if add(row):
            n_llm += 1
        if n_llm == 600:
            break
    assert n_llm == 600 and len(selected) == 20000

    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    docs, windows, chars, kinds = Counter(), Counter(), Counter(), Counter()
    for row in selected:
        family = ('DAMASHA' if row['source'] == 'DAMASHA clean published aggregate' else
                  'LLMTrace' if row['source'] == 'LLMTrace_detection' else row['source'])
        docs[family] += 1
        token_count = len(tokenizer(row['text'], add_special_tokens=False)['input_ids'])
        windows[family] += len(window_starts(token_count))
        kinds[row['kind']] += 1
        for span in row['spans']:
            chars[str(span['label'])] += span['end']-span['start']
    assert docs['LLMTrace'] == 600 and docs['LLMTrace']/len(selected) == .03
    assert windows['LLMTrace']/sum(windows.values()) <= .03, windows['LLMTrace']/sum(windows.values())
    assert max(docs.values())/len(selected) <= .30
    rng.shuffle(selected)
    OUT.mkdir(parents=True)
    train = OUT/'train.jsonl'
    with train.open('w') as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False)+'\n')
    # Keep the development validation split source-disjoint from DAMASHA and
    # document/group-disjoint from the paired parent source pool.
    val = OUT/'val.jsonl'
    val.write_bytes((DATA/'span_training_v4/val.jsonl').read_bytes())
    manifest = {'role': '20k source-capped experimental training mixture',
                'seed': SEED, 'documents': len(selected), 'kinds': dict(kinds),
                'source_documents': dict(docs), 'source_windows_512_256': dict(windows),
                'llmtrace_document_fraction': docs['LLMTrace']/len(selected),
                'llmtrace_window_fraction': windows['LLMTrace']/sum(windows.values()),
                'labeled_character_fraction_ai': chars['1']/(chars['0']+chars['1']),
                'damasha_rejected_overlap': dict(rejected),
                'damasha_caveat': 'Published aggregate lacks parent IDs; phrase screening cannot prove no latent overlap.',
                'v4_caveat': 'Synthetic composites reuse underlying diverse training documents.',
                'inputs_sha256': {str(k): sha(v) for k,v in inputs.items()},
                'protected_sha256': {str(p): sha(p) for p in protected_files},
                'train_sha256': sha(train), 'val_sha256': sha(val)}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('documents','kinds','source_documents',
          'llmtrace_window_fraction','labeled_character_fraction_ai','damasha_rejected_overlap')}, indent=2))


if __name__ == '__main__':
    main()
