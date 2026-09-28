"""Add source-disjoint essay/forum humans and open-ended MAGE AI to v13.

This is a small balanced addition, not a hard-negative mining pass. The
existing v13 training/validation/evaluation files stay frozen. Candidate text
is rejected on exact normalized text or sampled 24-word phrase overlap with
any protected file, and Writers components are split by thread and owner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from build_span_balanced_v6 import norm_words, phrase_fingerprints
from build_span_human_eval_v2 import components, from_row


SEED = 20260928
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/mnt/f/pangram-at-home')
PROTECTED = (
    'span_new_sources_v13/train.jsonl',
    'span_new_sources_v13/val.jsonl',
    'span_new_sources_v13/new_source_holdout.jsonl',
    'span_human_eval_v2/calibration.jsonl',
    'span_human_eval_v2/test.jsonl',
    'persuade_essay_calibration_v11/calibration.jsonl',
    'span_ai_eval_candidate_v1/test.jsonl',
    'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
    'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
    'pmc_publication_v6/test.jsonl',
    'cnn_dailymail_v1/locked_test.jsonl',
    'asap2_student_essays_v10/locked_test_human.jsonl',
)


def digest(path: Path) -> str:
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def text_key(text: str) -> bytes:
    return sha256(' '.join(norm_words(text)).encode()).digest()


def ranked(values: list[dict], salt: str) -> list[dict]:
    return sorted(values, key=lambda row: sha256(
        f'{SEED}:{salt}:{row["text_id"]}'.encode()).digest())


def bounded(row: dict, tokenizer, salt: str) -> dict:
    result = dict(row)
    tokens = tokenizer.encode(row['text'], add_special_tokens=False)
    if len(tokens) <= 512:
        return result
    start = random.Random(f'{SEED}:{salt}:{row["id"]}').randrange(len(tokens)-511)
    result['text'] = tokenizer.decode(tokens[start:start+512], skip_special_tokens=True)
    result['spans'] = [{'start': 0, 'end': len(result['text']), 'label': row['spans'][0]['label']}]
    result['excerpt_source_token_start'] = start
    result['original_text_sha256'] = sha256(row['text'].encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=REPO/'data/span_hardneg_v14')
    args = parser.parse_args()
    root, data, output = args.root, args.root/'data', args.output
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}')
    parent = data/'span_new_sources_v13'
    old = rows(parent/'train.jsonl')
    assert len(old) == 20000
    assert digest(parent/'train.jsonl') == 'dab782f2fb14569fac9acedc96d1b8416f265a297f933fd0d7937cc256876915'

    protected_hashes: set[bytes] = set()
    protected_phrases: set[bytes] = set()
    protected_ids: set[str] = set()
    protected_groups: set[str] = set()
    for relative in PROTECTED:
        for row in rows(data/relative):
            protected_hashes.add(text_key(row['text']))
            protected_phrases.update(phrase_fingerprints(row['text']))
            protected_ids.update(str(item.get('text_id', '')) for item in row.get('source_records', []))
            protected_groups.update(str(group) for group in row.get('source_groups', []))

    chosen_hashes: set[bytes] = set()
    chosen_phrases: set[bytes] = set()
    rejected = Counter()
    def eligible(text: str, source: str) -> bool:
        if len(text.split()) < 80:
            rejected[f'{source}:short'] += 1
            return False
        key = text_key(text)
        if key in protected_hashes or key in chosen_hashes:
            rejected[f'{source}:exact'] += 1
            return False
        fingerprints = phrase_fingerprints(text)
        if fingerprints & (protected_phrases | chosen_phrases):
            rejected[f'{source}:24word'] += 1
            return False
        chosen_hashes.add(key)
        chosen_phrases.update(fingerprints)
        return True

    tokenizer = AutoTokenizer.from_pretrained(root/'models/Qwen3-1.7B')
    essay_path = data/'persuade_essays_v1/human_eval.parquet'
    essay_pool = pq.read_table(essay_path).to_pylist()
    essay_pool = [row for row in essay_pool if row['text_id'] not in protected_ids
                  and str(row['group_id']) not in protected_groups]
    human = []
    for row in ranked(essay_pool, 'persuade'):
        if len(human) >= 400:
            break
        if eligible(row['text'], 'persuade'):
            human.append(bounded(from_row(row, 'train'), tokenizer, 'persuade'))
    assert len(human) == 400, len(human)

    writer_path = data/'stackexchange_writers_v1/human_eval.parquet'
    writer_pool = pq.read_table(writer_path).to_pylist()
    writer_components = components(writer_pool)
    unseen = [comp for comp in writer_components
              if not any(row['text_id'] in protected_ids or
                         str(row['group_id']) in protected_groups for row in comp)]
    writer_order = sorted(unseen, key=lambda comp: sha256(
        f'{SEED}:writers:{min(row["text_id"] for row in comp)}'.encode()).digest())
    writer_count = 0
    for comp in writer_order:
        for row in ranked(comp, 'writer-component'):
            if writer_count >= 200:
                break
            last_edit = row['last_edit_at'] or row['created_at']
            if row['created_at'][:10] > '2022-12-31' or last_edit[:10] > '2022-12-31':
                rejected['writers:date'] += 1
                continue
            if eligible(row['text'], 'writers'):
                human.append(bounded(from_row(row, 'train'), tokenizer, 'writers'))
                writer_count += 1
        if writer_count >= 200:
            break
    assert writer_count == 200, writer_count

    mage_path = data/'mage_external_v1/raw/train.csv'
    mage = pd.read_csv(mage_path, usecols=['text', 'label', 'src'])
    # These are open-ended ChangeMyView and ELI5 full responses, not the
    # continuation variants. MAGE uses label 0 for AI.
    ai = []
    targets = {'cmv': 400, 'eli5': 200}
    models = ('gpt-3.5-trubo', 'text-davinci-003', 'text-davinci-002')
    for domain, target in targets.items():
        shares = [target // 3] * 3
        for i in range(target % 3):
            shares[i] += 1
        for model, quota in zip(models, shares):
            pool = []
            for index, row in mage.iterrows():
                source = str(row['src'])
                if row['label'] != 0 or not source.startswith(domain+'_machine_'):
                    continue
                if not any(f'_{mode}_{model}' in source for mode in ('topical', 'specified')):
                    continue
                pool.append({'text_id': f'mage-train:{index}', 'text': row['text'],
                             'src': source, 'row_index': int(index)})
            selected = 0
            for row in ranked(pool, domain+model):
                if selected >= quota:
                    break
                text = row['text']
                if not isinstance(text, str) or not eligible(text, f'mage:{domain}'):
                    continue
                item = {'id': f'v14:{row["text_id"]}', 'text': text,
                        'spans': [{'start': 0, 'end': len(text), 'label': 1}],
                        'kind': 'ai', 'construction': 'unaltered_open_ended_response',
                        'source': f'mage:{domain}', 'domain': 'social_qa',
                        'generator': model, 'source_variant': row['src'],
                        'source_ids': [row['text_id']],
                        'source_groups': [row['text_id']]}
                ai.append(bounded(item, tokenizer, 'mage'))
                selected += 1
            assert selected == quota, (domain, model, selected, quota)
    assert len(ai) == 600

    output.mkdir(parents=True)
    train = old + human + ai
    random.Random(SEED).shuffle(train)
    assert len(train) == 21200 and len({row['id'] for row in train}) == len(train)
    with (output/'train.jsonl').open('w') as stream:
        for row in train:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    for name in ('val.jsonl', 'new_source_holdout.jsonl'):
        (output/name).write_bytes((parent/name).read_bytes())
    manifest = {'role': 'v14 balanced essay/forum human and open-ended AI addition',
                'seed': SEED, 'documents': len(train), 'new_human': len(human),
                'new_ai': len(ai), 'sources': dict(Counter(row['source'] for row in train)),
                'kinds': dict(Counter(row['kind'] for row in train)),
                'new_generators': dict(Counter(row['generator'] for row in ai)),
                'rejected_candidates': dict(rejected),
                'overlap_rule': 'exact normalized text and sampled 24-word phrase fingerprints',
                'protected_files': {name: digest(data/name) for name in PROTECTED},
                'parent_train_sha256': digest(parent/'train.jsonl'),
                'candidate_files_sha256': {str(path.relative_to(data)): digest(path)
                                          for path in (essay_path, writer_path, mage_path)},
                'train_sha256': digest(output/'train.jsonl'),
                'val_sha256': digest(output/'val.jsonl'),
                'holdout_sha256': digest(output/'new_source_holdout.jsonl')}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in
                      ('documents', 'new_human', 'new_ai', 'new_generators',
                       'rejected_candidates', 'kinds')}, indent=2))


if __name__ == '__main__':
    main()
