"""Add source-matched student essay pairs to the diverse v9 span corpus."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home')
ESSAYS = ROOT/'data/asap2_student_essays_v10'
PARENT = ROOT/'data/span_science_paired_v9'
OUTPUT = ROOT/'data/span_essay_paired_v10'
SEED = 20260927
TOKENIZER = ROOT/'models/Qwen3-1.7B'
DIAGNOSTIC = ROOT/'runs/qwen3_token_repeat2_science_paired_v9_20k/v10_asap2_train_diagnostic_v9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def highest_risk_start(scores, width, threshold):
    if len(scores) <= width:
        return 0
    marked = (scores >= threshold).astype(np.int32)
    if marked.any():
        cumulative = np.pad(marked.cumsum(), (1, 0))
    else:
        cumulative = np.pad(scores.cumsum(dtype=np.float64), (1, 0))
    window_values = cumulative[width:]-cumulative[:-width]
    return int(np.argmax(window_values))


def excerpt(row, tokenizer, start, width, label, parent_id, human_id):
    ids = tokenizer.encode(row['text'], add_special_tokens=False)
    text = tokenizer.decode(ids[start:start+width], skip_special_tokens=True)
    if len(tokenizer.encode(text, add_special_tokens=False)) != width:
        raise ValueError(f'Token count changed on decode: {parent_id}')
    return {'id': f'asap2_v10:{"ai" if label else "human"}:{parent_id}',
            'text': text, 'spans': [{'start': 0, 'end': len(text), 'label': label}],
            'kind': 'ai' if label else 'human', 'construction': 'source_matched_student_essay_excerpt',
            'source': 'asap2_student_essay', 'domain': 'student_argumentative_essay',
            'parent_id': parent_id, 'human_id': human_id,
            'generator': row.get('model'), 'source_window_tokens': width,
            'source_window_start': start,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}


def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    humans = {row['id']: row for row in (json.loads(line) for line in
                                          (ESSAYS/'selected_train_human.jsonl').open())}
    ai_files = [ESSAYS/f'accepted_{model}_train.jsonl'
                for model in ('qwen2_5_3b', 'smollm2_1_7b')]
    ai = [json.loads(line) for path in ai_files for line in path.open()]
    assert 150 <= len(ai) <= len(humans)
    assert len({row['human_id'] for row in ai}) == len(ai)
    assert all(row['human_id'] in humans and
               row['human_text_sha256'] == humans[row['human_id']]['text_sha256']
               for row in ai)
    diagnostic_report = json.loads(DIAGNOSTIC.with_suffix('.json').read_text())
    assert diagnostic_report['validation_sha256'] == sha(ESSAYS/'train_candidates.jsonl')
    with np.load(str(DIAGNOSTIC)+'_scores.npz') as data:
        offsets, ids, scores = (data[key] for key in ('document_offsets', 'document_ids', 'score'))
        risk_scores = {str(row_id): scores[offsets[i]:offsets[i+1]]
                       for i, row_id in enumerate(ids)}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    pairs = []
    hard_false_tokens = 0
    for generated in ai:
        human = humans[generated['human_id']]
        h_ids = tokenizer.encode(human['text'], add_special_tokens=False)
        a_ids = tokenizer.encode(generated['text'], add_special_tokens=False)
        width = min(512, len(h_ids), len(a_ids))
        assert width >= 300
        risk = risk_scores[human['id']]
        assert len(risk) == len(h_ids)
        h_start = highest_risk_start(risk, width, diagnostic_report['threshold'])
        hard_false_tokens += int((risk[h_start:h_start+width] >= diagnostic_report['threshold']).sum())
        h_max, a_max = len(h_ids)-width, len(a_ids)-width
        a_start = round(h_start/h_max*a_max) if h_max else 0
        h = excerpt(human, tokenizer, h_start, width, 0, human['id'], human['id'])
        a = excerpt(generated, tokenizer, a_start, width, 1, generated['id'], human['id'])
        assert h['kind'] == 'human' and a['kind'] == 'ai'
        pairs.extend((h, a))
    rng = random.Random(SEED)
    original = [json.loads(line) for line in (PARENT/'train.jsonl').open()]
    # Replace abstract-heavy pure rows while preserving v9 science pairs and all
    # mixed-document supervision.
    old_human = [r for r in original if r['source'] == 'mage:sci' and r['kind'] == 'human']
    old_ai = [r for r in original if r['source'] == 'mage:sci' and r['kind'] == 'ai']
    rng.shuffle(old_human)
    rng.shuffle(old_ai)
    assert len(old_human) >= len(ai) and len(old_ai) >= len(ai)
    removed = {r['id'] for r in old_human[:len(ai)]+old_ai[:len(ai)]}
    rows = [r for r in original if r['id'] not in removed]+pairs
    rng.shuffle(rows)
    assert len(rows) == 20000
    assert len({row['id'] for row in rows}) == len(rows)
    OUTPUT.mkdir(parents=True)
    with (OUTPUT/'train.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    (OUTPUT/'val.jsonl').write_bytes((PARENT/'val.jsonl').read_bytes())
    manifest = {'role': 'paired student essay pilot with source-exclusive prompt evaluation',
                'seed': SEED, 'documents': len(rows), 'paired_essay_topics': len(ai),
                'added_documents': len(pairs), 'replaced_mage_sci_pure_documents': len(removed),
                'matched_source_token_width_per_pair': True,
                'captured_v9_false_positive_tokens': hard_false_tokens,
                'v9_diagnostic_sha256': sha(DIAGNOSTIC.with_suffix('.json')),
                'added_labels': dict(Counter(row['kind'] for row in pairs)),
                'added_generators': dict(Counter(row['model'] for row in ai)),
                'sources': dict(Counter(row['source'] for row in rows)),
                'kinds': dict(Counter(row['kind'] for row in rows)),
                'parent_train_sha256': sha(PARENT/'train.jsonl'),
                'essay_human_sha256': sha(ESSAYS/'selected_train_human.jsonl'),
                'essay_ai_sha256': {path.name: sha(path) for path in ai_files},
                'train_sha256': sha(OUTPUT/'train.jsonl'),
                'val_sha256': sha(OUTPUT/'val.jsonl')}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in
                      ('documents','paired_essay_topics','added_documents','added_labels')}, indent=2))


if __name__ == '__main__':
    main()
