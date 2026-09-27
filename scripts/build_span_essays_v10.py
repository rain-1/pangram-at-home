"""Add source-matched student essay pairs to the diverse v9 span corpus."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home')
ESSAYS = ROOT/'data/asap2_student_essays_v10'
PARENT = ROOT/'data/span_science_paired_v9'
OUTPUT = ROOT/'data/span_essay_paired_v10'
SEED = 20260927


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    pairs = []
    for generated in ai:
        human = humans[generated['human_id']]
        h = {**human, 'id': 'asap2_v10:human:'+human['id'],
             'construction': 'source_matched_student_essay_pair'}
        a = {**generated, 'id': 'asap2_v10:ai:'+generated['id'],
             'construction': 'source_matched_student_essay_pair',
             'domain': 'student_argumentative_essay',
             'source': 'asap2_student_essay'}
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
