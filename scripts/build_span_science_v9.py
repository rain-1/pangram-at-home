"""Add paired, equal-length science articles to the balanced span corpus."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home')
SCIENCE = ROOT/'data/science_articles_v9'
PARENT = ROOT/'data/span_publication_hardneg_v8'
OUTPUT = ROOT/'data/span_science_paired_v9'
TOKENIZER = ROOT/'models/Qwen3-1.7B'
SEED = 20260927
WINDOW = 512


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_rows(row, tokenizer):
    ids = tokenizer.encode(row['text'], add_special_tokens=False)
    if len(ids) < WINDOW:
        return []
    out = []
    for position, start in (('opening', 0), ('ending', len(ids)-WINDOW)):
        text = tokenizer.decode(ids[start:start+WINDOW], skip_special_tokens=True)
        label = int(row['label']) if 'label' in row else int(row['kind'] == 'ai')
        out.append({'id': f'science_v9:{row["id"]}:{position}',
                    'text': text, 'spans': [{'start': 0, 'end': len(text), 'label': label}],
                    'kind': 'ai' if label else 'human',
                    'construction': 'matched_science_article_window',
                    'source': 'science_v9:'+row['source'], 'domain': 'science_publication',
                    'parent_id': row['id'], 'human_id': row.get('human_id', row['id']),
                    'generator': row.get('model'), 'window_position': position,
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return out


def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    humans = {r['id']: r for r in (json.loads(line) for line in (SCIENCE/'train_candidates.jsonl').open())}
    ai_files = [SCIENCE/f'generated_{model}_train.jsonl' for model in ('qwen2_5_3b', 'smollm2_1_7b')]
    ai = [json.loads(line) for path in ai_files for line in path.open()]
    assert len(ai) == len(humans) == 146
    assert {r['human_id'] for r in ai} == set(humans)
    assert all(r['human_text_sha256'] == humans[r['human_id']]['text_sha256'] for r in ai)
    assert len({r['text_sha256'] for r in ai}) == len(ai)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    pairs = []
    for generated in ai:
        human = humans[generated['human_id']]
        h = window_rows(human, tokenizer)
        a = window_rows(generated, tokenizer)
        if len(h) != 2 or len(a) != 2 or generated['words'] < 350:
            raise ValueError(f'Short pair {generated["human_id"]}: {len(h)} {len(a)} {generated["words"]}')
        pairs.extend(h+a)
    assert len(pairs) == 584
    original = [json.loads(line) for line in (PARENT/'train.jsonl').open()]
    rng = random.Random(SEED)
    damasha = [r for r in original if r['source'] == 'DAMASHA clean published aggregate']
    rng.shuffle(damasha)
    removed = {r['id'] for r in damasha[:len(pairs)]}
    rows = [r for r in original if r['id'] not in removed]+pairs
    rng.shuffle(rows)
    assert len(rows) == 20000
    OUTPUT.mkdir(parents=True)
    with (OUTPUT/'train.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    (OUTPUT/'val.jsonl').write_bytes((PARENT/'val.jsonl').read_bytes())
    sources = Counter(r['source'] for r in rows)
    kinds = Counter(r['kind'] for r in rows)
    manifest = {'role': 'paired science publication pilot, source-exclusive from NOAA calibration and archived EPA test',
                'seed': SEED, 'documents': len(rows), 'paired_article_topics': len(ai),
                'added_windows': len(pairs), 'replaced_damasha_documents': len(removed),
                'window_source_tokens': WINDOW, 'added_labels': dict(Counter(r['kind'] for r in pairs)),
                'added_generators': dict(Counter(r['generator'] for r in pairs if r['kind']=='ai')),
                'sources': dict(sources), 'kinds': dict(kinds),
                'human_provenance_caveat': 'NASA and NOAA pages carry pre-2023 dates; current extracted text is not independently archive-verified.',
                'parent_train_sha256': sha(PARENT/'train.jsonl'),
                'science_human_sha256': sha(SCIENCE/'train_candidates.jsonl'),
                'science_ai_sha256': {path.name: sha(path) for path in ai_files},
                'train_sha256': sha(OUTPUT/'train.jsonl'), 'val_sha256': sha(OUTPUT/'val.jsonl')}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('documents','paired_article_topics','added_windows',
                                             'added_labels','added_generators')}, indent=2))


if __name__ == '__main__':
    main()
