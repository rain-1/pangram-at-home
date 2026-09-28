"""Halve the GRADTEX dose in v12 while retaining every new human source."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

ROOT=Path('/mnt/f/pangram-at-home')
V10=ROOT/'data/span_essay_paired_v10'
V12=ROOT/'data/span_new_sources_v12'
OUTPUT=ROOT/'data/span_new_sources_v13'
SEED=20260928


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write(path: Path,rows: list[dict]) -> None:
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    v10=load(V10/'train.jsonl')
    v12=load(V12/'train.jsonl')
    assert len(v10)==len(v12)==20000
    assert digest(V12/'train.jsonl')=='22f648c6333c3e332aa3749ab759c059fcdcc68bb255e668dafa345c82186034'
    v12_ids={row['id'] for row in v12}
    gradtex=[row for row in v12 if row['source']=='elisabeth-pl-pl/GRADTEX'
             and row['kind']=='mixed']
    restored=[row for row in v10 if row['id'] not in v12_ids
              and row['source']=='DAMASHA clean published aggregate'
              and row['kind']=='mixed']
    assert len(gradtex)==len(restored)==1000
    rng=random.Random(SEED)
    rng.shuffle(gradtex)
    rng.shuffle(restored)
    removed={row['id'] for row in gradtex[:500]}
    rows=[row for row in v12 if row['id'] not in removed]+restored[:500]
    rng.shuffle(rows)
    assert len(rows)==20000 and len({row['id'] for row in rows})==20000
    holdout=load(V12/'new_source_holdout.jsonl')
    assert not {row['id'] for row in holdout} & {row['id'] for row in rows}
    assert not {row['group_id'] for row in holdout if 'group_id' in row} & {
        row['group_id'] for row in rows if 'group_id' in row}
    OUTPUT.mkdir(parents=True)
    write(OUTPUT/'train.jsonl',rows)
    (OUTPUT/'val.jsonl').write_bytes((V12/'val.jsonl').read_bytes())
    (OUTPUT/'new_source_holdout.jsonl').write_bytes((V12/'new_source_holdout.jsonl').read_bytes())
    manifest={'role':'v12 follow-up: 500 GRADTEX mixed, 500 DAMASHA mixed restored',
              'seed':SEED,'documents':len(rows),
              'v12_train_sha256':digest(V12/'train.jsonl'),
              'train_sha256':digest(OUTPUT/'train.jsonl'),
              'val_sha256':digest(OUTPUT/'val.jsonl'),
              'new_source_holdout_sha256':digest(OUTPUT/'new_source_holdout.jsonl'),
              'gradtex_mixed_documents':sum(row['source']=='elisabeth-pl-pl/GRADTEX'
                                             and row['kind']=='mixed' for row in rows),
              'restored_damasha_documents':500,
              'new_human_documents':860,
              'sources':dict(Counter(row['source'] for row in rows)),
              'kinds':dict(Counter(row['kind'] for row in rows))}
    assert manifest['gradtex_mixed_documents']==500
    assert manifest['kinds']=={'human':6425,'mixed':7663,'ai':5912}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in
        ('documents','gradtex_mixed_documents','restored_damasha_documents','new_human_documents','kinds')},indent=2))


if __name__=='__main__':
    main()
