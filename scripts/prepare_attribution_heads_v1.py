"""Create prompt-disjoint generator and work-disjoint author probe splits."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/mnt/f/pangram-at-home/data')
ARENA = Path('/home/ubuntu/.cache/huggingface/hub/datasets--woog--arena-prose-100-49-models/snapshots/66298b561a69c6ed6a9a3c4c79eeedb1f754ba49/data/test-00000-of-00001.parquet')
SOURCE_AUTHORS = ROOT/'author_attribution_probe_v1'
OUTPUT = ROOT/'attribution_heads_v1'


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, rows: list[dict]) -> dict:
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    return {'rows':len(rows),'sha256':digest(path),
            'labels':dict(sorted(Counter(row['label'] for row in rows).items()))}


def arena() -> tuple[dict,list[dict],dict]:
    source=pq.read_table(ARENA).to_pylist()
    assert len(source)==5000
    prompts=defaultdict(list)
    excluded=[]
    for row in source:
        text=row['response'] or ''
        if not text.strip():
            excluded.append({'id':row['id'],'model_id':row['model_id'],
                             'prompt_id':row['prompt_id'],'reason':'empty_response'})
            continue
        prompts[row['prompt_id']].append(row)
    labels={row['model_id'] for row in source}
    complete=[pid for pid,rows in prompts.items()
              if {row['model_id'] for row in rows}==labels]
    assert len(labels)==50 and len(prompts)==100 and len(complete)==98
    ordered=sorted(complete,key=lambda pid:hashlib.sha256(
                   ('attribution-heads-v1:'+pid).encode()).hexdigest())
    test=set(ordered[:3]);val=set(ordered[3:6])
    splits={'train':[],'val':[],'test':[]}
    for pid,records in prompts.items():
        split='test' if pid in test else 'val' if pid in val else 'train'
        for row in records:
            splits[split].append({'id':row['id'],'text':row['response'],
                'label':row['model_id'],'group_id':pid,'source':'arena_prose',
                'response_words':row['response_words'],
                'mechanically_eligible':row['mechanically_eligible'],
                'prompt_category':row['prompt_category']})
    for split in splits:
        splits[split].sort(key=lambda row:row['id'])
    assert all(Counter(row['label'] for row in splits[split])=={label:3 for label in labels}
               for split in ('val','test'))
    assert len(splits['train'])==4698 and len(splits['val'])==150 and len(splits['test'])==150
    meta={'source':'woog/arena-prose-100-49-models',
          'revision':'66298b561a69c6ed6a9a3c4c79eeedb1f754ba49',
          'source_sha256':digest(ARENA),'source_rows':len(source),
          'usable_rows':sum(map(len,splits.values())),
          'excluded_empty_responses':excluded,'labels':len(labels),
          'split_policy':'three shared prompts for test, three different prompts for validation; remaining prompts train',
          'test_prompt_ids':sorted(test),'val_prompt_ids':sorted(val)}
    return splits,excluded,meta


def authors() -> tuple[dict,dict]:
    originals=[]
    input_hashes={}
    for split in ('train','val','test'):
        path=SOURCE_AUTHORS/(split+'.jsonl')
        input_hashes[split]=digest(path)
        originals += [json.loads(line) for line in path.open()]
    by_label=defaultdict(list)
    for row in originals:
        by_label[row['author_id']].append(row)
    assert len(by_label)==4 and len(originals)==299
    splits={'train':[],'val':[],'test':[]}
    for label,rows in by_label.items():
        assert len({row['probe_group_id'] for row in rows})==len(rows)
        ordered=sorted(rows,key=lambda row:(row.get('publication_date') or '9999',
                                             row['probe_group_id']))
        for split,selected in (('train',ordered[:-6]),('val',ordered[-6:-3]),
                               ('test',ordered[-3:])):
            for row in selected:
                splits[split].append({'id':row['document_id'],'text':row['text'],
                    'label':label,'group_id':row['probe_group_id'],
                    'source':'named_writer','publication_date':row.get('publication_date'),
                    'human_origin_status':row['human_origin_status']})
    for split in splits:
        splits[split].sort(key=lambda row:row['id'])
    assert all(set(row['group_id'] for row in splits[a]).isdisjoint(
               row['group_id'] for row in splits[b])
               for a,b in (('train','val'),('train','test'),('val','test')))
    assert all(set(Counter(row['label'] for row in splits[split]).values())=={3}
               for split in ('val','test'))
    return splits,{'source':'four published-writer collection',
                   'source_split_sha256':input_hashes,'source_rows':len(originals),
                   'labels':len(by_label),
                   'split_policy':'whole grouped essays; newest three test, preceding three validation per author'}


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    tasks={}
    arena_splits,excluded,arena_meta=arena()
    author_splits,author_meta=authors()
    for task,splits,meta in [('arena',arena_splits,arena_meta),
                             ('authors',author_splits,author_meta)]:
        folder=OUTPUT/task
        folder.mkdir()
        meta['splits']={split:write(folder/(split+'.jsonl'),rows)
                        for split,rows in splits.items()}
        (folder/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        tasks[task]=meta
    (OUTPUT/'arena/excluded_empty.jsonl').write_text(''.join(
        json.dumps(row)+'\n' for row in excluded))
    (OUTPUT/'manifest.json').write_text(json.dumps(tasks,indent=2)+'\n')
    print(json.dumps({task:{split:info['splits'][split]['rows']
                            for split in ('train','val','test')}
                      for task,info in tasks.items()},indent=2))


if __name__=='__main__':
    main()
