"""Create local source-group holdouts from acquired OpAI train candidates.

These are derived splits of the upstream *train* files, not official OpAI
validation/test sets. Identical normalized text cannot cross source groups.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/mnt/f/pangram-at-home/data/candidate_span_sources')
INPUTS={
    'reports':ROOT/'ai/opai_bench/default_train_reports_gpt54/normalized_train.jsonl',
    'abstracts':ROOT/'real_mixed/opai/abstracts_gpt-5.4-nano_train.normalized.jsonl',
}
OUTPUT=ROOT/'prepared_opai_v1'


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def text_key(text):
    normalized=' '.join(re.findall(r'\w+',text.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_name(group):
    bucket=int.from_bytes(hashlib.sha256(('opai-v1:'+group).encode()).digest()[:4],'big')%10
    return 'train' if bucket<8 else ('validation' if bucket==8 else 'test')


def prepare(name,path):
    groups=defaultdict(list);text_groups=defaultdict(set);source_count=0
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            group=row.get('group_id')
            upstream_split=row.get('source_split',row.get('split'))
            if not group or upstream_split!='train':
                raise ValueError(f'Missing group or unexpected upstream split in {name}')
            if not row.get('spans') or row['spans'][0]['start']!=0 or row['spans'][-1]['end']!=len(row['text']):
                raise ValueError(f'Invalid span coverage: {row.get("id")}')
            groups[group].append(row)
            text_groups[text_key(row['text'])].add(group)
            source_count+=1
    discarded_groups=set()
    for matching_groups in text_groups.values():
        if len(matching_groups)>1:
            discarded_groups.update(sorted(matching_groups)[1:])
    output=OUTPUT/name
    output.mkdir(parents=True,exist_ok=True)
    stats={s:{'groups':0,'rows':0,'kinds':Counter()} for s in ('train','validation','test')}
    seen_hashes={s:set() for s in stats}
    duplicates_within_groups=0
    with (output/'train.jsonl').open('w') as train, \
         (output/'validation.jsonl').open('w') as validation, \
         (output/'test.jsonl').open('w') as test:
        streams={'train':train,'validation':validation,'test':test}
        for group,rows in sorted(groups.items()):
            if group in discarded_groups:
                continue
            split=split_name(name+':'+group)
            stats[split]['groups']+=1
            in_group=set()
            for row in sorted(rows,key=lambda r:(int(r.get('version_index',0)),r['id'])):
                digest=text_key(row['text'])
                if digest in in_group:
                    duplicates_within_groups+=1
                    continue
                in_group.add(digest)
                if digest in seen_hashes[split]:
                    raise ValueError(f'Cross-group duplicate survived: {row["id"]}')
                seen_hashes[split].add(digest)
                row['candidate_split']=split
                streams[split].write(json.dumps(row,ensure_ascii=False)+'\n')
                stats[split]['rows']+=1
                stats[split]['kinds'][row['kind']]+=1
    assert not any(seen_hashes[a]&seen_hashes[b]
                   for a,b in (('train','validation'),('train','test'),('validation','test')))
    result={'upstream_role':'all source rows came from official OpAI train',
            'source_file':str(path),'source_sha256':sha_file(path),
            'source_rows':source_count,'source_groups':len(groups),
            'excluded_cross_group_duplicate_groups':sorted(discarded_groups),
            'duplicate_rows_within_groups_removed':duplicates_within_groups,
            'split_rule':'SHA256(opai-v1:domain:group_id) modulo 10; 8 train, 1 validation, 1 test',
            'splits':{s:{'groups':item['groups'],'rows':item['rows'],
                         'kinds':dict(item['kinds']),
                         'sha256':sha_file(output/(s+'.jsonl'))}
                      for s,item in stats.items()}}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    result={name:prepare(name,path) for name,path in INPUTS.items()}
    (OUTPUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{'source_rows':r['source_rows'],'splits':r['splits']}
                      for name,r in result.items()},indent=2))


if __name__=='__main__':
    main()
