"""Prepare a news-enriched control if the balanced v6 article FPR remains high.

V7 replaces 2,000 DAMASHA examples with 1,000 matched human/AI news pairs
from unused EditLens training groups. It is a prepared candidate, not a run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from build_span_balanced_v6 import DATA, ROOT, phrase_fingerprints, sha
from span_data import window_starts

OUT=DATA/'span_news_enriched_v7'
SEED=20260927


def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite {OUT}')
    rng=random.Random(SEED)
    base=DATA/'span_balanced_v6/train.jsonl'
    rows=[json.loads(line) for line in base.open()]
    removed=[row for row in rows if row['source']=='DAMASHA clean published aggregate']
    rng.shuffle(removed)
    remove_ids={row['id'] for row in removed[:2000]}
    rows=[row for row in rows if row['id'] not in remove_ids]
    used_groups={group for row in rows for group in row.get('source_groups',[])}
    used_hashes={hashlib.sha256(row['text'].encode()).hexdigest() for row in rows}
    protected_phrases=set()
    for relative in ('span_ai_eval_candidate_v1/test.jsonl',
                     'span_human_eval_v2/test.jsonl',
                     'span_realistic_eval_v1/test.jsonl',
                     'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
                     'span_size_curve_v5/size_20000/test_llmtrace.jsonl'):
        with (DATA/relative).open() as file:
            for line in file:
                protected_phrases.update(phrase_fingerprints(json.loads(line)['text']))
    source=DATA/'editlens_pyramid_v1/train_large.parquet'
    grouped={}
    for row in pq.read_table(source).to_pylist():
        if row['source']!='news':continue
        grouped.setdefault(row['group_id'],{})[int(row['label'])]=row
    candidates=[]
    for group_id,pair in grouped.items():
        if group_id in used_groups or set(pair)!={0,1}:continue
        if any(hashlib.sha256(row['text'].encode()).hexdigest() in used_hashes for row in pair.values()):
            continue
        if any(phrase_fingerprints(row['text']) & protected_phrases for row in pair.values()):
            continue
        candidates.append((group_id,pair))
    rng.shuffle(candidates)
    assert len(candidates)>=1000,len(candidates)
    for group_id,pair in candidates[:1000]:
        for label,row in sorted(pair.items()):
            text=row['text'];key=hashlib.sha256(text.encode()).hexdigest()
            assert key not in used_hashes
            used_hashes.add(key)
            rows.append({'id':'v7-news:'+row['text_id'],'text':text,
                         'spans':[{'start':0,'end':len(text),'label':label}],
                         'kind':'ai' if label else 'human','construction':'unaltered_source',
                         'source':'editlens:news','domain':'news','generator':row['model'],
                         'source_ids':[row['text_id']],'source_groups':[group_id],
                         'text_sha256':row['text_sha256']})
    assert len(rows)==20000
    rng.shuffle(rows)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    docs=Counter();windows=Counter();chars=Counter();kinds=Counter()
    for row in rows:
        name=('DAMASHA' if row['source']=='DAMASHA clean published aggregate' else
              'LLMTrace' if row['source']=='LLMTrace_detection' else row['source'])
        docs[name]+=1;kinds[row['kind']]+=1
        n=len(tokenizer(row['text'],add_special_tokens=False)['input_ids'])
        windows[name]+=len(window_starts(n))
        for span in row['spans']:
            chars[str(span['label'])]+=span['end']-span['start']
    assert docs['LLMTrace']==600 and windows['LLMTrace']/sum(windows.values())<=.03
    assert max(docs.values())/len(rows)<.20
    OUT.mkdir()
    with (OUT/'train.jsonl').open('w') as file:
        for row in rows:file.write(json.dumps(row,ensure_ascii=False)+'\n')
    (OUT/'val.jsonl').write_bytes((DATA/'span_balanced_v6/val.jsonl').read_bytes())
    manifest={'role':'prepared news-enrichment control; train only if balanced v6 article FPR remains high',
              'seed':SEED,'rows':len(rows),'replaced_damasha_documents':2000,
              'added_editlens_news_pairs':1000,'source_documents':dict(docs),
              'source_windows':dict(windows),'kinds':dict(kinds),
              'ai_character_fraction':chars['1']/(chars['0']+chars['1']),
              'llmtrace_document_fraction':docs['LLMTrace']/len(rows),
              'llmtrace_window_fraction':windows['LLMTrace']/sum(windows.values()),
              'input_sha256':{'v6_train':sha(base),'editlens_train_large':sha(source)},
              'train_sha256':sha(OUT/'train.jsonl'),'val_sha256':sha(OUT/'val.jsonl'),
              'license_caveat':'EditLens source is CC BY-NC-SA 4.0; local noncommercial research only.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('rows','source_documents','kinds',
          'llmtrace_window_fraction','ai_character_fraction')},indent=2))


if __name__=='__main__':main()
