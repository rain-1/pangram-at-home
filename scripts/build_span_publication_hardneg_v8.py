"""Prepare a publication hard-negative pilot from dated, attributed CC BY news.

This replaces 500 DAMASHA mixed documents with 500 pure human articles
from five Common Pile publishers. It is an experiment, not a claim that the
publication false-positive problem is solved or that every byline is verified.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import glob
import gzip
import hashlib
import json
from pathlib import Path
import random
import re

from langdetect import DetectorFactory,detect,LangDetectException
from transformers import AutoTokenizer

from build_span_balanced_v6 import DATA, ROOT, phrase_fingerprints, sha
from span_data import window_starts

RAW=DATA/'common_pile_news_v1/raw/v0/documents'
OUT=DATA/'span_publication_hardneg_v8'
SEED=20260927
QUOTAS={'news-factly':100,'news-altnews':100,'news-milwaukeenns':100,
        'news-newcanadianmedia':100,'news-360info':100}
PROTECTED=(
    'span_balanced_v6/train.jsonl',
    'span_ai_eval_candidate_v1/test.jsonl',
    'span_human_eval_v2/test.jsonl',
    'span_realistic_eval_v1/test.jsonl',
    'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
    'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
    'pmc_publication_v6/calibration.jsonl',
    'pmc_publication_v6/test.jsonl',
    'cnn_dailymail_v1/calibration.jsonl',
    'cnn_dailymail_v1/locked_test.jsonl',
)


def normalized_body(row):
    lines=row['text'].splitlines()
    if len(lines)<4:return None
    created=str(row.get('created',''))
    year_match=re.search(r'(?<!\d)(?:19|20)\d{2}(?!\d)',created)
    if not year_match or not 2017<=int(year_match.group())<=2022:return None
    # Common Pile news records start with title, byline, publication date.
    if year_match.group() not in lines[2]:return None
    body='\n'.join(lines[3:]).strip()
    words=list(re.finditer(r'\S+',body))
    if len(words)<500:return None
    body=body[words[0].start():words[min(len(words)-1,849)].end()]
    bad=('cookie policy','accept all cookies','donate now','subscribe to our newsletter')
    if sum(term in body[:1800].lower() for term in bad)>=2:return None
    try:
        if detect(body[:1800])!='en':return None
    except LangDetectException:
        return None
    return body


def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite {OUT}')
    DetectorFactory.seed=SEED
    rng=random.Random(SEED)
    protected=set()
    for relative in PROTECTED:
        with (DATA/relative).open() as file:
            for line in file:
                protected.update(phrase_fingerprints(json.loads(line)['text']))
    candidates=defaultdict(list);rejected=Counter();raw_files=[]
    for path in sorted(glob.glob(str(RAW/'*.jsonl.gz'))):
        raw_files.append(Path(path))
        with gzip.open(path,'rt') as file:
            for line in file:
                row=json.loads(line)
                source=row['source']
                if source not in QUOTAS:continue
                metadata=row.get('metadata') or {}
                if 'creativecommons.org/licenses/by/4.0/' not in metadata.get('license',''):
                    rejected['license']+=1;continue
                author=(metadata.get('author') or '').strip()
                url=(metadata.get('url') or '').strip()
                if not author or not url.startswith('http') or author.lower() in source:
                    rejected['author_or_url']+=1;continue
                body=normalized_body(row)
                if body is None:
                    rejected['date_length_language_or_boilerplate']+=1;continue
                candidates[source].append((row,body))
    selected=[];seen=set()
    for source,quota in QUOTAS.items():
        entries=candidates[source];rng.shuffle(entries)
        used=0
        for raw,body in entries:
            key=hashlib.sha256(body.encode()).hexdigest()
            if key in seen or phrase_fingerprints(body)&protected:
                rejected['duplicate_or_protected_phrase']+=1;continue
            seen.add(key)
            meta=raw['metadata'];url=meta['url']
            selected.append({'id':'commonpile:'+source+':'+str(raw['id']),
                             'text':body,'spans':[{'start':0,'end':len(body),'label':0}],
                             'kind':'human','construction':'published_source_body',
                             'source':'commonpile:'+source,'domain':'published_nonfiction',
                             'source_ids':[url],'source_groups':[url],
                             'author':meta['author'],'created':raw['created'],
                             'license':meta['license'],'canonical_url':url,
                             'text_sha256':key})
            used+=1
            if used==quota:break
        if used!=quota:raise ValueError(f'{source}: needed {quota}, found {used}')
    assert len(selected)==500
    original=[json.loads(line) for line in (DATA/'span_balanced_v6/train.jsonl').open()]
    damasha=[row for row in original if row['source']=='DAMASHA clean published aggregate']
    rng.shuffle(damasha);removed={row['id'] for row in damasha[:500]}
    rows=[row for row in original if row['id'] not in removed]+selected
    assert len(rows)==20000
    rng.shuffle(rows)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    docs=Counter();windows=Counter();chars=Counter();kinds=Counter()
    for row in rows:
        source=('DAMASHA' if row['source']=='DAMASHA clean published aggregate' else
                'LLMTrace' if row['source']=='LLMTrace_detection' else row['source'])
        docs[source]+=1;kinds[row['kind']]+=1
        n=len(tokenizer(row['text'],add_special_tokens=False)['input_ids'])
        windows[source]+=len(window_starts(n))
        for span in row['spans']:chars[str(span['label'])]+=span['end']-span['start']
    assert docs['LLMTrace']==600 and windows['LLMTrace']/sum(windows.values())<=.03
    assert max(docs.values())/len(rows)<.25
    OUT.mkdir()
    with (OUT/'train.jsonl').open('w') as file:
        for row in rows:file.write(json.dumps(row,ensure_ascii=False)+'\n')
    (OUT/'val.jsonl').write_bytes((DATA/'span_balanced_v6/val.jsonl').read_bytes())
    manifest={'role':'prepared publication hard-negative pilot; only local research, no raw Git text',
              'seed':SEED,'documents':len(rows),'replaced_damasha_documents':500,
              'added_human_publication_documents':500,'publisher_quotas':QUOTAS,
              'source_documents':dict(docs),'source_windows':dict(windows),
              'kinds':dict(kinds),'ai_character_fraction':chars['1']/(chars['0']+chars['1']),
              'llmtrace_document_fraction':docs['LLMTrace']/len(rows),
              'llmtrace_window_fraction':windows['LLMTrace']/sum(windows.values()),
              'rejected_candidates':dict(rejected),
              'provenance_caveat':'Pre-2023 publication date and named byline are strong human evidence, not a guarantee of every editing step.',
              'source_card':'https://huggingface.co/datasets/common-pile/news',
              'source_sha256':{str(p):sha(p) for p in raw_files},
              'parent_sha256':sha(DATA/'span_balanced_v6/train.jsonl'),
              'protected_sha256':{name:sha(DATA/name) for name in PROTECTED},
              'train_sha256':sha(OUT/'train.jsonl'),'val_sha256':sha(OUT/'val.jsonl')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('documents','publisher_quotas','kinds',
          'ai_character_fraction','llmtrace_window_fraction','rejected_candidates')},indent=2))


if __name__=='__main__':main()
