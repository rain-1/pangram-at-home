"""Independent pre-2023 PMC publication calibration and locked human test."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from build_span_balanced_v6 import DATA, phrase_fingerprints, sha

OUT=DATA/'pmc_publication_v6'


def main():
    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUT}')
    excluded=set()
    for split in ('train_full','val_full','test_full'):
        path=DATA/'diverse_pyramid_v1'/f'{split}.parquet'
        excluded.update(row['source_id'] for row in
                        pq.read_table(path,columns=['source','source_id']).to_pylist()
                        if row['source']=='paper:pmc_oa')
    # Other source families can contain the same article as prose excerpts.
    other_phrases=set()
    train=DATA/'span_balanced_v6/train.jsonl'
    with train.open() as file:
        for line in file:
            row=json.loads(line)
            if row['source'] in ('DAMASHA clean published aggregate','LLMTrace_detection'):
                other_phrases.update(phrase_fingerprints(row['text']))
    source=DATA/'pmc_pilot_v1/documents.jsonl.gz'
    rows=[];rejected=0
    with gzip.open(source,'rt') as file:
        for line in file:
            row=json.loads(line)
            if row['source_id'] in excluded or row['publication_date'][:4] > '2022':
                continue
            if row['license'] not in ('CC BY 4.0','CC0 1.0'):
                continue
            body=row['body'] or ''
            matches=list(re.finditer(r'\S+',body))
            if len(matches)<500:
                continue
            text=body[matches[0].start():matches[min(849,len(matches)-1)].end()]
            if phrase_fingerprints(text)&other_phrases:
                rejected+=1;continue
            rows.append({'id':'pmc-publication:'+row['source_id'],'text':text,
                         'spans':[{'start':0,'end':len(text),'label':0}],
                         'kind':'human','domain':'published_scientific_article',
                         'source':'pmc_oa_pre2023','source_id':row['source_id'],
                         'group_id':row['source_id'],'publication_date':row['publication_date'],
                         'license':row['license'],'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    random.Random(20260926).shuffle(rows)
    assert len(rows)>=400,len(rows)
    OUT.mkdir()
    files={}
    for name,portion in (('calibration',rows[:200]),('test',rows[200:])):
        path=OUT/f'{name}.jsonl'
        with path.open('w') as file:
            for row in portion:
                file.write(json.dumps(row,ensure_ascii=False)+'\n')
        files[name]={'rows':len(portion),'sha256':sha(path)}
    (OUT/'manifest.json').write_text(json.dumps({
        'role':'pre-2023 human publication calibration and held-out test',
        'source':'NCBI PMC CC BY/CC0 article bodies',
        'article_ids_excluded_if_in_diverse_train_val_test':len(excluded),
        'candidate_rejected_for_sampled_24word_overlap_with_damasha_or_llmtrace':rejected,
        'disjoint_by_pmc_article_id':True,
        'note':'Calibration and test articles are different PMC IDs; preserve test as locked.',
        'source_sha256':sha(source),'balanced_train_sha256':sha(train),'files':files},indent=2)+'\n')
    print(json.dumps(files))


if __name__=='__main__':main()
