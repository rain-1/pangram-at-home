"""Reserve historical journalist-written news articles for source-transfer tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from build_span_balanced_v6 import DATA, phrase_fingerprints, sha

RAW=DATA/'cnn_dailymail_v1/raw/3.0.0'
OUT=DATA/'cnn_dailymail_v1'


def main():
    for name in ('calibration.jsonl','locked_test.jsonl','manifest.json'):
        if (OUT/name).exists():
            raise SystemExit(f'Refusing to overwrite {OUT/name}')
    protected=set()
    for file in [DATA/'span_balanced_v6/train.jsonl',
                 DATA/'span_ai_eval_candidate_v1/test.jsonl']:
        with file.open() as handle:
            for line in handle:
                protected.update(phrase_fingerprints(json.loads(line)['text']))
    rng=random.Random(20260926)
    summaries={};seen=set()
    for split,limit,name in [('validation',300,'calibration'),('test',500,'locked_test')]:
        table=pq.read_table(RAW/f'{split}-00000-of-00001.parquet',columns=['article','id'])
        candidates=table.to_pylist();rng.shuffle(candidates)
        output=[];rejected={'short_or_long':0,'duplicate':0,'phrase_overlap':0}
        for row in candidates:
            text=row['article'].strip()
            words=len(re.findall(r'\S+',text))
            if not 450<=words<=1200:
                rejected['short_or_long']+=1;continue
            fingerprint=hashlib.sha256(text.encode()).hexdigest()
            if fingerprint in seen:
                rejected['duplicate']+=1;continue
            if phrase_fingerprints(text)&protected:
                rejected['phrase_overlap']+=1;continue
            seen.add(fingerprint)
            output.append({'id':'cnn-dm:'+split+':'+row['id'],'text':text,
                           'spans':[{'start':0,'end':len(text),'label':0}],
                           'kind':'human','source':'cnn_dailymail_2007_2015',
                           'domain':'published_news_article','source_id':row['id'],
                           'group_id':row['id'],'text_sha256':fingerprint})
            if len(output)==limit:break
        assert len(output)==limit,(split,len(output),rejected)
        path=OUT/f'{name}.jsonl'
        with path.open('w') as handle:
            for row in output:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        summaries[name]={'rows':len(output),'sha256':sha(path),'rejected_candidates':rejected}
    manifest={'source':'abisee/cnn_dailymail version 3.0.0',
              'source_page':'https://huggingface.co/datasets/abisee/cnn_dailymail',
              'provenance':'CNN journalist articles 2007-2015; Daily Mail journalist articles 2010-2015, per dataset card',
              'role':'source-transfer human-only calibration and locked test; not in Qwen balanced v6 training',
              'calibration_upstream_split':'validation','locked_test_upstream_split':'test',
              'raw_sha256':{split:sha(RAW/f'{split}-00000-of-00001.parquet')
                            for split in ('validation','test')},
              'overlap_filter':'sampled 1/16 24-word phrases against balanced v6 train and external article stress set',
              'files':summaries}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
