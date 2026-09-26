"""Hold out dated CC BY human articles from publishers absent in v8 training."""
from __future__ import annotations

from collections import Counter
import glob
import gzip
import hashlib
import json
from pathlib import Path
import random

from langdetect import DetectorFactory

from build_span_balanced_v6 import DATA,phrase_fingerprints,sha
from build_span_publication_hardneg_v8 import RAW,normalized_body

OUT=DATA/'common_pile_publication_eval_v8'
SOURCES={'news-freedom','news-zimfact','news-mekongeye','news-minorityafrica'}
PROTECTED=('span_publication_hardneg_v8/train.jsonl',
           'span_ai_eval_candidate_v1/test.jsonl',
           'span_size_curve_v5/size_20000/test_llmtrace.jsonl')


def main():
    if OUT.exists():raise SystemExit(f'Refusing to overwrite {OUT}')
    DetectorFactory.seed=20260928
    protected=set()
    for relative in PROTECTED:
        with (DATA/relative).open() as file:
            for line in file:protected.update(phrase_fingerprints(json.loads(line)['text']))
    candidates=[];seen=set()
    for path in sorted(glob.glob(str(RAW/'*.jsonl.gz'))):
        with gzip.open(path,'rt') as file:
            for line in file:
                raw=json.loads(line)
                if raw['source'] not in SOURCES:continue
                meta=raw.get('metadata') or {}
                if 'creativecommons.org/licenses/by/4.0/' not in meta.get('license',''):continue
                if not meta.get('author') or not str(meta.get('url','')).startswith('http'):continue
                body=normalized_body(raw)
                if body is None:continue
                h=hashlib.sha256(body.encode()).hexdigest()
                if h in seen or phrase_fingerprints(body)&protected:continue
                seen.add(h)
                candidates.append({'id':'commonpile:'+raw['source']+':'+str(raw['id']),
                                   'text':body,'spans':[{'start':0,'end':len(body),'label':0}],
                                   'kind':'human','construction':'published_source_body',
                                   'source':'commonpile:'+raw['source'],'domain':'published_nonfiction',
                                   'source_id':meta['url'],'group_id':meta['url'],
                                   'author':meta['author'],'created':raw['created'],
                                   'license':meta['license'],'canonical_url':meta['url'],
                                   'text_sha256':h})
    rng=random.Random(20260928);rng.shuffle(candidates)
    selected=[];counts=Counter()
    for row in candidates:
        if counts[row['source']]>=100:continue
        selected.append(row);counts[row['source']]+=1
        if len(selected)==300:break
    if len(selected)<300:raise ValueError(f'Need 300, found {len(selected)}: {counts}')
    OUT.mkdir()
    files={}
    for name,portion in (('calibration',selected[:100]),('locked_test',selected[100:])):
        path=OUT/f'{name}.jsonl'
        with path.open('w') as file:
            for row in portion:file.write(json.dumps(row,ensure_ascii=False)+'\n')
        files[name]={'rows':len(portion),'sources':dict(Counter(r['source'] for r in portion)),
                     'sha256':sha(path)}
    manifest={'role':'source-held-out historical human publication calibration and test',
              'training_publishers':['news-factly','news-altnews','news-milwaukeenns',
                                     'news-newcanadianmedia','news-360info'],
              'held_out_publishers':sorted(SOURCES),
              'license':'per-document CC BY 4.0 metadata, local research only',
              'provenance_caveat':'Publication date and byline do not prove every edit was human.',
              'protected_sha256':{p:sha(DATA/p) for p in PROTECTED},'files':files}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(files,indent=2))


if __name__=='__main__':main()
