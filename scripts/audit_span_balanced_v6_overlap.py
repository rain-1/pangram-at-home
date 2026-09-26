"""Check exact and sampled phrase reuse against the protected evaluations."""
from __future__ import annotations

import hashlib
import json

from build_span_balanced_v6 import DATA, phrase_fingerprints

REFERENCES=(
    'span_ai_eval_candidate_v1/test.jsonl',
    'span_human_eval_v2/test.jsonl',
    'span_realistic_eval_v1/test.jsonl',
    'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
    'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
)


def rows(relative):
    with (DATA/relative).open() as file:
        for line in file:
            yield json.loads(line)


def main():
    protected={}
    for name in REFERENCES:
        phrases=set();exact=set();count=0
        for row in rows(name):
            count+=1
            phrases.update(phrase_fingerprints(row['text']))
            exact.add(hashlib.sha256(row['text'].encode()).digest())
        protected[name]={'phrases':phrases,'exact':exact,'count':count,
                         'exact_matches':0,'sampled_phrase_matches':0}
    train_count=0
    for row in rows('span_balanced_v6/train.jsonl'):
        train_count+=1
        phrases=phrase_fingerprints(row['text'])
        digest=hashlib.sha256(row['text'].encode()).digest()
        for item in protected.values():
            item['exact_matches']+=digest in item['exact']
            item['sampled_phrase_matches']+=bool(phrases & item['phrases'])
    print(json.dumps({'train_rows':train_count,
                      'references':{name:{k:v for k,v in item.items() if k not in ('phrases','exact')}
                                    for name,item in protected.items()}},indent=2))


if __name__=='__main__':main()
