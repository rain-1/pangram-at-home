"""Screen candidate JSONL for span validity and overlap with protected data.

Reports IDs and counts only. Phrase matches are review flags, not automatic
evidence of leakage: stock wording and quotations can match legitimately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

DATA=Path('/mnt/f/pangram-at-home/data')
REFERENCES={
    'current_train':'span_essay_paired_v10/train.jsonl',
    'current_val':'span_essay_paired_v10/val.jsonl',
    'human_calibration':'span_human_eval_v2/calibration.jsonl',
    'human_locked_test':'span_human_eval_v2/test.jsonl',
    'external_articles':'span_ai_eval_candidate_v1/test.jsonl',
    'llmtrace_locked_test':'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
    'aitdna_locked_test':'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
    'student_essays_locked_test':'asap2_student_essays_v10/locked_test_human.jsonl',
    'pmc_locked_test':'pmc_publication_v6/test.jsonl',
    'cnn_locked_test':'cnn_dailymail_v1/locked_test.jsonl',
    'archived_epa':'science_articles_v9/archived_epa_science_matters_human.jsonl',
    'archived_magazine':'smithsonian_archive_v10/locked_test_human.jsonl',
}


def words(text):
    return re.findall(r'\w+',text.casefold())


def fingerprint(text):
    tokens=words(text)
    exact=hashlib.sha256(' '.join(tokens).encode()).digest()
    phrases=set()
    for i in range(max(0,len(tokens)-23)):
        digest=hashlib.blake2b(' '.join(tokens[i:i+24]).encode(),digest_size=8).digest()
        if digest[0]<16:
            phrases.add(digest)
    return exact,phrases


def rows(path):
    with path.open() as stream:
        for line_number,line in enumerate(stream,1):
            if line.strip():
                yield line_number,json.loads(line)


def check_spans(row):
    text=row['text'];spans=row.get('spans')
    if spans is None:
        return 'unlabeled'
    if not isinstance(spans,list) or not spans:
        return 'invalid_spans'
    cursor=0;labels=set()
    for span in spans:
        if not isinstance(span,dict):
            return 'invalid_spans'
        start,end,label=span.get('start'),span.get('end'),span.get('label')
        if not (isinstance(start,int) and isinstance(end,int) and
                isinstance(label,int) and label in (0,1) and
                start==cursor and start<end<=len(text)):
            return 'invalid_spans'
        cursor=end;labels.add(label)
    if cursor!=len(text):
        return 'invalid_spans'
    expected={0:'human',1:'ai'}
    if row.get('kind') in ('human','ai') and labels!={0 if row['kind']=='human' else 1}:
        return 'kind_mismatch'
    if row.get('kind')=='mixed' and labels!={0,1}:
        return 'kind_mismatch'
    return 'valid'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=DATA)
    args=parser.parse_args()
    reference={};sizes={}
    for name,relative in REFERENCES.items():
        path=args.data_root/relative
        if not path.exists():
            continue
        exact=set();phrases=set();count=0
        for _,row in rows(path):
            digest,parts=fingerprint(row['text'])
            exact.add(digest);phrases.update(parts);count+=1
        reference[name]=(exact,phrases)
        sizes[name]=count
    counts=Counter();sources=Counter();kinds=Counter();matches=defaultdict(Counter)
    first_hits=defaultdict(list);seen_ids=set();seen_text=set();duplicate_ids=[]
    candidate_sha=hashlib.sha256()
    with args.candidate.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            candidate_sha.update(block)
    for line_number,row in rows(args.candidate):
        counts['rows']+=1
        identifier=str(row.get('id',f'line:{line_number}'))
        text=row.get('text')
        if not isinstance(text,str) or not text.strip():
            counts['missing_or_empty_text']+=1
            continue
        if identifier in seen_ids:
            counts['duplicate_ids']+=1
            if len(duplicate_ids)<20:duplicate_ids.append(identifier)
        seen_ids.add(identifier)
        if not row.get('group_id'):
            counts['missing_group_id']+=1
        if not row.get('source'):
            counts['missing_source']+=1
        sources[str(row.get('source','missing'))]+=1
        kinds[str(row.get('kind','missing'))]+=1
        counts['span_'+check_spans(row)]+=1
        digest,parts=fingerprint(text)
        if digest in seen_text:counts['duplicate_normalized_text']+=1
        seen_text.add(digest)
        for name,(exact,phrases) in reference.items():
            hit_exact=digest in exact
            hit_phrase=bool(parts & phrases)
            if hit_exact or hit_phrase:
                matches[name]['rows_with_match']+=1
                matches[name]['exact_rows']+=hit_exact
                matches[name]['sampled_24_word_phrase_rows']+=hit_phrase
                if len(first_hits[name])<20:
                    first_hits[name].append({'id':identifier,'exact':hit_exact,
                                             'sampled_phrase':hit_phrase})
    result={'candidate':str(args.candidate),'sha256':candidate_sha.hexdigest(),
            'checks':dict(counts),'sources':dict(sources),'kinds':dict(kinds),
            'reference_rows':sizes,'overlap':{name:dict(value) for name,value in matches.items()},
            'first_overlap_ids':dict(first_hits),'first_duplicate_ids':duplicate_ids,
            'caveat':'Sampled phrase hits require source/work review; no hit is not proof of independence.'}
    rendered=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__=='__main__':
    main()
