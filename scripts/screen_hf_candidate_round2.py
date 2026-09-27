"""Exclude overlapping source groups and normalized duplicates from round-2 candidates."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path('/mnt/f/pangram-at-home/data/candidate_span_sources/round2')
SOURCES={
    'travis_shortstory':'normalized_candidate.jsonl',
    'dolly':'normalized_candidate.jsonl',
    'shp':'normalized_candidate.jsonl',
    'story_contrastive':'normalized_candidate.jsonl',
    'gradtex':'normalized_completion_candidate.jsonl',
    'cory_doctorow':'normalized_essays_candidate.jsonl',
    'aaron_swartz':'normalized_essays_candidate.jsonl',
}


def key(text: str) -> str:
    words=re.findall(r'\w+',text.casefold())
    return sha256(' '.join(words).encode()).hexdigest()


def screen(name: str,filename: str) -> dict:
    directory=ROOT/name
    input_path=directory/filename
    if not input_path.exists():
        return {'status':'not_acquired'}
    audit_path=directory/('completion_intake_audit.json' if name=='gradtex' else 'intake_audit.json')
    if not audit_path.exists():
        raise FileNotFoundError(f'Audit first: {audit_path}')
    overlap_path=directory/'overlap_groups.json'
    overlaps=json.loads(overlap_path.read_text()) if overlap_path.exists() else {}
    cross_path=directory/'cross_candidate_groups.json'
    cross=json.loads(cross_path.read_text()) if cross_path.exists() else {}
    excluded=set().union(*(set(groups) for groups in overlaps.values())) if overlaps else set()
    cross_excluded=set().union(*(set(groups) for groups in cross.values())) if cross else set()
    excluded.update(cross_excluded)
    rows=[json.loads(line) for line in input_path.open()]
    text_groups=defaultdict(set)
    for row in rows:
        if row['group_id'] not in excluded:
            text_groups[key(row['text'])].add(row['group_id'])
    cross_duplicate_groups=set()
    for groups in text_groups.values():
        if len(groups)>1:
            cross_duplicate_groups.update(sorted(groups)[1:])
    excluded.update(cross_duplicate_groups)
    output=directory/'screened_candidate.jsonl'
    seen=set();kept=0;duplicate_rows=0
    with output.open('w') as stream:
        for row in rows:
            if row['group_id'] in excluded:
                continue
            fingerprint=key(row['text'])
            if fingerprint in seen:
                duplicate_rows+=1;continue
            seen.add(fingerprint)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            kept+=1
    result={'input_rows':len(rows),'screened_rows':kept,
            'excluded_overlap_groups':len(set().union(*(set(g) for g in overlaps.values()))) if overlaps else 0,
            'excluded_cross_candidate_groups':len(cross_excluded),
            'excluded_cross_group_duplicate_groups':len(cross_duplicate_groups),
            'dropped_duplicate_rows_within_group':duplicate_rows,
            'screened_sha256':sha256(output.read_bytes()).hexdigest(),
            'role':'candidate only; check source rights and provenance before mix intake'}
    (directory/'screen_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    result={name:screen(name,filename) for name,filename in SOURCES.items()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
