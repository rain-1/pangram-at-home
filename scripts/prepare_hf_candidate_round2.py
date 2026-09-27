"""Normalize a bounded second intake of HF candidates outside the Git checkout.

These files are candidates. In particular, GRADTEX completion boundaries are
inferred from exact retained context and are not published gold span labels.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/mnt/f/pangram-at-home/data/candidate_span_sources/round2')
DATE_LIMIT = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()[:24]


def full_row(*, identifier: str, group: str, text: str, label: int,
             source: str, **extra: object) -> dict:
    return {'id': identifier, 'group_id': group, 'text': text,
            'spans': [{'start': 0, 'end': len(text), 'label': label}],
            'kind': 'human' if label == 0 else 'ai', 'source': source, **extra}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare_travis() -> dict:
    path = ROOT/'travis_shortstory/data/train.jsonl'
    rows=[];authors=set();titles=set()
    for line in path.open():
        item=json.loads(line)
        story=item['text']
        if not story.strip():
            continue
        identifier=str(item['id'])
        authors.add(item['author']);titles.add((item['author'],item['title']))
        rows.append(full_row(identifier=f'travis:{identifier}',
            group=f'travis:{identifier}',text=story,label=0,
            source='Travis-ML/ShortStory-SFT-jsonl',author=item['author'],
            work_title=item['title'],collection=item['book'],genres=item['genres'],
            source_license=item['license'],provenance='historical published fiction'))
    write_rows(ROOT/'travis_shortstory/normalized_candidate.jsonl',rows)
    return {'rows':len(rows),'authors':len(authors),'distinct_titles':len(titles)}


def prepare_dolly() -> dict:
    rows=[];counts=Counter();seen=set()
    path=ROOT/'dolly/databricks-dolly-15k.jsonl'
    for number,line in enumerate(path.open(),1):
        item=json.loads(line)
        category=item['category'];response=item['response']
        if category not in ('creative_writing','brainstorming','general_qa','open_qa'):
            continue
        if not isinstance(response,str) or len(response.split())<50:
            counts['short_response']+=1;continue
        key=digest(response)
        if key in seen:
            counts['duplicate_response']+=1;continue
        seen.add(key);counts[category]+=1
        rows.append(full_row(identifier=f'dolly:{number}',group=f'dolly:{number}',
            text=response,label=0,source='databricks/databricks-dolly-15k',
            category=category,instruction=item['instruction'],
            source_license='CC BY-SA 3.0',
            provenance='Databricks employee response; contributors instructed not to use generative AI',
            intake_tier='core_open_ended' if category in ('creative_writing','brainstorming')
                        else 'review_question_type'))
    write_rows(ROOT/'dolly/normalized_candidate.jsonl',rows)
    return {'rows':len(rows),'counts':dict(counts)}


def prepare_shp() -> dict:
    rows=[];seen={};counts=Counter();dates=[]
    files=sorted((ROOT/'shp').glob('*/train.json'))
    for path in files:
        subreddit=path.parent.name
        for line in path.open():
            pair=json.loads(line)
            for suffix in ('A','B'):
                timestamp=pair[f'created_at_utc_{suffix}']
                if not isinstance(timestamp,(int,float)) or timestamp>=DATE_LIMIT:
                    counts['excluded_undated_or_post_2022']+=1;continue
                comment_id=str(pair[f'c_root_id_{suffix}'])
                text=pair[f'human_ref_{suffix}']
                if not isinstance(text,str) or len(text.split())<35:
                    counts['excluded_short']+=1;continue
                key=(subreddit,comment_id)
                if key in seen:
                    if seen[key]!=digest(text):
                        counts['conflicting_comment_text']+=1
                    else:
                        counts['repeated_pair_reference']+=1
                    continue
                seen[key]=digest(text)
                dates.append(timestamp)
                rows.append(full_row(identifier=f'shp:{subreddit}:{comment_id}',
                    group=f'shp:{subreddit}:{pair["post_id"]}',text=text,label=0,
                    source='stanfordnlp/SHP',subreddit=subreddit,
                    post_id=pair['post_id'],comment_id=comment_id,
                    comment_created_at_utc=timestamp,
                    provenance='dated Reddit top-level comment; human asserted',
                    source_license='Reddit user content; SHP repository does not grant item rights'))
                counts[subreddit]+=1
    write_rows(ROOT/'shp/normalized_candidate.jsonl',rows)
    return {'rows':len(rows),'posts':len({r['group_id'] for r in rows}),
            'earliest_comment_utc':datetime.fromtimestamp(min(dates),timezone.utc).isoformat(),
            'latest_comment_utc':datetime.fromtimestamp(max(dates),timezone.utc).isoformat(),
            'counts':dict(counts)}


def prepare_story_contrastive() -> dict:
    rows=[];models=Counter();groups=set()
    path=ROOT/'story_contrastive/dataset.jsonl'
    for line in path.open():
        item=json.loads(line);group='story_contrastive:'+item['group_id'];groups.add(group)
        texts=[('human',item['human_response'],0),
               ('gpt56_blind',item['gpt56_blind']['text'],1),
               ('gpt56_conditioned',item['gpt56_conditioned']['text'],1)]
        for model,text,label in texts:
            if not isinstance(text,str) or not text.strip():
                continue
            models[model]+=1
            rows.append(full_row(identifier=f'{group}:{model}',group=group,
                text=text,label=label,source='schonsense/human_ai_story_contrastive_v4',
                generator=model if label else None,genre=item.get('source_genre'),
                source_license='not specified in dataset card',
                generation_mode='human_conditioned_rewrite' if model=='gpt56_conditioned' else 'blind_or_human',
                source_dataset_rows=item.get('source_dataset_rows')))
    write_rows(ROOT/'story_contrastive/normalized_candidate.jsonl',rows)
    return {'rows':len(rows),'groups':len(groups),'models':dict(models)}


def prepare_gradtex() -> dict:
    rows=[];seen_human=set();counts=Counter();groups=set()
    columns=['text','human_source_text','scenario','domain','sub_source','generator_model']
    for item in pq.read_table(ROOT/'gradtex/train.parquet',columns=columns).to_pylist():
        scenario=item['scenario']
        if scenario not in ('complete_beginning','complete_ending'):
            continue
        text=item['text'];human=item['human_source_text']
        if not isinstance(text,str) or not isinstance(human,str) or not text or not human:
            counts['missing_text']+=1;continue
        if scenario=='complete_ending':
            kept=len(os.path.commonprefix((text,human)))
            spans=[{'start':0,'end':kept,'label':0},
                   {'start':kept,'end':len(text),'label':1}]
        else:
            kept=len(os.path.commonprefix((text[::-1],human[::-1])))
            boundary=len(text)-kept
            spans=[{'start':0,'end':boundary,'label':1},
                   {'start':boundary,'end':len(text),'label':0}]
        if kept<200 or kept/len(text)<0.5 or len(text)-kept<80:
            counts['weak_exact_context']+=1;continue
        group='gradtex:'+digest(human);groups.add(group)
        if group not in seen_human:
            seen_human.add(group)
            rows.append(full_row(identifier=group+':human',group=group,text=human,
                label=0,source='elisabeth-pl-pl/GRADTEX',domain=item['domain'],
                sub_source=item['sub_source'],source_license='CC BY 4.0 derivative; MAGE source rights carry through',
                label_quality='MAGE human source'))
        rows.append({'id':f'{group}:{scenario}:{item["generator_model"]}:{digest(text)}',
                     'group_id':group,'text':text,'spans':spans,'kind':'mixed',
                     'source':'elisabeth-pl-pl/GRADTEX','domain':item['domain'],
                     'sub_source':item['sub_source'],'scenario':scenario,
                     'generator':item['generator_model'],
                     'source_license':'CC BY 4.0 derivative; MAGE source rights carry through',
                     'label_quality':'boundary inferred from exact retained context; not published gold'})
        counts[scenario]+=1
    write_rows(ROOT/'gradtex/normalized_completion_candidate.jsonl',rows)
    return {'rows':len(rows),'groups':len(groups),'counts':dict(counts)}


def main() -> None:
    results={name:func() for name,func in (
        ('travis',prepare_travis),('dolly',prepare_dolly),('shp',prepare_shp),
        ('story_contrastive',prepare_story_contrastive),('gradtex',prepare_gradtex))}
    (ROOT/'normalization_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
