"""Acquire bounded pre-2013 first-party Aaron Swartz essays for private research.

The official site does not present a blanket item license; do not redistribute
the fetched text. Site attribution and historical dates are evidence, not a
keystroke-level provenance record for the current served copy.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urljoin

from lxml import html
import requests

BASE='http://www.aaronsw.com/weblog/'
ROOT=Path('/mnt/f/pangram-at-home/data/candidate_span_sources/round2/aaron_swartz')
MAX_PAGES=120


def clean(value: str) -> str:
    return ' '.join(value.split())


def fetch_one(item: tuple[str,str,str]) -> dict | None:
    title,slug,date=item
    url=urljoin(BASE,slug)
    raw_path=ROOT/'raw'/(slug+'.html')
    try:
        if raw_path.exists():
            raw=raw_path.read_bytes()
        else:
            response=requests.get(url,timeout=20)
            response.raise_for_status()
            raw=response.content
            raw_path.parent.mkdir(parents=True,exist_ok=True)
            raw_path.write_bytes(raw)
        document=html.fromstring(raw)
        content=document.cssselect('div.content')
        if len(content)!=1:
            return None
        paragraphs=[]
        for child in content[0]:
            if child.tag.lower() in ('p','li') and 'footnote' not in (child.get('class') or ''):
                value=clean(child.text_content())
                if value:paragraphs.append(value)
        text='\n\n'.join(paragraphs)
        if len(re.findall(r'\w+',text))<300:
            return None
        identifier='aaron_swartz:'+slug
        return {'id':identifier,'group_id':identifier,'text':text,
            'spans':[{'start':0,'end':len(text),'label':0}],
            'kind':'human','source':'aaronsw.com/weblog',
            'author':'Aaron Swartz','work_title':title,
            'publication_date':date,'source_url':url,
            'raw_sha256':sha256(raw).hexdigest(),
            'source_license':'not established; private research candidate',
            'provenance':'first-party archive dates post before 2013; current copy could have later edits'}
    except (requests.RequestException,ValueError,UnicodeError):
        return None


def main() -> None:
    ROOT.mkdir(parents=True,exist_ok=True)
    archive_path=ROOT/'fullarchive.html'
    if archive_path.exists():
        raw=archive_path.read_bytes()
    else:
        response=requests.get(urljoin(BASE,'fullarchive'),timeout=20)
        response.raise_for_status();raw=response.content
        archive_path.write_bytes(raw)
    document=html.fromstring(raw)
    candidates=[];seen=set()
    for paragraph in document.xpath('//p'):
        links=paragraph.xpath('./a[1]')
        if not links:continue
        slug=links[0].get('href') or ''
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',slug) or slug in seen:
            continue
        match=re.search(r'\(([A-Za-z]+\s+\d+,\s+\d{4})\)',
                        clean(paragraph.text_content()))
        if not match:continue
        published=datetime.strptime(clean(match.group(1)),'%B %d, %Y')
        if published.year>2012:continue
        seen.add(slug)
        candidates.append((clean(links[0].text_content()),slug,published.date().isoformat()))
        if len(candidates)>=MAX_PAGES:break
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=[record for record in pool.map(fetch_one,candidates) if record]
    output=ROOT/'normalized_essays_candidate.jsonl'
    with output.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    manifest={'author':'Aaron Swartz','archive_url':urljoin(BASE,'fullarchive'),
              'archive_sha256':sha256(raw).hexdigest(),
              'candidate_pages':len(candidates),'essay_rows':len(records),
              'normalized_sha256':sha256(output.read_bytes()).hexdigest(),
              'publication_years':{str(year):sum(r['publication_date'].startswith(str(year)) for r in records)
                                   for year in range(2000,2013)},
              'rights':'Site/article license not established; text retained privately on external drive.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
