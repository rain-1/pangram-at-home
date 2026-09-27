"""Collect Cory Doctorow's pre-LLM, CC-licensed essay collections.

Keep only body paragraphs from his essays; exclude forewords, front matter,
interviews, blockquotes, and book boilerplate. Text stays on the external disk.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

from lxml import html
import requests

ROOT=Path('/mnt/f/pangram-at-home/data/candidate_span_sources/round2/cory_doctorow')
BOOKS={
    'content_2008':'https://craphound.com/content/Cory_Doctorow_-_Content.html',
    'context_2011':'https://craphound.com/context/Cory_Doctorow_-_Context.xhtml',
}
START={'content_2008':'Microsoft Research DRM Talk',
       'context_2011':'Jack and the Interstalk'}
END={'content_2008':'About the Author',
     'context_2011':'About the Author'}
EXCLUDE={'content_2008':('And now a brief commercial interlude',
                         'When the Singularity is More Than a Literary Device',
                         'Hope you enjoyed it'),
         'context_2011':()}


def normalized(value: str) -> str:
    return ' '.join(value.split())


def source_bytes(url: str,path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    response=requests.get(url,timeout=30)
    response.raise_for_status()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def main() -> None:
    ROOT.mkdir(parents=True,exist_ok=True)
    records=[];sources={}
    for book,url in BOOKS.items():
        raw=source_bytes(url,ROOT/(book+'.html'))
        sources[book]={'url':url,'sha256':sha256(raw).hexdigest(),'bytes':len(raw)}
        document=html.fromstring(raw)
        started=False
        for heading in document.xpath('//body/h2'):
            title=normalized(heading.text_content())
            if title.startswith(END[book]):
                break
            if title.startswith(START[book]):
                started=True
            if not started or any(title.startswith(bad) for bad in EXCLUDE[book]):
                continue
            paragraphs=[];node=heading.getnext()
            while node is not None and node.tag.lower()!='h2':
                if node.tag.lower()=='p':
                    paragraph=normalized(node.text_content())
                    if paragraph:
                        paragraphs.append(paragraph)
                node=node.getnext()
            text='\n\n'.join(paragraphs)
            if len(re.findall(r'\w+',text))<300:
                continue
            identifier='doctorow:'+book+':'+sha256(title.encode()).hexdigest()[:16]
            records.append({'id':identifier,'group_id':identifier,'text':text,
                'spans':[{'start':0,'end':len(text),'label':0}],
                'kind':'human','source':'craphound.com',
                'author':'Cory Doctorow','book':book,'work_title':title,
                'publication_year':int(book[-4:]),'source_url':url,
                'source_sha256':sources[book]['sha256'],
                'source_license':'CC BY-NC-SA 3.0',
                'provenance':'author-published pre-LLM essay collection; quotations may remain'})
    output=ROOT/'normalized_essays_candidate.jsonl'
    with output.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    manifest={'author':'Cory Doctorow','sources':sources,'rows':len(records),
              'per_book':{name:sum(r['book']==name for r in records) for name in BOOKS},
              'normalized_sha256':sha256(output.read_bytes()).hexdigest(),
              'caveat':'Book publication predates modern LLMs; extraction may retain quotes and current server version is not an archived 2008/2011 capture.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
