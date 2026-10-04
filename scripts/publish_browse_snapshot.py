"""Prepare immutable, PDF-verified browsing pages during upload publication."""
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path
from upload_paper_pdfs import ROOT, BASE, request, remote_objects


def build_rows(objects, published, archive, discovery, enrichment=None):
    public = {p['pdf_key']: p for p in published['items']}
    metadata = {p['key']: p for p in archive['papers']}
    rich = {p['id']: p for p in (enrichment or {}).get('papers', [])}
    rich.update({p['id']: {**p, 'collection':'iclr/2027'} for p in discovery['papers']})
    rows = []
    for key, obj in sorted(objects.items()):
        digest = key.removeprefix('papers/').removesuffix('.pdf')
        if key != f'papers/{digest}.pdf' or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            continue
        p = public.get(key, {}); meta = metadata.get(key, {})
        custom = obj.get('custom_metadata') or {}
        extra = rich.get(digest[:24], {})
        filename = p.get('filename') or meta.get('filename') or custom.get('filename') or digest+'.pdf'
        collection = p.get('collection') or meta.get('collection') or (f"{meta['conference']}/{meta['year']}" if meta.get('conference') and meta.get('year') else custom.get('collection','Other uploads'))
        if extra.get('collection'): collection = extra['collection']
        row = dict(id=p.get('id',digest[:24]),title=p.get('title') or meta.get('title') or custom.get('title') or filename.removesuffix('.pdf'),filename=filename,collection=collection,bytes=obj['size'],classified=bool(p.get('classified')),models=p.get('models',[]))
        row.update({k:v for k,v in extra.items() if k not in ('id','key','bytes','classified','models','collection')})
        rows.append(row)
    assert len({p['id'] for p in rows}) == len(rows), 'Duplicate public IDs'
    return rows


def main(prepared=False, enrich=False):
    out = ROOT/'app/.sites-runtime/browse';out.mkdir(parents=True,exist_ok=True)
    with (out/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_revision=None
        if enrich:
            source=request(BASE+'/indexes/browse/current.json');source_revision=source['revision']
            rows=request(BASE+'/'+source['index_key'])['items']
            metadata={p['id']:p for p in request(BASE+'/indexes/paper-discovery.json')['papers']}
            canonical={p['id']:p['pdf_key'][7:31] for p in request(BASE+'/atlas-public/catalogue.json')['items']}
            for row in rows:
                extra=metadata.get(canonical.get(row['id'],row['id']),{})
                row.update({k:v for k,v in extra.items() if k not in ('id','key','bytes','classified','models','collection')})
        elif prepared:
            receipt=json.loads((out/'receipt.json').read_text())
            assert request(BASE+'/indexes/browse/current.json')['revision']==receipt['revision'], 'Newer snapshot published; rebuild from live inventory'
            rows=json.loads((out/'papers.json').read_text())['items']
        else:
            objects=remote_objects()
            rows=build_rows(objects,request(BASE+'/atlas-public/catalogue.json'),request(BASE+'/indexes/downloaded-papers.json'),request(BASE+'/indexes/iclr2027-discovery.json'),request(BASE+'/indexes/paper-discovery.json') if 'indexes/paper-discovery.json' in objects else None)
        raw=json.dumps({'items':rows},ensure_ascii=False,separators=(',',':')).encode()
        (out/'papers.json').write_bytes(raw)
        # Use the exact same sort/filter implementation as the live search endpoint.
        code="""import fs from 'node:fs';import {browsePage} from './app/lib/browse-page.ts';const data=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));fs.writeFileSync(process.argv[2],JSON.stringify(browsePage(data.items,new URLSearchParams())));"""
        subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',code,str(out/'papers.json'),str(out/'home.json')],cwd=ROOT,check=True)
        home=json.loads((out/'home.json').read_text())
        revision=hashlib.sha256(raw+json.dumps(home,sort_keys=True).encode()).hexdigest()
        home['revision']=revision
        manifest=dict(revision=revision,index_key=f'indexes/browse/{revision}/papers.json',home_key=f'indexes/browse/{revision}/home.json',published_at=time.time(),total=len(rows))
        def put(key,data):
            value=json.loads(data);request(BASE+'/'+key,data,'application/json');assert request(BASE+'/'+key)==value
        put(manifest['index_key'],raw)
        put(manifest['home_key'],json.dumps(home,ensure_ascii=False,separators=(',',':')).encode())
        # Advance only after both immutable objects passed readback verification.
        if source_revision:assert request(BASE+'/indexes/browse/current.json')['revision']==source_revision, 'Newer snapshot published; retry enrichment'
        put('indexes/browse/current.json',json.dumps(manifest).encode())
        (out/'receipt.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Published fast browse snapshot: {len(rows)} papers; first page {len(json.dumps(home).encode()):,} bytes; revision {revision}',flush=True)


if __name__=='__main__':
    import sys
    main(prepared='--prepared' in sys.argv)
