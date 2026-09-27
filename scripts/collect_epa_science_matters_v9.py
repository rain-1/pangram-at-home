#!/usr/bin/env python3
"""Collect pre-2023 EPA Science Matters features as an independent publisher pool."""
import datetime as dt
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from collect_science_articles_v9 import ROOT, fetch, phrase_fingerprints, protected_phrases

BASE = 'https://www.epa.gov'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'raw').mkdir(exist_ok=True)
    urls = set()
    for year in range(2017, 2023):
        page = BeautifulSoup(fetch(f'{BASE}/sciencematters/{year}-science-matters-stories'), 'html.parser')
        for a in page.select('article a[href]'):
            href = a.get('href', '')
            if href.startswith('/sciencematters/') and 'stories' not in href and 'science-matters' not in href:
                urls.add(urljoin(BASE, href))
    urls = sorted(urls)
    random.Random(20260927).shuffle(urls)
    protected, hashes, screened = protected_phrases()
    rejected = Counter()
    kept = []
    for i, url in enumerate(urls):
        try:
            ident = hashlib.sha256(url.encode()).hexdigest()[:20]
            cached = ROOT/'raw'/f'{ident}.html.gz'
            raw = gzip.open(cached, 'rb').read() if cached.exists() else fetch(url)
        except Exception:
            rejected['fetch_error'] += 1
            continue
        soup = BeautifulSoup(raw, 'html.parser')
        article = soup.select_one('article.article')
        if article is None:
            rejected['no_article'] += 1
            continue
        title_el = article.select_one('h1')
        title = title_el.get_text(' ', strip=True) if title_el else ''
        date_match = re.search(r'Published\s+([A-Z][a-z]+\s+\d{1,2},\s+20\d\d)', article.get_text(' ', strip=True))
        if not date_match:
            rejected['no_publication_date'] += 1
            continue
        date = dt.datetime.strptime(date_match.group(1), '%B %d, %Y').date().isoformat()
        if date >= '2023-01-01':
            rejected['post2022'] += 1
            continue
        for selector in ('figure', 'figcaption', 'aside', 'nav', 'script', 'style', '.related-content', '.share'): 
            for node in article.select(selector):
                node.decompose()
        paragraphs = []
        for p in article.select('p'):
            if p.find_parent(['blockquote', 'table']):
                continue
            t = re.sub(r'\s+', ' ', p.get_text(' ', strip=True)).strip()
            if re.match(r'^(learn more|references|related (links|articles)|for more information)\s*:', t, re.I):
                break
            if re.match(r'^(updated\s+.+?;\s*)?published\s+', t, re.I) or len(t.split()) < 8:
                continue
            if re.search(r'©|all rights reserved|reprinted with permission', t, re.I):
                paragraphs = []
                break
            paragraphs.append(t)
        text = '\n\n'.join(paragraphs)
        if len(text.split()) < 450 or len(text.split()) > 2200:
            rejected['length_or_rights'] += 1
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        fp = phrase_fingerprints(text)
        if digest in hashes or fp & protected:
            rejected['overlap_protected'] += 1
            continue
        hashes.add(digest)
        protected.update(fp)
        row = {
            'id': hashlib.sha256(url.encode()).hexdigest()[:20],
            'text': text,
            'label': 0,
            'spans': [{'start': 0, 'end': len(text), 'label': 0}],
            'kind': 'human',
            'construction': 'dated_editorial_article',
            'source': 'epa_science_matters',
            'source_group': 'epa_science_matters',
            'title': title,
            'author': 'EPA Science Matters editorial staff',
            'url': url,
            'published_at': date,
            'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'rights_evidence_url': 'https://www.epa.gov/web-policies-and-procedures/epa-disclaimers',
            'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'text_sha256': digest,
            'extraction_method': 'article_paragraphs_v1',
            'words': len(text.split()),
        }
        kept.append(row)
        with gzip.open(ROOT/'raw'/f"{row['id']}.html.gz", 'wb') as f:
            f.write(raw)
        if (i+1) % 50 == 0:
            print('EPA', i+1, len(kept), dict(rejected), flush=True)
    with (ROOT/'epa_science_matters.jsonl').open('w') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'source': 'epa_science_matters', 'urls': len(urls), 'accepted': len(kept),
                'rejected': dict(rejected), 'protected_files': screened,
                'note': 'Page-level publication date retained; page may have been edited later. Individual EPA documents can contain third-party rights.'}
    (ROOT/'epa_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
