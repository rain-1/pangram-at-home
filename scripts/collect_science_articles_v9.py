#!/usr/bin/env python3
"""Collect dated, human-edited science articles for publication-domain experiments.

Only public NASA/NOAA pages are fetched. The full corpus lives on the F disk;
the repository receives only the script and aggregate manifest.
"""
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_span_balanced_v6 import phrase_fingerprints

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
DATA = Path('/mnt/f/pangram-at-home/data')
SOURCES = {
    'nasa_earth_observatory': {
        'sitemap': 'https://science.nasa.gov/wp-sitemap.xml',
        'sitemap_match': 'wp-sitemap-posts-post-',
        'url_match': '/earth/earth-observatory/',
        'body': '.entry-content',
        'rights': 'https://science.nasa.gov/earth/faq/',
    },
    'noaa_climate': {
        'sitemap': 'https://www.climate.gov/sitemap.xml',
        'url_match': '/news-features/understanding-climate/',
        'body': '.node__content',
        'rights': 'https://www.climate.gov/about',
    },
    'noaa_fisheries': {
        'sitemap': 'https://www.fisheries.noaa.gov/sitemap.xml',
        'url_match': '/feature-story/',
        'body': '.article__content--news',
        'rights': 'https://www.fisheries.noaa.gov/website-policies-and-disclaimers',
    },
}


def fetch(url):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=35, headers={'User-Agent': 'pangram-at-home-research/1.0 (academic dataset provenance audit)'})
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def sitemap_urls(source):
    spec = SOURCES[source]
    root = BeautifulSoup(fetch(spec['sitemap']), 'xml')
    children = [x.get_text(strip=True) for x in root.find_all('loc')]
    if root.find('sitemapindex'):
        children = [u for u in children if not spec.get('sitemap_match') or spec['sitemap_match'] in u]
        pages = []
        for child in children:
            s = BeautifulSoup(fetch(child), 'xml')
            pages += [x.get_text(strip=True) for x in s.find_all('loc')]
        children = pages
    return sorted(set(u for u in children if spec['url_match'] in u))


def article_date(soup, source):
    if source == 'noaa_fisheries':
        el = soup.select_one('.content-header__date')
        raw = el.get_text(' ', strip=True) if el else ''
    else:
        el = soup.find('meta', attrs={'property': 'article:published_time'})
        raw = el.get('content', '') if el else ''
    try:
        if source == 'noaa_fisheries':
            return dt.datetime.strptime(raw, '%B %d, %Y').date().isoformat(), raw
        from email.utils import parsedate_to_datetime
        try:
            return dt.datetime.fromisoformat(raw).date().isoformat(), raw
        except ValueError:
            return parsedate_to_datetime(raw).date().isoformat(), raw
    except (ValueError, TypeError):
        return None, raw


def extract(url, source):
    try:
        raw = fetch(url)
    except requests.RequestException as exc:
        return None, 'fetch_error:' + type(exc).__name__, None
    soup = BeautifulSoup(raw, 'html.parser')
    date, date_raw = article_date(soup, source)
    if not date or date >= '2023-01-01':
        return None, 'missing_or_post2022_date', None
    body = soup.select_one(SOURCES[source]['body'])
    if body is None:
        return None, 'missing_body', None
    for selector in ('figure', 'figcaption', 'aside', 'nav', 'script', 'style', '.caption', '.share', '.related', '.social-share'):
        for node in body.select(selector):
            node.decompose()
    paras = []
    for p in body.find_all(['p', 'h2', 'h3', 'h4']):
        if p.name != 'p':
            if re.search(r'^(references|references & resources|resources|downloads|related)', p.get_text(' ', strip=True), re.I):
                break
            continue
        if p.find_parent(['blockquote', 'table']):
            continue
        t = re.sub(r'\s+', ' ', p.get_text(' ', strip=True)).strip()
        if len(t.split()) < 8 or re.search(r'^(image|photo|figure|credit|references|read more)[:\s]', t, re.I):
            continue
        if re.search(r'©|all rights reserved|reprinted with permission', t, re.I):
            return None, 'possible_third_party_rights', None
        paras.append(t)
    text = '\n\n'.join(paras)
    words = len(text.split())
    if words < 500:
        return None, 'too_short', None
    if words > 2400:
        text = '\n\n'.join(paras[:next((i for i in range(1, len(paras)+1) if len(' '.join(paras[:i]).split()) >= 1800), len(paras))])
    title = (soup.find('meta', attrs={'property': 'og:title'}) or {}).get('content') or (soup.title.get_text(' ', strip=True) if soup.title else '')
    author = None
    if source == 'noaa_climate':
        el = soup.select_one('.metadata__item--author')
        author = re.sub(r'^By\s+', '', el.get_text(' ', strip=True), flags=re.I) if el else None
    elif source == 'nasa_earth_observatory':
        m = re.search(r'(?:NASA Earth Observatory )?[Ss]tory by ([^.\n]{5,120})', body.get_text(' ', strip=True))
        author = m.group(1).strip() if m else None
        if not author:
            return None, 'missing_nasa_story_credit', None
    else:
        author = 'NOAA Fisheries editorial staff'
    modified = soup.find('meta', attrs={'property': 'article:modified_time'})
    row = {
        'id': hashlib.sha256(url.encode()).hexdigest()[:20],
        'text': text,
        'label': 0,
        'spans': [{'start': 0, 'end': len(text), 'label': 0}],
        'kind': 'human',
        'construction': 'dated_editorial_article',
        'source': source,
        'source_group': source,
        'title': title,
        'author': author,
        'url': url,
        'published_at': date,
        'published_raw': date_raw,
        'modified_at': modified.get('content') if modified else None,
        'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'rights_evidence_url': SOURCES[source]['rights'],
        'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'raw_sha256': hashlib.sha256(raw).hexdigest(),
        'extraction_method': 'body_paragraphs_v1',
        'words': len(text.split()),
    }
    return row, 'accepted', raw


def protected_phrases():
    paths = [
        DATA/'span_balanced_v6/train.jsonl',
        DATA/'span_publication_hardneg_v8/train.jsonl',
        DATA/'span_realistic_eval_v1/eval.jsonl',
        DATA/'span_realistic_eval_v1/test.jsonl',
        DATA/'span_human_eval_v2/test.jsonl',
        DATA/'span_human_eval_v2/calibration.jsonl',
        DATA/'span_publication_hardneg_v8/calibration.jsonl',
        DATA/'span_publication_hardneg_v8/test.jsonl',
    ]
    phrases, hashes = set(), set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                t = row.get('text', '')
                if t:
                    hashes.add(hashlib.sha256(t.encode()).hexdigest())
                    phrases.update(phrase_fingerprints(t))
    return phrases, hashes, [str(p) for p in paths if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nasa', type=int, default=250)
    ap.add_argument('--climate', type=int, default=100)
    ap.add_argument('--fisheries', type=int, default=150)
    ap.add_argument('--max-candidates', type=int, default=2500)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'raw').mkdir(exist_ok=True)
    protected, hashes, screened = protected_phrases()
    print('protected phrase hashes', len(protected), 'files', screened, flush=True)
    rng = random.Random(20260927)
    accepted = []
    summary = {}
    for source, target in [('nasa_earth_observatory', args.nasa), ('noaa_climate', args.climate), ('noaa_fisheries', args.fisheries)]:
        urls = sitemap_urls(source)
        rng.shuffle(urls)
        rejected = Counter()
        kept = []
        examined = 0
        # Each source is handled sequentially to keep request rates modest.
        # Small batches bound the number of requests beyond the quota.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
          for offset in range(0, min(len(urls), args.max_candidates), 20):
            if len(kept) >= target:
                break
            batch = urls[offset:offset+20]
            results = list(pool.map(lambda u: extract(u, source), batch))
            for url, (row, reason, raw) in zip(batch, results):
              examined += 1
              if row:
                  fp = phrase_fingerprints(row['text'])
                  if row['text_sha256'] in hashes or fp & protected:
                      reason = 'overlap_protected'
                      row = None
                  else:
                      hashes.add(row['text_sha256'])
                      protected.update(fp)
              if row:
                  kept.append(row)
                  with gzip.open(ROOT/'raw'/f"{row['id']}.html.gz", 'wb') as f:
                      f.write(raw)
              else:
                  rejected[reason] += 1
              if examined % 100 == 0:
                  print(source, examined, len(kept), dict(rejected), flush=True)
        accepted += kept
        summary[source] = {'sitemap_urls': len(urls), 'examined': examined, 'accepted': len(kept), 'rejected': dict(rejected)}
        print(source, summary[source], flush=True)
    with (ROOT/'human_articles.jsonl').open('w') as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'version': 'v9', 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'counts': summary,
                'rows': len(accepted), 'protected_files': screened,
                'note': 'Original publication dates are page metadata. Later CMS modifications may have changed content; retain raw pages and modification timestamps. Screened against protected corpora by sampled 24-word fingerprints.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
