#!/usr/bin/env python3
"""Fetch dated Common Crawl WARC captures and extract historical EPA prose."""
import argparse
import gzip
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def grams(text, n=13):
    words = re.findall(r'\w+', text.casefold())
    return {' '.join(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def extract_epa(raw):
    soup = BeautifulSoup(raw, 'html.parser')
    article = soup.select_one('article.article')
    if article is None:
        return None
    for selector in ('figure', 'figcaption', 'aside', 'nav', 'script', 'style', '.related-content', '.share'):
        for node in article.select(selector):
            node.decompose()
    paras = []
    for p in article.select('p'):
        if p.find_parent(['blockquote', 'table']):
            continue
        t = re.sub(r'\s+', ' ', p.get_text(' ', strip=True)).strip()
        if re.match(r'^(learn more|references|related (links|articles)|for more information)\s*:', t, re.I):
            break
        if re.match(r'^(updated\s+.+?;\s*)?published\s+', t, re.I) or len(t.split()) < 8:
            continue
        paras.append(t)
    text = '\n\n'.join(paras)
    return text if len(text.split()) >= 400 else None


def extract_fisheries(raw):
    soup = BeautifulSoup(raw, 'html.parser')
    article = soup.select_one('.article__content--news')
    if article is None:
        return None
    for selector in ('figure', 'figcaption', 'aside', 'nav', 'script', 'style', '.caption',
                     '.share', '.related', '.social-share'):
        for node in article.select(selector):
            node.decompose()
    paras = []
    for p in article.select('p'):
        if p.find_parent(['blockquote', 'table']):
            continue
        t = re.sub(r'\s+', ' ', p.get_text(' ', strip=True)).strip()
        if re.match(r'^(story by|photos? by|for more information|to contact|media contact)\b', t, re.I):
            break
        if len(t.split()) < 8 or re.match(r'^(image|photo|figure|credit|references|read more)[:\s]', t, re.I):
            continue
        paras.append(t)
    text = '\n\n'.join(paras)
    return text if len(text.split()) >= 400 else None


def fetch_capture(item):
    url = 'https://data.commoncrawl.org/'+item['filename']
    start, length = item['offset'], item['length']
    r = requests.get(url, headers={'Range': f'bytes={start}-{start+length-1}'}, timeout=35)
    r.raise_for_status()
    if r.status_code != 206 or len(r.content) != length:
        raise ValueError('Unexpected Range response')
    record = gzip.decompress(r.content)
    _, payload = record.split(b'\r\n\r\n', 1)
    http_header, html = payload.split(b'\r\n\r\n', 1)
    if html.startswith(b'\x1f\x8b'):
        html = gzip.decompress(html)
    return r.content, html


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='epa_science_matters',
                   choices=['epa_science_matters', 'noaa_fisheries'])
    p.add_argument('--interval', type=float, default=0.25)
    args = p.parse_args()
    ledger = [json.loads(x) for x in (ROOT/f'archive_ledger_{args.source}.jsonl').open()]
    source_file = 'epa_science_matters.jsonl' if args.source == 'epa_science_matters' else 'human_articles.jsonl'
    current = {r['id']: r for r in (json.loads(x) for x in (ROOT/source_file).open())
               if r['source'] == args.source}
    (ROOT/'raw_archive').mkdir(exist_ok=True)
    accepted, rejected, overlaps = [], Counter(), []
    for i, item in enumerate(ledger):
        try:
            compressed, html = fetch_capture(item)
            text = extract_epa(html) if args.source == 'epa_science_matters' else extract_fisheries(html)
        except Exception as exc:
            rejected[type(exc).__name__] += 1
            time.sleep(args.interval)
            continue
        if not text:
            rejected['missing_or_short_article'] += 1
            time.sleep(args.interval)
            continue
        parent = current[item['human_id']]
        if item['timestamp'][:8] < parent['published_at'].replace('-', ''):
            rejected['capture_predates_publication'] += 1
            time.sleep(args.interval)
            continue
        fp_now, fp_old = grams(parent['text']), grams(text)
        fraction = len(fp_now & fp_old)/len(fp_now) if fp_now else 0
        overlaps.append(fraction)
        row = {**parent, 'id': parent['id']+':archive:'+item['timestamp'],
               'text': text, 'words': len(text.split()),
               'spans': [{'start': 0, 'end': len(text), 'label': 0}],
               'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'capture_timestamp': item['timestamp'],
               'capture_index': item['index'], 'capture_url': item['capture_url'],
               'capture_warc_filename': item['filename'],
               'capture_warc_offset': item['offset'],
               'capture_warc_length': item['length'],
               'capture_warc_sha256': hashlib.sha256(compressed).hexdigest(),
               'current_13gram_fraction_in_archive': fraction,
               'extraction_method': f'pre2023_commoncrawl_{args.source}_article_paragraphs_v1'}
        accepted.append(row)
        (ROOT/'raw_archive'/f"{parent['id']}.warc.gz").write_bytes(compressed)
        if (i+1) % 25 == 0:
            print(i+1, 'accepted', len(accepted), 'rejected', dict(rejected), flush=True)
        time.sleep(args.interval)
    out = ROOT/f'archived_{args.source}_human.jsonl'
    with out.open('w') as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'ledger': len(ledger), 'archived_text_accepted': len(accepted),
                'rejected': dict(rejected),
                'median_current_13gram_fraction_in_archive': statistics.median(overlaps) if overlaps else None,
                'current_text_90pct_archived_count': sum(x >= 0.9 for x in overlaps),
                'note': 'Rows contain text actually captured before 2023. Current-page text may differ; use these historical extracts as the verified-human subset.'}
    (ROOT/f'archived_{args.source}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
