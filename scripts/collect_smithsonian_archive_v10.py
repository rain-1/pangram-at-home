"""Collect pre-2023 archived magazine prose as candidate publication hard negatives.

Only Common Crawl historical WARC text is retained. The source articles are
copyrighted, so raw text stays on F: and is never added to the repository.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path('/mnt/f/pangram-at-home/data/smithsonian_archive_v10')
DATA = Path('/mnt/f/pangram-at-home/data')
INDEXES = ('CC-MAIN-2022-05', 'CC-MAIN-2022-21', 'CC-MAIN-2021-49',
           'CC-MAIN-2022-33', 'CC-MAIN-2022-40')
BASE = 'https://www.smithsonianmag.com'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30, **kwargs)
            if response.status_code in (429, 500, 502, 503):
                time.sleep(1+attempt)
                continue
            return response
        except requests.RequestException:
            time.sleep(1+attempt)
    return None


def archive_capture(session, url):
    for index in INDEXES:
        r = get(session, f'https://index.commoncrawl.org/{index}-index',
                params={'url': url, 'output': 'json', 'filter': 'status:200'})
        if r is None or r.status_code != 200:
            continue
        for line in r.text.splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item['timestamp'] < '20230101':
                item['index'] = index
                return item
    return None


def warc_html(session, item):
    start, length = int(item['offset']), int(item['length'])
    r = get(session, 'https://data.commoncrawl.org/'+item['filename'],
            headers={'Range': f'bytes={start}-{start+length-1}'})
    if r is None or r.status_code != 206 or len(r.content) != length:
        return None, None
    record = gzip.decompress(r.content)
    _, payload = record.split(b'\r\n\r\n', 1)
    _, html = payload.split(b'\r\n\r\n', 1)
    if html.startswith(b'\x1f\x8b'):
        html = gzip.decompress(html)
    return r.content, html


def extract(html, item):
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.select_one('article .articleLeft')
    if article is None:
        return None
    date = soup.select_one('article time')
    raw_date = date.get_text(' ', strip=True) if date else ''
    from email.utils import parsedate_to_datetime
    try:
        import datetime as dt
        published = dt.datetime.strptime(raw_date, '%B %d, %Y').date().isoformat()
    except ValueError:
        try:
            published = parsedate_to_datetime(raw_date).date().isoformat()
        except Exception:
            return None
    if published >= '2023-01-01' or item['timestamp'][:8] < published.replace('-', ''):
        return None
    h1 = soup.select_one('article h1')
    title = h1.get_text(' ', strip=True) if h1 else ''
    authors = [m.get('content', '').strip() for m in soup.select('meta[name="author"]')]
    author = next((a for a in authors if a and a != 'Smithsonian Magazine'), None)
    if not title or not author:
        return None
    paras = []
    for p in article.find_all('p', recursive=False):
        value = re.sub(r'\s+', ' ', p.get_text(' ', strip=True)).strip()
        if re.match(r'^(read more|recommended|related|subscribe|advertisement)\b', value, re.I):
            break
        if len(value.split()) < 8:
            continue
        paras.append(value)
    body = '\n\n'.join(paras)
    if not 450 <= len(body.split()) <= 5000:
        return None
    return {'text': body, 'title': title, 'author': author,
            'published_at': published, 'published_raw': raw_date,
            'words': len(body.split())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=int, default=100)
    p.add_argument('--pages', type=int, nargs='+', default=list(range(345, 366)))
    p.add_argument('--interval', type=float, default=0.35)
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'raw').mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'pangram-at-home-research/1.0 (historical dataset audit)'})
    protected = {r['publication_url'].rstrip('/') for r in
                 (json.loads(line) for line in (DATA/'span_ai_eval_candidate_v1/test.jsonl').open())
                 if r['kind'] == 'human' and r.get('publication_url')}
    candidates = []
    for page in args.pages:
        r = get(session, BASE+'/category/smart-news/', params={'page': page})
        if r is None or r.status_code != 200:
            continue
        soup = BeautifulSoup(r.content, 'html.parser')
        for a in soup.select('a[href*="/smart-news/"]'):
            url = urljoin(BASE, a.get('href', '')).split('?', 1)[0]
            if re.match(r'https://www\.smithsonianmag\.com/smart-news/.+-\d+/?$', url):
                candidates.append(url)
        time.sleep(args.interval)
    candidates = sorted(u for u in set(candidates) if u.rstrip('/') not in protected)
    random.Random(20260927).shuffle(candidates)
    out = ROOT/'articles.jsonl'
    done = set()
    if out.exists():
        done = {json.loads(line)['url'] for line in out.open()}
    counts = Counter()
    for url in candidates:
        if len(done) >= args.target:
            break
        if url in done:
            continue
        item = archive_capture(session, url)
        if item is None:
            counts['no_pre2023_capture'] += 1
            time.sleep(args.interval)
            continue
        compressed, html = warc_html(session, item)
        if html is None:
            counts['warc_fetch_error'] += 1
            time.sleep(args.interval)
            continue
        extracted = extract(html, item)
        if extracted is None:
            counts['extraction_or_provenance_reject'] += 1
            time.sleep(args.interval)
            continue
        text = extracted.pop('text')
        id_ = sha(url.encode())[:20]
        row = {'id': id_, 'url': url, **extracted, 'text': text,
               'text_sha256': sha(text.encode()), 'label': 0, 'kind': 'human',
               'spans': [{'start': 0, 'end': len(text), 'label': 0}],
               'source': 'smithsonian_archive', 'domain': 'published_nonfiction',
               'construction': 'pre2023_archived_editorial_feature',
               'source_groups': [url], 'source_ids': [url],
               'capture_timestamp': item['timestamp'], 'capture_index': item['index'],
               'capture_warc_filename': item['filename'],
               'capture_warc_offset': item['offset'], 'capture_warc_length': item['length'],
               'capture_warc_sha256': sha(compressed)}
        with out.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        (ROOT/'raw'/f'{id_}.warc.gz').write_bytes(compressed)
        done.add(url)
        if len(done) % 10 == 0:
            print('accepted', len(done), 'attempted', sum(counts.values())+len(done), flush=True)
        time.sleep(args.interval)
    manifest = {'accepted': len(done), 'candidate_urls': len(candidates),
                'rejections_this_run': dict(counts), 'protected_external_urls_excluded': len(protected),
                'capture_cutoff': '2023-01-01', 'source': BASE,
                'raw_text_policy': 'local research only; do not redistribute copyrighted article text',
                'articles_sha256': sha(out.read_bytes()) if out.exists() else None}
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
