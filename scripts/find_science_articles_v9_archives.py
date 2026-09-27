#!/usr/bin/env python3
"""Find pre-2023 Common Crawl captures for the collected article URLs."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
PATTERNS = {
    'epa_science_matters': 'www.epa.gov/sciencematters/*',
    'noaa_climate': 'www.climate.gov/news-features/*',
    'noaa_fisheries': 'www.fisheries.noaa.gov/feature-story/*',
}


def normalized(url):
    u = urlsplit(url)
    return urlunsplit(('', u.netloc.lower().removeprefix('www.'), u.path.rstrip('/'), '', ''))


def query(index, source):
    response = requests.get(f'https://index.commoncrawl.org/{index}-index',
                            params={'url': PATTERNS[source], 'output': 'json', 'filter': 'status:200'}, timeout=45)
    if response.status_code == 404:
        return index, []
    response.raise_for_status()
    return index, [json.loads(line) for line in response.text.splitlines() if line]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=PATTERNS, default='epa_science_matters')
    p.add_argument('--years', nargs='+', type=int, default=[2022])
    p.add_argument('--interval', type=float, default=1.0, help='Seconds between index API requests')
    args = p.parse_args()
    response = requests.get('https://index.commoncrawl.org/collinfo.json', timeout=20)
    response.raise_for_status()
    indices = [r['id'] for r in response.json()
               if r['id'].startswith(tuple(f'CC-MAIN-{y}' for y in args.years))]
    rows = [json.loads(line) for filename in ('human_articles.jsonl', 'epa_science_matters.jsonl')
            for line in (ROOT/filename).open()]
    selected = {normalized(r['url']): r for r in rows if r['source'] == args.source}
    captures = {}
    out = ROOT/f'archive_ledger_{args.source}.jsonl'
    if out.exists():
        for line in out.open():
            item = json.loads(line)
            captures[normalized(item['source_url'])] = item
    failures = {}
    # Common Crawl asks clients to avoid parallel index requests and pause
    # between calls; keep this loop sequential.
    for index in indices:
        try:
            _, found = query(index, args.source)
        except Exception as exc:
            failures[index] = str(exc)
            time.sleep(args.interval)
            continue
        for cap in found:
            key = normalized(cap['url'])
            if key not in selected or cap['timestamp'] >= '20230101':
                continue
            item = {'human_id': selected[key]['id'], 'source_url': selected[key]['url'],
                    'capture_url': cap['url'], 'timestamp': cap['timestamp'], 'index': index,
                    'filename': cap['filename'], 'offset': int(cap['offset']),
                    'length': int(cap['length']), 'digest': cap['digest']}
            if key not in captures or cap['timestamp'] > captures[key]['timestamp']:
                captures[key] = item
        print(index, 'captures', len(found), 'matched_unique', len(captures), flush=True)
        time.sleep(args.interval)
    with out.open('w') as f:
        for item in sorted(captures.values(), key=lambda x: x['human_id']):
            f.write(json.dumps(item)+'\n')
    manifest = {'source': args.source, 'article_count': len(selected), 'archive_matched': len(captures),
                'years_queried_this_run': args.years, 'indices_queried_this_run': len(indices),
                'index_failures_this_run': failures,
                'capture_years': dict(Counter(v['timestamp'][:4] for v in captures.values())),
                'note': 'A capture entry proves an archived page existed; its text must be fetched and compared before labeling the current extract as historically human.'}
    (ROOT/f'archive_ledger_{args.source}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
