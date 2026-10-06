#!/usr/bin/env python3
"""Remove footer credits/contacts from already-collected science prose."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def main():
    path = ROOT/'human_articles.jsonl'
    rows = [json.loads(line) for line in path.open()]
    changed = 0
    for row in rows:
        if row['source'] != 'noaa_fisheries':
            continue
        paras = row['text'].split('\n\n')
        body = []
        for p in paras:
            if re.match(r'^(story by|photos? by|for more information|to contact|media contact)\b', p, re.I):
                if p.lower().startswith('story by'):
                    row['author'] = re.sub(r'^story by\s+', '', p, flags=re.I).split('.')[0].strip()
                break
            body.append(p)
        cleaned = '\n\n'.join(body)
        if cleaned != row['text'] and len(cleaned.split()) >= 450:
            row['text'] = cleaned
            row['words'] = len(cleaned.split())
            row['spans'] = [{'start': 0, 'end': len(cleaned), 'label': 0}]
            row['text_sha256'] = hashlib.sha256(cleaned.encode()).hexdigest()
            row['extraction_method'] += '+footer_strip_v1'
            changed += 1
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print({'rows': len(rows), 'footer_stripped': changed})


if __name__ == '__main__':
    main()
