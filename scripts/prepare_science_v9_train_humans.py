"""Prefer historically captured NOAA text for training-side diagnostics."""
import json
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')


def main():
    rows = {r['id']: r for r in (json.loads(line) for line in (ROOT/'train_candidates.jsonl').open())}
    for line in (ROOT/'archived_noaa_fisheries_human.jsonl').open():
        archived = json.loads(line)
        key = archived['id'].split(':archive:')[0]
        if key in rows:
            archived['id'] = key
            rows[key] = archived
    assert len(rows) == 146
    output = ROOT/'train_candidates_archive_preferred.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows.values()))
    print(output, 'rows', len(rows), 'archive_verified', sum('capture_timestamp' in r for r in rows.values()))


if __name__ == '__main__':
    main()
