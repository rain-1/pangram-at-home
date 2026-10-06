#!/usr/bin/env python3
"""Make source-exclusive publication pools and topic-matched AI prompt manifests."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
ASSIGNMENT = {
    'nasa_earth_observatory': 'train_candidates',
    'noaa_fisheries': 'train_candidates',
    'noaa_climate': 'calibration_human',
    'epa_science_matters': 'locked_test_human',
}


def read(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def write(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    rows = list(read(ROOT/'human_articles.jsonl')) + list(read(ROOT/'epa_science_matters.jsonl'))
    assert rows and len({r['id'] for r in rows}) == len(rows)
    assert len({r['text_sha256'] for r in rows}) == len(rows)
    outputs = {name: [] for name in ASSIGNMENT.values()}
    prompts = []
    for row in rows:
        assert row['published_at'] < '2023-01-01'
        assert row['spans'] == [{'start': 0, 'end': len(row['text']), 'label': 0}]
        split = ASSIGNMENT[row['source']]
        outputs[split].append(row)
        if split != 'train_candidates':
            topic = row['title'].split(' | ')[0].removesuffix(' - NASA Science').strip()
            prompts.append({
                'human_id': row['id'], 'split': split.replace('_human', '_ai_candidate'),
                'source': row['source'], 'topic_title': topic,
                'human_text_sha256': row['text_sha256'],
                'prompt_version': 'science_feature_topic_v1',
                'prompt': (
                    'Write an original science feature of about 500 words for a general audience '
                    'on the topic below. Explain the scientific context, evidence, and limitations '
                    'in coherent paragraphs. Write the article itself, with no title, byline, '
                    'bullet list, or introductory note. Do not quote or paraphrase a particular '
                    f'published article.\n\nTopic: {topic}'
                ),
            })
    for name, values in outputs.items():
        values.sort(key=lambda x: x['id'])
        write(ROOT/f'{name}.jsonl', values)
    prompts.sort(key=lambda x: x['human_id'])
    write(ROOT/'matched_ai_prompts.jsonl', prompts)
    summary = {
        'split_policy': ASSIGNMENT,
        'counts': {name: len(v) for name, v in outputs.items()},
        'source_counts': dict(Counter(r['source'] for r in rows)),
        'all_human': True,
        'matched_ai_prompt_count': len(prompts),
        'matched_ai_generated_count': 0,
        'sha256': {name: hashlib.sha256((ROOT/f'{name}.jsonl').read_bytes()).hexdigest() for name in outputs},
        'limitations': [
            'Publication dates are current page metadata, not archived snapshots from before 2023.',
            'NOAA Climate.gov and NOAA Fisheries share an agency, although their editorial sections differ.',
            'These are source-exclusive publication pools, not a fully balanced human/AI evaluation until AI texts are generated and reviewed.',
        ],
    }
    (ROOT/'split_manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
