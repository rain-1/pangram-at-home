"""Normalize bounded, local Hugging Face human-source candidates.

Raw inputs and generated JSONL live outside Git under /mnt/f/... . This script
never uploads or commits text. It records one known reviewer group per MARC
row and preserves the original story benchmark split.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path('/mnt/f/pangram-at-home/data/candidate_span_sources/human')
REVISION = 'a7b1fa9703f1f930d5c8e1a65c5d2774aaab0c85'
STORY_REVISION = 'be75c1948d97f4d0d4ec8fa5f0dfc59398654d22'
SEED = 20260927
DATA = Path('/mnt/f/pangram-at-home/data')


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def phrase_fingerprints(text: str) -> set[bytes]:
    words = re.findall(r'\w+', text.casefold())
    result = set()
    for i in range(max(0, len(words) - 23)):
        fingerprint = hashlib.blake2b(' '.join(words[i:i + 24]).encode(), digest_size=8).digest()
        if fingerprint[0] < 16:
            result.add(fingerprint)
    return result


def audit_overlap() -> dict:
    refs = [
        'span_balanced_v6/train.jsonl',
        'span_publication_hardneg_v8/train.jsonl',
        'span_essay_paired_v10/train.jsonl',
        'span_human_eval_v2/calibration.jsonl',
        'span_human_eval_v2/test.jsonl',
        'span_realistic_eval_v1/test.jsonl',
        'span_ai_eval_candidate_v1/test.jsonl',
    ]
    protected: set[bytes] = set()
    seen = {}
    for relative in refs:
        path = DATA / relative
        if not path.exists():
            continue
        rows = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                rows += 1
                protected.update(phrase_fingerprints(json.loads(line)['text']))
        seen[relative] = rows
    outcomes = {}
    base = ROOT / 'normalized'
    for path in sorted(base.rglob('*.jsonl')):
        row_count = overlaps = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                row_count += 1
                overlaps += bool(phrase_fingerprints(json.loads(line)['text']) & protected)
        outcomes[str(path.relative_to(base))] = {'rows': row_count, 'rows_with_sampled_24word_overlap': overlaps}
    result = {'reference_rows': seen, 'fingerprint_count': len(protected), 'candidates': outcomes,
              'method': 'normalized 24-word shingle fingerprints sampled at hash bucket <16, against local v6/v8/v10 training and frozen evaluation files'}
    (base / 'overlap_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def build_reviews() -> dict:
    raw = ROOT / 'amazon_reviews_multi_en_raw'
    out = ROOT / 'normalized' / 'amazon_reviews_multi_en'
    # Use one review per reviewer and ensure reviewers do not cross output
    # partitions. Retain five-rating balance within each partition.
    rng = random.Random(SEED)
    chosen: dict[str, list[dict]] = {'train': [], 'calibration': [], 'test': []}
    used_reviewers: set[str] = set()
    requests = [('train', 'train', 1000), ('validation', 'calibration', 250), ('test', 'test', 250)]
    for input_split, output_split, target in requests:
        path = raw / 'en' / f'{input_split}.jsonl.gz'
        by_star: dict[int, list[dict]] = {star: [] for star in range(1, 6)}
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                body = (row.get('review_body') or '').strip()
                title = (row.get('review_title') or '').strip()
                reviewer = str(row.get('reviewer_id') or '')
                if not body or not reviewer or len(body) < 80:
                    continue
                text = body if not title or body.casefold().startswith(title.casefold()) else title + '\n\n' + body
                row['_normalized_text'] = text
                by_star[int(row['stars'])].append(row)
        per_star, remainder = divmod(target, 5)
        allocations = {star: per_star + (1 if i < remainder else 0)
                       for i, star in enumerate(range(1, 6))}
        rows = []
        for star in range(1, 6):
            options = [r for r in by_star[star] if r['reviewer_id'] not in used_reviewers]
            rng.shuffle(options)
            got = 0
            for row in options:
                if got >= allocations[star]:
                    break
                reviewer = row['reviewer_id']
                if reviewer in used_reviewers:
                    continue
                used_reviewers.add(reviewer)
                text = row.pop('_normalized_text')
                rowid = str(row.get('review_id') or row.get('reviewer_id') + ':' + row.get('product_id', ''))
                rows.append({
                    'id': f'marc-en-{rowid}', 'text': text, 'kind': 'human_candidate',
                    'domain': 'consumer_review',
                    'spans': [{'start': 0, 'end': len(text), 'label': 0}],
                    'source': 'goosmanlei/amazon_reviews_multi',
                    'source_id': str(row.get('review_id', '')),
                    'group_id': sha('reviewer:' + reviewer),
                    'source_split': input_split,
                    'original_publication_date': None,
                    'source_collection_period': '2015-2019',
                    'human_origin_confidence': 'weak: platform user submission, not independently verified',
                    'source_version': REVISION,
                    'license': 'other: original Amazon Reviews Multi academic-research-only terms',
                    'canonical_url': f"https://www.amazon.com/gp/product/{row.get('product_id','')}",
                    'attribution': 'Amazon reviewer pseudonymous ID retained only as a salted-by-hash group key',
                    'raw_sha256': sha((row.get('review_title') or '') + '\0' + (row.get('review_body') or '')),
                    'clean_sha256': sha(text), 'genre': 'consumer product review',
                    'extraction_method': 'review_body with nonduplicative review_title prefix',
                    'star_rating': int(row['stars']),
                    'product_category': row.get('product_category'),
                })
                got += 1
        if len(rows) != target:
            raise RuntimeError(f'{output_split}: wanted {target}, got {len(rows)}')
        rng.shuffle(rows)
        chosen[output_split] = rows
        atomic_jsonl(out / f'{output_split}.jsonl', rows)
    manifest = {
        'dataset': 'goosmanlei/amazon_reviews_multi', 'revision': REVISION,
        'source_files': {s: sha_bytes((ROOT / 'amazon_reviews_multi_en_raw' / 'en' / f'{s}.jsonl.gz').read_bytes()) for s in ['train','validation','test']},
        'seed': SEED, 'selection': 'one unique reviewer per record; no reviewer crosses splits; 5-star-stratified; review body >=80 chars',
        'counts': {k: len(v) for k,v in chosen.items()},
        'license_note': 'The upstream card delegates to the original Amazon Reviews Multi terms. Academic research only; no redistribution or commercial use. HF Apache/CC tags on derived mirrors do not replace these terms.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'counts': manifest['counts'], 'source_rows_selected': sum(manifest['counts'].values()),
            'unique_reviewers': len(used_reviewers), 'min_chars': min(len(r['text']) for v in chosen.values() for r in v),
            'median_chars': sorted(len(r['text']) for v in chosen.values() for r in v)[len(used_reviewers)//2],
            'max_chars': max(len(r['text']) for v in chosen.values() for r in v)}


def build_stories() -> dict:
    raw = ROOT / 'tell_me_a_story'
    out = ROOT / 'normalized' / 'tell_me_a_story'
    stats = {}
    for split in ['train', 'validation', 'test']:
        rows = []
        with (raw / 'data' / f'{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                source = json.loads(line)
                text = (source.get('targets') or '').strip()
                if not text:
                    continue
                eid = str(source['example_id'])
                story_id = f'{split}:{eid}'  # Upstream example IDs restart in each split.
                rows.append({
                    'id': f'tell-me-a-story-{story_id}', 'text': text,
                    'kind': 'human', 'domain': 'creative_fiction',
                    'human_origin_confidence': 'dataset-asserted human-written target; collection date unknown',
                    'spans': [{'start': 0, 'end': len(text), 'label': 0}],
                    'source': 'virtualkevin/tell-me-a-story', 'source_id': story_id,
                    'group_id': story_id, 'source_split': split,
                    'original_publication_date': None,
                    'source_collection_period': None,
                    'source_version': STORY_REVISION, 'license': 'CC BY 4.0',
                    'canonical_url': 'https://github.com/google-deepmind/tell_me_a_story',
                    'attribution': 'DeepMind Technologies Limited; Huot et al. (2025), Agents\' Room',
                    'raw_sha256': sha(text), 'clean_sha256': sha(text),
                    'genre': 'prompted narrative fiction',
                    'extraction_method': 'targets field; prompt excluded',
                })
        atomic_jsonl(out / f'{split}.jsonl', rows)
        lens = sorted(len(r['text']) for r in rows)
        stats[split] = {'rows': len(rows), 'min_chars': lens[0] if lens else 0,
                        'median_chars': lens[len(lens)//2] if lens else 0,
                        'max_chars': lens[-1] if lens else 0}
    return stats


def build_scitechnews() -> dict:
    raw = ROOT / 'scitechnews' / 'train.json'
    out = ROOT / 'normalized' / 'scitechnews'
    eligible = []
    with raw.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = (row.get('pr-article') or '').strip()
            if len(text.split()) >= 300:
                eligible.append((row, text))
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    sample = eligible[:600]
    parts = {'train': sample[:400], 'calibration': sample[400:500], 'test': sample[500:600]}
    stats = {}
    for split, selected in parts.items():
        rows = []
        for source, text in selected:
            sid = str(source['id'])
            rows.append({
                'id': f'scitechnews-{sid}', 'text': text,
                'kind': 'human_candidate', 'domain': 'science_technology_press_release',
                'human_origin_confidence': 'dataset-asserted human-made content; no row-level authorship/date evidence',
                'spans': [{'start': 0, 'end': len(text), 'label': 0}],
                'source': 'ronaldahmed/scitechnews', 'source_id': sid,
                'group_id': sid, 'source_split': 'train',
                'original_publication_date': None,
                'source_collection_period': '1999-2021 (dataset-card range; no per-row date)',
                'source_version': '955dfa3b4c7c5edf29ad019a7b91734a739ef3df',
                'license': 'unclear at item level; README says all human-made; loader mentions CC BY-SA 3.0 but disables the license field',
                'canonical_url': 'https://technews.acm.org/ (original article URL absent in release)',
                'attribution': 'ACM TechNews dataset curators; article-level bylines absent',
                'raw_sha256': sha(text), 'clean_sha256': sha(text),
                'genre': 'science and technology press release/news article',
                'extraction_method': 'pr-article field; title and summary excluded',
                'source_title': source.get('pr-title'),
            })
        atomic_jsonl(out / f'{split}.jsonl', rows)
        lengths = sorted(len(r['text'].split()) for r in rows)
        stats[split] = {'rows': len(rows), 'min_words': lengths[0],
                        'median_words': lengths[len(lengths)//2],
                        'max_words': lengths[-1]}
    (out / 'manifest.json').write_text(json.dumps({
        'dataset': 'ronaldahmed/scitechnews',
        'revision': '955dfa3b4c7c5edf29ad019a7b91734a739ef3df',
        'raw_train_sha256': sha_bytes(raw.read_bytes()),
        'seed': SEED, 'eligible_rule': 'nonempty pr-article, >=300 whitespace words',
        'sample_rule': 'deterministically shuffled eligible training partition; 400 train, 100 calibration, 100 test',
        'overlap_audit': 'run as final step of this script; see ../overlap_audit.json',
        'limits': ['dataset release states all text human-produced and source collection window is 1999-2021, but row-level date/byline/canonical URL are absent', 'item-level reuse rights are unresolved; do not train until reviewed'],
        'counts': stats,
    }, indent=2) + '\n')
    return stats


if __name__ == '__main__':
    results = {'reviews': build_reviews(), 'stories': build_stories(), 'scitechnews': build_scitechnews()}
    results['overlap_audit'] = audit_overlap()
    print(json.dumps(results, indent=2))
