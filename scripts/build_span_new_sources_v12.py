"""Replace selected v10 rows with screened, group-disjoint new sources."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home')
INPUT = ROOT/'data/span_essay_paired_v10'
CANDIDATES = ROOT/'data/candidate_span_sources/round2'
OUTPUT = ROOT/'data/span_new_sources_v12'
SEED = 20260927


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def shuffled(rows: list[dict], salt: str) -> list[dict]:
    result = list(rows)
    random.Random(f'{SEED}:{salt}').shuffle(result)
    return result


def select(rows: list[dict], count: int, salt: str) -> tuple[list[dict], list[dict]]:
    result = shuffled(rows, salt)
    if len(result) < count:
        raise ValueError(f'Only {len(result)} records for {salt}; need {count}')
    return result[:count], result[count:]


def completion_groups(rows: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row['group_id'], []).append(row)
    result = []
    for group in groups.values():
        if len(group) != 2 or {row['kind'] for row in group} != {'human', 'mixed'}:
            raise ValueError('GRADTEX group must contain one human and one mixed row')
        result.append(group)
    return result


def source_row(row: dict, construction: str, domain: str | None = None) -> dict:
    result = dict(row)
    result['construction'] = construction
    result['domain'] = domain or result.get('domain') or result.get('category') or result.get('sub_source') or 'unspecified'
    return result


def bounded_human(row: dict, tokenizer, construction: str, salt: str) -> dict:
    """Give long works one reproducible training window without source dominance."""
    result = source_row(row, construction)
    ids = tokenizer.encode(row['text'], add_special_tokens=False)
    if len(ids) <= 512:
        return result
    start = random.Random(f'{SEED}:{salt}:{row["id"]}').randrange(len(ids)-512+1)
    text = tokenizer.decode(ids[start:start+512], skip_special_tokens=True)
    result['text'] = text
    result['spans'] = [{'start': 0, 'end': len(text), 'label': 0}]
    result['excerpt_source_token_start'] = start
    result['excerpt_source_token_limit'] = 512
    result['original_text_sha256'] = sha256(row['text'].encode()).hexdigest()
    return result


def check_spans(rows: list[dict]) -> None:
    for row in rows:
        spans = row['spans']
        if not row['text'] or not spans or spans[0]['start'] != 0 or spans[-1]['end'] != len(row['text']):
            raise ValueError(f'Incomplete spans: {row["id"]}')
        if any(a['end'] > b['start'] or a['end'] <= a['start'] for a, b in zip(spans, spans[1:])):
            raise ValueError(f'Invalid spans: {row["id"]}')
        labels = {span['label'] for span in spans}
        expected = {'human': {0}, 'ai': {1}, 'mixed': {0, 1}}[row['kind']]
        if labels != expected:
            raise ValueError(f'Kind mismatch: {row["id"]}')


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    parent = load(INPUT/'train.jsonl')
    assert len(parent) == 20000 and len({row['id'] for row in parent}) == len(parent)
    assert digest(INPUT/'train.jsonl') == '11ce9525b9e342e66c14c03f7d0f805af122742d3aef4b60ae1885c84d88f81a'

    dolly = load(CANDIDATES/'dolly/screened_candidate.jsonl')
    dolly_train = []
    dolly_holdout = []
    for category in ('creative_writing', 'brainstorming'):
        pool = [r for r in dolly if r['category'] == category]
        chosen, rest = select(pool, 300, f'dolly:{category}')
        dolly_train += chosen
        dolly_holdout += rest[:75]
    stories = load(CANDIDATES/'travis_shortstory/screened_candidate.jsonl')
    story_train, story_rest = select(stories, 200, 'stories')
    story_holdout = story_rest[:100]
    authors_train = []
    authors_holdout = []
    for source, n in (('cory_doctorow', 24), ('aaron_swartz', 36)):
        authors = load(CANDIDATES/f'{source}/screened_candidate.jsonl')
        chosen, rest = select(authors, n, source)
        authors_train += chosen
        authors_holdout += rest
    gradtex = load(CANDIDATES/'gradtex/screened_candidate.jsonl')
    gradtex_train_groups, gradtex_rest = select(completion_groups(gradtex), 1000, 'gradtex')
    gradtex_holdout_groups = gradtex_rest[:250]
    gradtex_train = [source_row(next(row for row in group if row['kind'] == 'mixed'),
                                'preserved_context_completion') for group in gradtex_train_groups]
    gradtex_holdout = [source_row(row, 'preserved_context_completion')
                       for group in gradtex_holdout_groups for row in group]

    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')

    human_add = ([source_row(r, 'employee_open_ended_response', 'creative_writing') for r in dolly_train[:300]]
                 + [source_row(r, 'employee_open_ended_response', 'brainstorming') for r in dolly_train[300:]]
                 + [bounded_human(r, tokenizer, 'historical_fiction', 'story') for r in story_train]
                 + [bounded_human(r, tokenizer, 'pre_llm_named_essay', 'author') for r in authors_train])
    assert len(human_add) == 860
    remove_human = {'mage:sci': 250, 'editlens:fineweb_edu': 150,
                    'editlens:reddit_writing_prompts': 160, 'mage:roct': 100,
                    'mage:eli5': 100, 'mage:wp': 100}
    removed = set()
    for source, count in remove_human.items():
        chosen, _ = select([r for r in parent if r['source'] == source and r['kind'] == 'human'],
                           count, f'remove:{source}')
        removed.update(row['id'] for row in chosen)
    mixed, _ = select([r for r in parent if r['source'] == 'DAMASHA clean published aggregate'
                       and r['kind'] == 'mixed'], 1000, 'remove:damasha')
    removed.update(row['id'] for row in mixed)
    assert len(removed) == 1860
    train = shuffled([r for r in parent if r['id'] not in removed] + human_add + gradtex_train, 'train')
    assert len(train) == 20000 and len({row['id'] for row in train}) == 20000
    check_spans(train)
    holdout = ([source_row(r, 'employee_open_ended_response') for r in dolly_holdout]
               + [source_row(r, 'historical_fiction') for r in story_holdout]
               + [source_row(r, 'pre_llm_named_essay') for r in authors_holdout]
               + gradtex_holdout)
    check_spans(holdout)
    assert not {row['id'] for row in train} & {row['id'] for row in holdout}
    assert not {row['group_id'] for row in train if 'group_id' in row} & {
        row['group_id'] for row in holdout if 'group_id' in row}
    assert not {sha256(row['text'].encode()).digest() for row in train} & {
        sha256(row['text'].encode()).digest() for row in holdout}

    OUTPUT.mkdir(parents=True)
    write(OUTPUT/'train.jsonl', train)
    (OUTPUT/'val.jsonl').write_bytes((INPUT/'val.jsonl').read_bytes())
    write(OUTPUT/'new_source_holdout.jsonl', holdout)
    manifest = {'role': 'v10 fixed-size new-source pilot; holdout is development evaluation',
                'seed': SEED, 'documents': len(train),
                'parent_train_sha256': digest(INPUT/'train.jsonl'),
                'val_sha256': digest(OUTPUT/'val.jsonl'),
                'train_sha256': digest(OUTPUT/'train.jsonl'),
                'holdout_sha256': digest(OUTPUT/'new_source_holdout.jsonl'),
                'removed_parent_documents': len(removed),
                'added_documents': len(human_add)+len(gradtex_train),
                'new_source_holdout_documents': len(holdout),
                'sources': dict(Counter(r['source'] for r in train)),
                'kinds': dict(Counter(r['kind'] for r in train)),
                'new_source_holdout_sources': dict(Counter(r['source'] for r in holdout)),
                'screened_source_sha256': {
                    source: digest(CANDIDATES/source/'screened_candidate.jsonl')
                    for source in ('dolly', 'travis_shortstory', 'cory_doctorow',
                                   'aaron_swartz', 'gradtex')}}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('documents','added_documents',
        'new_source_holdout_documents','kinds','new_source_holdout_sources')}, indent=2))


if __name__ == '__main__':
    main()
