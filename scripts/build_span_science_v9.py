"""Add paired, equal-length science articles to the balanced span corpus."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home')
SCIENCE = ROOT/'data/science_articles_v9'
PARENT = ROOT/'data/span_publication_hardneg_v8'
OUTPUT = ROOT/'data/span_science_paired_v9'
TOKENIZER = ROOT/'models/Qwen3-1.7B'
SEED = 20260927
WINDOW = 512


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_rows(row, tokenizer, positions):
    ids = tokenizer.encode(row['text'], add_special_tokens=False)
    if len(ids) < WINDOW:
        return []
    out = []
    for position, start in positions:
        assert 0 <= start <= len(ids)-WINDOW
        text = tokenizer.decode(ids[start:start+WINDOW], skip_special_tokens=True)
        label = int(row['label']) if 'label' in row else int(row['kind'] == 'ai')
        out.append({'id': f'science_v9:{row["id"]}:{position}',
                    'text': text, 'spans': [{'start': 0, 'end': len(text), 'label': label}],
                    'kind': 'ai' if label else 'human',
                    'construction': 'matched_science_article_window',
                    'source': 'science_v9:'+row['source'], 'domain': 'science_publication',
                    'parent_id': row['id'], 'human_id': row.get('human_id', row['id']),
                    'generator': row.get('model'), 'window_position': position,
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return out


def highest_risk_start(scores, threshold):
    """Choose the 512-token training window containing most frozen false alarms."""
    scores = np.asarray(scores)
    if len(scores) <= WINDOW:
        return 0
    above = (scores >= threshold).astype(np.int32)
    if above.any():
        cumulative = np.pad(above.cumsum(), (1, 0))
        counts = cumulative[WINDOW:]-cumulative[:-WINDOW]
        return int(np.argmax(counts))
    cumulative = np.pad(scores.cumsum(dtype=np.float64), (1, 0))
    means = cumulative[WINDOW:]-cumulative[:-WINDOW]
    return int(np.argmax(means))


def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    humans = {r['id']: r for r in (json.loads(line) for line in (SCIENCE/'train_candidates.jsonl').open())}
    ai_files = [SCIENCE/f'accepted_{model}_train.jsonl' for model in ('qwen2_5_3b', 'smollm2_1_7b')]
    ai = [json.loads(line) for path in ai_files for line in path.open()]
    assert len(humans) == 146 and 120 <= len(ai) <= len(humans)
    assert len({r['human_id'] for r in ai}) == len(ai)
    assert {r['human_id'] for r in ai} <= set(humans)
    assert all(r['human_text_sha256'] == humans[r['human_id']]['text_sha256'] for r in ai)
    assert len({r['text_sha256'] for r in ai}) == len(ai)
    archive_file = SCIENCE/'archived_noaa_fisheries_human.jsonl'
    archives = {}
    if archive_file.exists():
        for line in archive_file.open():
            row = json.loads(line)
            base_id = row['id'].split(':archive:')[0]
            if base_id in humans:
                row['id'] = base_id
                archives[base_id] = row
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    diagnostic = ROOT/'runs/qwen3_token_repeat2_publication_v8_20k/v9_train_human_diagnostic_v8'
    baseline_report = json.loads(diagnostic.with_suffix('.json').read_text())
    candidate_file = SCIENCE/'train_candidates_archive_preferred.jsonl'
    assert baseline_report['validation_sha256'] == sha(candidate_file)
    with np.load(str(diagnostic)+'_scores.npz') as data:
        offsets = data['document_offsets']
        ids = data['document_ids']
        scores = data['score']
        risk_scores = {str(row_id): scores[offsets[i]:offsets[i+1]]
                       for i, row_id in enumerate(ids)}
    assert set(risk_scores) == set(humans)
    pairs = []
    used_archives = 0
    hard_false_tokens = 0
    hard_false_documents = 0
    for generated in ai:
        human = archives.get(generated['human_id'], humans[generated['human_id']])
        h_ids = tokenizer.encode(human['text'], add_special_tokens=False)
        a_ids = tokenizer.encode(generated['text'], add_special_tokens=False)
        risk = risk_scores[generated['human_id']]
        assert len(risk) == len(h_ids)
        h_hard = highest_risk_start(risk, baseline_report['threshold'])
        selected_false = int(np.sum(risk[h_hard:h_hard+WINDOW] >= baseline_report['threshold']))
        hard_false_tokens += selected_false
        hard_false_documents += selected_false > 0
        h_max, a_max = len(h_ids)-WINDOW, len(a_ids)-WINDOW
        a_hard = round(h_hard/h_max*a_max) if h_max > 0 else 0
        # Put the ordinary control opposite the hard window where possible.
        ordinary_end = h_hard < h_max/2
        h_ordinary = h_max if ordinary_end else 0
        a_ordinary = a_max if ordinary_end else 0
        h = window_rows(human, tokenizer, [('hard', h_hard), ('ordinary', h_ordinary)])
        used_archives += generated['human_id'] in archives
        a = window_rows(generated, tokenizer, [('hard', a_hard), ('ordinary', a_ordinary)])
        if len(h) != 2 or len(a) != 2 or generated['words'] < 350:
            raise ValueError(f'Short pair {generated["human_id"]}: {len(h)} {len(a)} {generated["words"]}')
        pairs.extend(h+a)
    assert len(pairs) == 4*len(ai)
    original = [json.loads(line) for line in (PARENT/'train.jsonl').open()]
    rng = random.Random(SEED)
    # Preserve mixed-document supervision; replace pure science abstracts with
    # equal numbers of human and AI feature windows instead.
    old_human = [r for r in original if r['source'] == 'mage:sci' and r['kind'] == 'human']
    old_ai = [r for r in original if r['source'] == 'mage:sci' and r['kind'] == 'ai']
    rng.shuffle(old_human)
    rng.shuffle(old_ai)
    per_class = len(pairs)//2
    assert len(old_human) >= per_class and len(old_ai) >= per_class
    removed = {r['id'] for r in old_human[:per_class]+old_ai[:per_class]}
    rows = [r for r in original if r['id'] not in removed]+pairs
    rng.shuffle(rows)
    assert len(rows) == 20000
    OUTPUT.mkdir(parents=True)
    with (OUTPUT/'train.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    (OUTPUT/'val.jsonl').write_bytes((PARENT/'val.jsonl').read_bytes())
    sources = Counter(r['source'] for r in rows)
    kinds = Counter(r['kind'] for r in rows)
    manifest = {'role': 'paired science publication pilot, source-exclusive from NOAA calibration and archived EPA test',
                'seed': SEED, 'documents': len(rows), 'paired_article_topics': len(ai),
                'added_windows': len(pairs), 'replaced_mage_sci_pure_documents': len(removed),
                'window_source_tokens': WINDOW, 'added_labels': dict(Counter(r['kind'] for r in pairs)),
                'added_generators': dict(Counter(r['generator'] for r in pairs if r['kind']=='ai')),
                'sources': dict(sources), 'kinds': dict(kinds),
                'archive_verified_human_topics': used_archives,
                'hard_false_positive_tokens_captured': hard_false_tokens,
                'hard_false_positive_article_count': hard_false_documents,
                'hard_window_selection': 'v8 frozen token scores on training-side humans only; maximize count above 5.03125',
                'hard_window_diagnostic_sha256': sha(diagnostic.with_suffix('.json')),
                'human_provenance_caveat': 'NASA and nonarchived NOAA pages carry pre-2023 dates; current extracted text is not independently archive-verified.',
                'parent_train_sha256': sha(PARENT/'train.jsonl'),
                'science_human_sha256': sha(SCIENCE/'train_candidates.jsonl'),
                'science_archive_sha256': sha(archive_file) if archive_file.exists() else None,
                'science_ai_sha256': {path.name: sha(path) for path in ai_files},
                'train_sha256': sha(OUTPUT/'train.jsonl'), 'val_sha256': sha(OUTPUT/'val.jsonl')}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('documents','paired_article_topics','added_windows',
                                             'added_labels','added_generators')}, indent=2))


if __name__ == '__main__':
    main()
