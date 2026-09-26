"""Audit the actual document, window, and supervised-token mix of Qwen v5.

This uses the same tokenizer, offset alignment, 512-token windows, and 256-token
stride as train_token_lora.py. Repeat2 first-copy tokens are ignored in the
supervised-token counts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from span_data import encode_document, window_starts

ROOT = Path('/mnt/f/pangram-at-home')
TRAIN = ROOT / 'data/span_size_curve_v5/size_20000/train.jsonl'
RUN = ROOT / 'runs/qwen3_token_repeat2_v5_20k_e1_local'
OUT = Path(__file__).resolve().parents[1] / 'reports'
INIT_TRAIN = ROOT / 'data/diverse_pyramid_v1/train_full.parquet'


def new_counter():
    return Counter()


def family(row):
    return 'LLMTrace' if row['source'] == 'LLMTrace_detection' else 'earlier v4 composites'


def audit():
    config = json.loads((RUN / 'run_config.json').read_text())
    assert config['max_source_tokens'] == 512 and config['stride'] == 256
    assert config['repeat2'] and config['first_copy_loss_masked']
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    init = pq.read_table(INIT_TRAIN, columns=['group_id','text_sha256']).to_pydict()
    init_groups = {group for group in init['group_id'] if group}
    init_hashes = {value for value in init['text_sha256'] if value}
    overlap = Counter()
    v4_groups = set()
    groups = {level: defaultdict(new_counter) for level in
              ('family', 'source', 'domain', 'kind', 'construction',
               'family_domain', 'family_kind', 'source_kind', 'source_domain')}
    lengths = defaultdict(list)
    total = Counter()
    for line in TRAIN.open():
        row = json.loads(line)
        ids, _, labels = encode_document(row, tokenizer)
        starts = window_starts(len(ids), 512, 256)
        source_labels = Counter(label for label in labels if label in (0, 1))
        window_labels = Counter(label for start in starts
                                for label in labels[start:start + 512] if label in (0, 1))
        values = Counter({'documents': 1, 'windows': len(starts),
                          'source_tokens': len(ids),
                          'source_human_tokens': source_labels[0],
                          'source_ai_tokens': source_labels[1],
                          'supervised_human_tokens': window_labels[0],
                          'supervised_ai_tokens': window_labels[1],
                          'processed_tokens_repeat2': 2*sum(min(512, len(ids)-start) for start in starts)})
        total.update(values)
        if family(row) == 'earlier v4 composites':
            source_groups = set(row.get('source_groups', []))
            v4_groups.update(source_groups)
            overlap['v4_documents'] += 1
            overlap['v4_documents_with_any_initialization_group'] += bool(source_groups & init_groups)
            overlap['v4_documents_with_all_groups_in_initialization'] += bool(source_groups) and source_groups <= init_groups
            overlap['v4_exact_text_hashes_in_initialization'] += row.get('text_sha256') in init_hashes
        keys = {
            'family': family(row), 'source': row['source'], 'domain': row['domain'],
            'kind': row['kind'], 'construction': row['construction'],
            'family_domain': f"{family(row)} / {row['domain']}",
            'family_kind': f"{family(row)} / {row['kind']}",
            'source_kind': f"{row['source']} / {row['kind']}",
            'source_domain': f"{row['source']} / {row['domain']}",
        }
        for level, key in keys.items():
            groups[level][key].update(values)
        lengths[family(row)].append(len(ids))
    assert total['documents'] == 20000
    assert total['windows'] == config['train_windows']
    report = {'run': RUN.name,
              'rule': 'uniform shuffled windows, 512 source tokens, 256 stride; Repeat2 first copy masked',
              'training_steps': config['max_steps'],
              'effective_batch_size': config['effective_batch_size'],
              'planned_window_draws': config['max_steps']*config['effective_batch_size'],
              'initialization_overlap': {**dict(overlap),
                                         'v4_unique_source_groups': len(v4_groups),
                                         'v4_source_groups_in_initialization': len(v4_groups & init_groups)},
              'total': dict(total),
              'groups': {level:{key:dict(value) for key,value in sorted(counter.items())}
                         for level,counter in groups.items()},
              'lengths': {key:{'median_source_tokens':statistics.median(values),
                                'p90_source_tokens':sorted(values)[int(.9*(len(values)-1))]}
                          for key,values in lengths.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / 'dataset_mixture_audit_v5.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    result = audit()
    print(json.dumps({'total':result['total'],'family':result['groups']['family'],
                      'lengths':result['lengths']},indent=2))
