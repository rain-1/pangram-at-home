"""Score the remaining v10 human test sources with frozen Open Pangram models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_open_pangram_pure_v5 import MODELS, ROOT, load_model, read_rows, score_dataset

SOURCES = {
    'magazine': 'smithsonian_archive_v10/locked_test_human.jsonl',
    'asap': 'asap2_student_essays_v10/locked_test_human.jsonl',
    'epa': 'science_articles_v9/archived_epa_science_matters_human.jsonl',
    'pmc': 'pmc_publication_v6/test.jsonl',
    'cnn': 'cnn_dailymail_v1/locked_test.jsonl',
}
QWEN_STEMS = {
    'magazine':'v10_magazine_locked_test',
    'asap':'v10_asap2_locked_test',
    'epa':'v10_archived_epa_human',
    'pmc':'v10_pmc_article_test',
    'cnn':'v10_cnn_article_test',
}
QWEN_RUN = ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k'


def verify(rows: list[dict], key: str) -> None:
    assert rows and all(row['kind'] == 'human' for row in rows), key
    with np.load(QWEN_RUN/(QWEN_STEMS[key]+'_scores.npz')) as data:
        assert [row['id'] for row in rows] == list(data['document_ids']), key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--sets', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    spec = MODELS[args.model]
    output = ROOT/'runs'/f'open_pangram_editlens_{args.model}_v5'
    todo = []
    for key in args.sets:
        rows = read_rows(ROOT/'data'/SOURCES[key], {'human'}, None)
        verify(rows,key)
        path = output/f'additional_human_{key}.jsonl'
        if path.exists():
            cached = [json.loads(line) for line in path.open()]
            if [row['id'] for row in cached] == [row['id'] for row in rows]:
                print(key,'cached',len(rows),flush=True)
                continue
        todo.append((key,rows,path))
    if not todo:
        return
    tokenizer,model = load_model(args.model)
    for key,rows,path in todo:
        print('scoring',args.model,key,len(rows),flush=True)
        predictions = score_dataset(key,rows,tokenizer,model,
                                    spec['max_length'],spec['batch_size'],path)
        assert [row['id'] for row in predictions] == [row['id'] for row in rows]
        print('complete',args.model,key,len(predictions),flush=True)


if __name__ == '__main__':
    main()
