"""Evaluate Pangram EditLens checkpoints as pure-document baselines.

The upstream EditLens inference score is E[bucket]/(num_buckets-1). We apply
its text normalization, split long documents into overlapping native-token
windows, average window scores per document, and select a threshold using
only the shared pure-human calibration set. Upstream reference:
https://github.com/pangramlabs/EditLens/tree/05a588f15d792330ccaf91be8ee4fdb54ce26835
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import emoji
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/mnt/f/pangram-at-home')
SOURCES = {
    'calibration': (ROOT / 'data/span_human_eval_v2/calibration.jsonl', {'human'}),
    'llmtrace_pure': (ROOT / 'data/span_size_curve_v5/size_20000/test_llmtrace.jsonl', {'human', 'ai'}),
    'synthetic_v4_pure': (ROOT / 'data/span_training_v4/val.jsonl', {'human', 'ai'}),
    'locked_human': (ROOT / 'data/span_human_eval_v2/test.jsonl', {'human'}),
}
MODELS = {
    'roberta': {
        'checkpoint': ROOT / 'models/pangram_editlens_roberta-large',
        'source': 'pangram/editlens_roberta-large',
        'revision': 'f93e1ace74528cfb48f337ab2fe946fb71a728cb',
        'max_length': 512,
        'batch_size': 16,
    },
    'llama': {
        'checkpoint': ROOT / 'models/pangram_editlens_Llama-3.2-3B',
        'base': ROOT / 'models/meta-llama_Llama-3.2-3B',
        'source': 'pangram/editlens_Llama-3.2-3B',
        'revision': 'b5f8044f631f5b455eafbcb569dcf175f2b0726d',
        'max_length': 1024,
        'batch_size': 4,
    },
}


class NormedLinear(torch.nn.Module):
    """Pangram's LayerNorm then bias-free Linear sequence head."""
    def __init__(self, hidden_size: int, num_labels: int, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))


def clean_text(text: str) -> str:
    """Match upstream EditLens scripts/preprocess.py normalization."""
    text = emoji.demojize(text)
    if '</think>' in text:
        text = text.split('</think>')[1].strip()
    paragraphs = [p for p in text.split('\n') if p.strip()]
    if paragraphs:
        first = re.sub(r'^[^a-zA-Z0-9]*', '', paragraphs[0])
        first = emoji.replace_emoji(first, '')
        if any(first.startswith(phrase) for phrase in
               ('Sure', 'Here', 'Abstract', 'Title', "I'm happy to help", 'Certainly')):
            if len(paragraphs) > 1:
                text = '\n'.join(paragraphs[1:])
    return re.sub(r'\s+', ' ', text.lower()).strip()


def load_model(which: str):
    spec = MODELS[which]
    if which == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint'], use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            spec['checkpoint'], dtype=torch.bfloat16).to('cuda')
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec['base'], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForSequenceClassification.from_pretrained(
            spec['base'], num_labels=4, quantization_config=quant, device_map='auto',
            dtype=torch.bfloat16)
        base.config.pad_token_id = tokenizer.pad_token_id
        base.score = NormedLinear(base.config.hidden_size, 4,
                                  device=next(base.parameters()).device, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, spec['checkpoint'])
    model.eval()
    assert model.config.num_labels == 4
    return tokenizer, model


def starts(length: int, width: int, stride: int) -> list[int]:
    if length <= width:
        return [0]
    points = list(range(0, length - width + 1, stride))
    final = length - width
    if points[-1] != final:
        points.append(final)
    return points


def read_rows(path: Path, kinds: set[str], limit: int | None) -> list[dict]:
    rows = [row for line in path.open() if (row := json.loads(line))['kind'] in kinds]
    return rows[:limit] if limit is not None else rows


def score_dataset(name: str, rows: list[dict], tokenizer, model, max_length: int,
                  batch_size: int, output: Path) -> list[dict]:
    probe = tokenizer('window', add_special_tokens=False)['input_ids']
    with_special = tokenizer('window', add_special_tokens=True)['input_ids']
    probe_start = next(i for i in range(len(with_special) - len(probe) + 1)
                       if with_special[i:i + len(probe)] == probe)
    prefix = with_special[:probe_start]
    suffix = with_special[probe_start + len(probe):]
    body_width = max_length - len(prefix) - len(suffix)
    window_ids: list[list[int]] = []
    owner: list[int] = []
    for index, row in enumerate(rows):
        ids = tokenizer(clean_text(row['text']), add_special_tokens=False)['input_ids']
        for start in starts(len(ids), body_width, body_width // 2):
            window_ids.append(prefix + ids[start:start + body_width] + suffix)
            owner.append(index)
    sums = np.zeros(len(rows), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int32)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(window_ids), batch_size):
            batch = tokenizer.pad({'input_ids': window_ids[start:start + batch_size]},
                                  padding=True, return_tensors='pt').to(device)
            logits = model(**batch).logits.float()
            probs = logits.softmax(-1)
            positions = torch.arange(4, device=probs.device, dtype=probs.dtype) / 3
            scores = (probs @ positions).cpu().numpy()
            for idx, score in enumerate(scores, start):
                sums[owner[idx]] += float(score)
                counts[owner[idx]] += 1
            if start // batch_size % 100 == 0:
                print(name, f'{min(start + batch_size, len(window_ids))}/{len(window_ids)} windows', flush=True)
    if (counts == 0).any():
        raise RuntimeError('One or more documents have no windows')
    result = []
    for row, score, count in zip(rows, sums / counts, counts):
        result.append({'id': row['id'], 'kind': row['kind'], 'domain': row.get('domain'),
                       'score': float(score), 'windows': int(count)})
    output.write_text(''.join(json.dumps(item) + '\n' for item in result))
    return result


def threshold_at_human_fpr(scores: np.ndarray, target: float) -> float:
    allowed = math.floor(len(scores) * target)
    values = np.sort(scores)
    if allowed == 0:
        return float(np.nextafter(values[-1], np.inf))
    return float(np.nextafter(values[-(allowed + 1)], np.inf))


def metrics(rows: list[dict], threshold: float) -> dict:
    human = [r for r in rows if r['kind'] == 'human']
    ai = [r for r in rows if r['kind'] == 'ai']
    fp = sum(r['score'] >= threshold for r in human)
    tp = sum(r['score'] >= threshold for r in ai)
    result = {'documents': len(rows), 'human_documents': len(human), 'ai_documents': len(ai),
              'human_false_alarms': fp, 'human_fpr': fp / len(human) if human else None,
              'ai_detected': tp, 'ai_recall': tp / len(ai) if ai else None}
    if human and ai:
        result['auroc'] = float(roc_auc_score([r['kind'] == 'ai' for r in rows],
                                             [r['score'] for r in rows]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sets', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    spec = MODELS[args.model]
    output = ROOT / 'runs' / f'open_pangram_editlens_{args.model}_v5'
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model)
    all_scores = {}
    for name in args.sets:
        path, kinds = SOURCES[name]
        rows = read_rows(path, kinds, args.limit)
        destination = output / f'{name}{"_smoke" if args.limit else ""}.jsonl'
        all_scores[name] = score_dataset(name, rows, tokenizer, model,
                                         spec['max_length'], spec['batch_size'], destination)
    if args.limit:
        return
    if 'calibration' not in all_scores:
        raise ValueError('Full evaluation requires calibration set')
    threshold = threshold_at_human_fpr(np.array([r['score'] for r in all_scores['calibration']]), .05)
    summary = {'model': spec['source'], 'revision': spec['revision'],
               'score_definition': 'mean over native-token windows of upstream E[bucket]/3',
               'window_size': spec['max_length'], 'window_stride':
                   (spec['max_length'] - tokenizer.num_special_tokens_to_add(pair=False)) // 2,
               'threshold_source': 'separate pure-human calibration, at most 5% document FPR',
               'threshold': threshold,
               'sets': {name: metrics(rows, threshold) for name, rows in all_scores.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
