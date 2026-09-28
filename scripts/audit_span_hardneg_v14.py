"""Audit v14 token exposure and split integrity before remote training."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from transformers import AutoTokenizer

from span_data import encode_document, window_starts


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
FOLDER = REPO/'data/span_hardneg_v14'


def main() -> None:
    manifest = json.loads((FOLDER/'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    source_tokens: dict[str, Counter] = defaultdict(Counter)
    kind_tokens: dict[str, Counter] = defaultdict(Counter)
    class_tokens = Counter()
    source_windows = Counter()
    seen_ids = set()
    source_groups = set()
    for line in (FOLDER/'train.jsonl').open():
        row = json.loads(line)
        assert row['id'] not in seen_ids, row['id']
        seen_ids.add(row['id'])
        assert row['spans'][0]['start'] == 0 and row['spans'][-1]['end'] == len(row['text'])
        assert all(span['end'] > span['start'] for span in row['spans'])
        if row['id'].startswith(('human_eval_v2:train:', 'v14:mage-train:')):
            source_groups.update(row.get('source_groups', []))
        ids, _, labels = encode_document(row, tokenizer)
        for start in window_starts(len(ids), 512, 256):
            count = Counter(labels[start:start+512])
            class_tokens.update(count)
            source_tokens[row['source']].update(count)
            kind_tokens[row['kind']].update(count)
            source_windows[row['source']] += 1
    assert len(seen_ids) == manifest['documents'] == 21200
    for name in ('val.jsonl', 'new_source_holdout.jsonl'):
        for line in (FOLDER/name).open():
            row = json.loads(line)
            assert row['id'] not in seen_ids, (name, row['id'])
            assert not source_groups.intersection(row.get('source_groups', [])), (name, row['id'])
    supervised = class_tokens[0]+class_tokens[1]
    report = {'documents': len(seen_ids), 'windows': sum(source_windows.values()),
              'supervised_token_positions': supervised,
              'class_supervised_tokens': dict(class_tokens),
              'ai_supervised_token_fraction': class_tokens[1]/supervised,
              'source_supervised_token_fraction': {
                  name: (count[0]+count[1])/supervised
                  for name,count in source_tokens.items()},
              'source_windows': dict(source_windows),
              'source_class_supervised_tokens': {
                  name: dict(count) for name,count in source_tokens.items()},
              'kind_class_supervised_tokens': {
                  name: dict(count) for name,count in kind_tokens.items()}}
    (FOLDER/'exposure_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert .40 <= report['ai_supervised_token_fraction'] <= .50
    assert max(report['source_supervised_token_fraction'].values()) < .35
    assert report['source_supervised_token_fraction']['LLMTrace_detection'] <= .03
    assert report['source_supervised_token_fraction']['persuade_2.0'] <= .05
    assert report['source_supervised_token_fraction']['writers.stackexchange.com'] <= .03
    print(json.dumps({k:report[k] for k in
        ('documents','windows','ai_supervised_token_fraction',
         'source_supervised_token_fraction')},indent=2))


if __name__ == '__main__':
    main()
