"""Count supervised token exposure for the v12 fixed-size mixture."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from transformers import AutoTokenizer
from span_data import encode_document, window_starts

ROOT = Path('/mnt/f/pangram-at-home')
DATA = ROOT/'data/span_new_sources_v12'


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    source_tokens = defaultdict(Counter)
    class_tokens = Counter()
    source_windows = Counter()
    rows = 0
    for line in (DATA/'train.jsonl').open():
        row = json.loads(line)
        ids, _, labels = encode_document(row, tokenizer)
        rows += 1
        source = row['source']
        for start in window_starts(len(ids), 512, 256):
            count = Counter(labels[start:start+512])
            class_tokens[0] += count[0]
            class_tokens[1] += count[1]
            source_tokens[source][0] += count[0]
            source_tokens[source][1] += count[1]
            source_windows[source] += 1
    total = sum(class_tokens.values())
    report = {
        'documents': rows,
        'windows': sum(source_windows.values()),
        'supervised_token_positions': total,
        'class_supervised_tokens': dict(class_tokens),
        'ai_supervised_token_fraction': class_tokens[1]/total,
        'source_supervised_token_fraction': {
            name: sum(count.values())/total for name, count in source_tokens.items()},
        'source_windows': dict(source_windows),
        'source_class_supervised_tokens': {
            name: dict(count) for name, count in source_tokens.items()},
    }
    (DATA/'exposure_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    assert rows == 20000
    assert .42 <= report['ai_supervised_token_fraction'] <= .58
    assert max(report['source_supervised_token_fraction'].values()) < .35
    assert report['source_supervised_token_fraction'].get('LLMTrace_detection', 0) <= .03
    print(json.dumps({key: report[key] for key in
        ('documents', 'windows', 'ai_supervised_token_fraction',
         'source_supervised_token_fraction')}, indent=2))


if __name__ == '__main__':
    main()
