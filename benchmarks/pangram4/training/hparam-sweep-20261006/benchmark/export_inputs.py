"""Export an aidet_eval benchmark's detector inputs exactly as `aidet_eval predict` builds them (run where the benchmark lives).

shared_windows_v1: input_id = window_id, text = examples.text[start:end]; native_truncate_v1: input_id = '<example_id>#native',
text = full canonical text. Detectors never see labels; only ids, text and offsets are exported.
Usage: python export_inputs.py BENCH_DIR OUT.jsonl.gz
"""
import gzip, hashlib, json, sys
from pathlib import Path
import pandas as pd

bench, out = Path(sys.argv[1]), Path(sys.argv[2])
ex = pd.read_parquet(bench / 'examples.parquet', columns=['example_id', 'text'])
texts = dict(zip(ex['example_id'], ex['text']))
win = pd.read_parquet(bench / 'windows' / 'shared_windows_v1.parquet')
rows = [{'input_id': w.window_id, 'protocol_id': 'shared_windows_v1', 'text': texts[w.example_id][w.start:w.end], 'offset': int(w.start)}
        for w in win.itertuples(index=False) if w.example_id in texts]
rows += [{'input_id': f'{e}#native', 'protocol_id': 'native_truncate_v1', 'text': t, 'offset': 0} for e, t in texts.items()]
manifest = json.loads((bench / 'manifest.json').read_text())
b = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
out.write_bytes(gzip.compress(b))
print(json.dumps({'benchmark_id': manifest['benchmark_id'], 'benchmark_hash': manifest['benchmark_hash'], 'inputs': len(rows),
                  'sha256': hashlib.sha256(b).hexdigest()}))
