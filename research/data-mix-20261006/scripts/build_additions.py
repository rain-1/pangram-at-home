"""Build additions-v1.jsonl.gz, the single input schema setup_mix.py reads.

Sources: rain1 v14 span rows (open-text-detector/span-detection-rain1-v14, config span_hardneg_v14: train split, plus the
GRADTEX span rows of new_source_holdout) and heterogeneous-ai-spans v1.3.0 (config all, train split only; val/test stay
held out). Hetero labels map human->0, ai->1; pair_id = source_id so a mixed document and its control share an epoch.
Usage: python build_additions.py V14_DIR HETERO_DIR OUT  (V14_DIR has span_hardneg_v14/*.parquet, HETERO_DIR has data/all/*.parquet)
"""
import ast, gzip, hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq

v14, het, out = map(Path, sys.argv[1:4]); rows = []


def v14_row(r):
    regs = ast.literal_eval(r['regions']) if isinstance(r['regions'], str) else r['regions']
    return {'id': r['id'], 'source_key': 'v14:' + r['source'], 'group_id': r['group_id'], 'pair_id': None, 'text': r['text'],
            'regions': [{'start': int(x['start']), 'end': int(x['end']), 'label': int(x['label'])} for x in regs],
            'generator': r['generator'], 'noncommercial': '[NONCOMMERCIAL]' in (r['license'] or '')}


for r in pq.read_table(next((v14 / 'span_hardneg_v14').glob('train-*.parquet'))).to_pylist():
    if r['label_kind'] == 'span':
        rows.append(v14_row(r))
for r in pq.read_table(next((v14 / 'span_hardneg_v14').glob('new_source_holdout-*.parquet'))).to_pylist():
    if r['label_kind'] == 'span' and 'GRADTEX' in r['source']:
        rows.append(v14_row(r))
lab = {'human': 0, 'ai': 1}
for r in pq.read_table(het / 'data/all/train.parquet').to_pylist():
    kind = 'mixed' if r['kind'] != 'human_control' else 'human_control'
    regs = [{'start': int(s['start']), 'end': int(s['end']), 'label': lab.get(s['label'], -100)} for s in r['spans']] if r['spans'] else [{'start': 0, 'end': len(r['text']), 'label': 0}]
    gens = sorted({s.get('requested_model') or '' for s in (r['spans'] or []) if s['label'] == 'ai'})
    rows.append({'id': r['id'], 'source_key': f"hetero:{kind}:{r['source_dataset']}", 'group_id': r['group_id'], 'pair_id': r['source_id'],
                 'text': r['text'], 'regions': regs, 'generator': ','.join(gens) or 'human', 'noncommercial': False})
b = ''.join(json.dumps(x) + '\n' for x in rows).encode(); out.write_bytes(gzip.compress(b))
print(len(rows), 'rows', hashlib.sha256(b).hexdigest())
