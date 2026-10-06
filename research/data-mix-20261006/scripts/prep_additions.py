"""Convert v14 train + hetero train into one additions schema (same as setup_mix.py expects), plus hetero val/test for eval checks."""
import ast, gzip, json
import numpy as np, pandas as pd
from pathlib import Path

S = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'work'; OUT.mkdir(exist_ok=True)


def norm_regions(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return [{'start': int(g['start']), 'end': int(g['end']), 'label': int(g['label'])} for g in list(x)]


def v14():
    d = pd.read_parquet(S / 'hfdata/data/span_hardneg_v14/train-00000-of-00001.parquet')
    rows = []
    for r in d.itertuples():
        p = json.loads(r.provenance) if r.provenance else {}
        rows.append({'id': r.id, 'source_key': f'v14:{r.source}', 'group_id': r.group_id, 'pair_id': r.group_id, 'text': r.text,
                     'regions': norm_regions(r.regions), 'generator': r.generator, 'noncommercial': 'NONCOMMERCIAL' in (r.license or ''),
                     'license': r.license, 'label_kind': r.label_kind, 'construction': p.get('construction'), 'domain': p.get('domain'),
                     'prov_kind': p.get('kind'), 'scenario': p.get('scenario'), 'prov_generator': p.get('generator'),
                     'n_components': len(p.get('components') or [])})
    return rows


def hetero(split):
    d = pd.read_parquet(S / f'hetero/data/all/{split}.parquet')
    rows = []
    for r in d.itertuples():
        regs = [{'start': int(g['start']), 'end': int(g['end']), 'label': 1 if g['label'] == 'ai' else 0} for g in list(r.spans)]
        gens = sorted({g['requested_model'] for g in list(r.spans) if g['label'] == 'ai' and g['requested_model']})
        rows.append({'id': r.id, 'source_key': f'hetero:{r.kind}:{r.source_dataset}', 'group_id': r.group_id, 'pair_id': r.source_id,
                     'text': r.text, 'source_text': r.source_text, 'regions': regs, 'generator': ','.join(gens) or 'human',
                     'noncommercial': False, 'license': r.source_license, 'cohort': r.cohort, 'split': split,
                     'source_title': r.source_title, 'source_author': r.source_author, 'source_reference': r.source_reference})
    return rows


if __name__ == '__main__':
    for name, rows in [('v14-train', v14()), ('hetero-train', hetero('train')), ('hetero-validation', hetero('validation')), ('hetero-test', hetero('test'))]:
        with gzip.open(OUT / f'{name}.jsonl.gz', 'wt') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print(name, len(rows))
