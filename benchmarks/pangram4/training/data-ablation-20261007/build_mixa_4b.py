"""Rebuild woog's moe-mixA-full prepared-v2 (10-08) as three stage-2 shards for the 4B ablation trainer (read-only on her folder).

Her merge_epochs.py concatenated the three disjoint shard files in order into stage2-epoch0 (sizes in manifest.merged_stage2.from),
so slicing the merged file by those sizes restores them exactly. Stage 1, selection and calibration windows are copied unchanged.
Output: runs/qwen35-4b-mixa/prepared-v2 under the ablation root, with a manifest in the format train_sweep.py checks.
Usage (in /tmp/pangram-ablation-20261007): python build_mixa_4b.py /tmp/mixfull-20261008/runs/moe-mixA-full/prepared-v2
"""
import gzip, hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1]); out = Path('runs/qwen35-4b-mixa/prepared-v2'); out.mkdir(parents=True, exist_ok=True)
man = json.loads((src / 'manifest.json').read_text()); sizes = man['merged_stage2']['from']
b = gzip.decompress((src / 'stage2-epoch0.jsonl.gz').read_bytes()); assert hashlib.sha256(b).hexdigest() == man['files']['stage2-epoch0']['sha256']
lines = b.splitlines(); assert len(lines) == sum(sizes.values())
files, counts, i = {}, {}, 0
for k in sorted(sizes):
    part = lines[i:i + sizes[k]]; i += sizes[k]
    data = b'\n'.join(part) + b'\n'; (out / f'{k}.jsonl.gz').write_bytes(gzip.compress(data, mtime=0))
    files[k] = {'rows': len(part), 'sha256': hashlib.sha256(data).hexdigest()}
    counts[k] = dict(Counter(json.loads(l)['dataset'] for l in part))
for k in ('stage1-epoch0', 'selection-windows', 'calibration-windows'):
    shutil.copyfile(src / f'{k}.jsonl.gz', out / f'{k}.jsonl.gz'); files[k] = man['files'][k]
    if k in man['counts']:
        counts[k] = man['counts'][k]
new = {**{k: v for k, v in man.items() if k not in ('files', 'counts', 'merged_stage2')}, 'files': files, 'counts': counts,
       'rebuilt_from': {'path': str(src), 'manifest_sha256': hashlib.sha256((src / 'manifest.json').read_bytes()).hexdigest(), 'merged_stage2': man['merged_stage2']}}
(out / 'manifest.json').write_text(json.dumps(new, indent=1))
print(json.dumps({k: v['rows'] for k, v in files.items()}))
