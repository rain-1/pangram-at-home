#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pangram-at-home
python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/workspace')
manifest = json.loads((root / 'pangram-at-home/package_manifest.json').read_text())
for name, expected in manifest['files'].items():
    h = hashlib.sha256()
    with (root / name).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise SystemExit(f'Package hash mismatch: {name}')
print('Verified', len(manifest['files']), 'package files; no test sets included.')
PY
python -m pip install -q -r requirements-tune.txt
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
manifest = json.loads(Path('/workspace/pangram-at-home/package_manifest.json').read_text())
model = manifest['model']
snapshot_download(repo_id=model['repo'], revision=model['revision'],
                  local_dir='/workspace/pangram-data/models/Qwen3-1.7B', max_workers=4)
print('Pinned model downloaded:', model['repo'], model['revision'])
PY
python - <<'PY'
import torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.is_bf16_supported(), 'bf16 not supported on selected GPU'
print('CUDA GPUs:', torch.cuda.device_count())
print('Train set present:', Path('/workspace/pangram-data/data/diverse_pyramid_v1/train_full.parquet').exists())
PY
