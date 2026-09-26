#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pangram-at-home
# This disposable PyTorch image uses Debian's externally-managed Python.
export PIP_BREAK_SYSTEM_PACKAGES=1
python - <<'PY'
import hashlib
import json
from pathlib import Path
manifest = json.loads(Path('span_package_manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = Path('/workspace') / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Package hash mismatch: {name}')
print('Package checksums verified', flush=True)
PY
python -m pip install -q -r requirements-span.txt
python -m pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true
python - <<'PY'
import torch
import transformers
from huggingface_hub import snapshot_download
assert torch.__version__.split('+')[0] == '2.11.0', torch.__version__
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert transformers.__version__ == '5.4.0'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
snapshot_download('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                  local_dir='/workspace/pangram-data/models/Qwen3-1.7B', max_workers=4)
print('Pinned backbone ready', flush=True)
PY
