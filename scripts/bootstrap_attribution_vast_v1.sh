#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pangram-at-home
export PIP_BREAK_SYSTEM_PACKAGES=1
python -m pip install -q -r requirements-span.txt
python -m pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true
python - <<'PY'
from pathlib import Path
import torch
from huggingface_hub import snapshot_download

assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('GPU:',torch.cuda.get_device_name(0),flush=True)
destination=Path('/workspace/pangram-data/models/Qwen3-1.7B')
snapshot_download('Qwen/Qwen3-1.7B',
                  revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                  local_dir=destination,max_workers=4)
assert (destination/'model.safetensors.index.json').exists()
print('Pinned backbone ready',flush=True)
PY
