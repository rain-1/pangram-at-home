#!/usr/bin/env bash
set -uo pipefail
cd /workspace/pangram-at-home
set -a
source /workspace/span-v14-runtime.env
set +a
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=pangram-at-home
export WANDB_ENTITY=eac-adsf
export WANDB_LOG_MODEL=false
export WANDB_DIR=/workspace/pangram-data/wandb
mkdir -p "$WANDB_DIR"
bash scripts/bootstrap_span_hardneg_v14.sh > /workspace/pangram-data/span_v14_bootstrap.log 2>&1
bootstrap_rc=$?
if [ "$bootstrap_rc" -ne 0 ]; then
    python - <<'PY'
import json
from pathlib import Path
p=Path('/workspace/pangram-data/span_hardneg_v14_status.json')
p.write_text(json.dumps({'phase':'failed','error':'bootstrap failed; see span_v14_bootstrap.log'},indent=2)+'\n')
PY
    exit "$bootstrap_rc"
fi
exec python -u scripts/run_span_hardneg_v14.py
