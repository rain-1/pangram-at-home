#!/usr/bin/env bash
set -uo pipefail
cd /workspace/pangram-at-home
set -a
source /workspace/span-runtime.env
set +a
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=pangram-at-home
export WANDB_ENTITY=eac-adsf
export WANDB_LOG_MODEL=false
export WANDB_DIR=/workspace/pangram-data/wandb
mkdir -p "$WANDB_DIR"
bash scripts/bootstrap_span_vast.sh > /workspace/pangram-data/span_size_curve_v5_bootstrap.log 2>&1
bootstrap_status=$?
if [ "$bootstrap_status" -ne 0 ]; then
    python - <<'PY'
import json
from pathlib import Path
Path('/workspace/pangram-data/span_size_curve_v5_status.json').write_text(
    json.dumps({'phase': 'bootstrap_failed',
                'error': 'See span_size_curve_v5_bootstrap.log'}, indent=2) + '\n')
PY
    exit "$bootstrap_status"
fi
exec python -u scripts/run_span_size_curve_v5.py
