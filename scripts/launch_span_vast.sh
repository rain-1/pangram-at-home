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
bash scripts/bootstrap_span_vast.sh > /workspace/pangram-data/span_v4_bootstrap.log 2>&1
bootstrap_status=$?
if [ "$bootstrap_status" -ne 0 ]; then
    PYTHONPATH=scripts python - <<'PY'
from pathlib import Path
from run_span_v4_experiment import atomic_json, export
root = Path('/workspace/pangram-data')
status = root / 'span_v4_status.json'
atomic_json(status, {'phase': 'failing', 'error': 'Bootstrap failed; see span_v4_bootstrap.log'})
info = export(root, status)
atomic_json(status, {'phase': 'failed', 'error': 'Bootstrap failed', 'export': info})
PY
    exit "$bootstrap_status"
fi
exec python -u scripts/run_span_v4_experiment.py
