#!/usr/bin/env bash
set -uo pipefail
cd /workspace/pangram-at-home
set -a
source /workspace/attribution-runtime.env
set +a
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=pangram-at-home
export WANDB_ENTITY=eac-adsf
export WANDB_LOG_MODEL=false
export WANDB_DIR=/workspace/pangram-data/wandb
mkdir -p "$WANDB_DIR" /workspace/pangram-data/runs/attribution_heads_v1
status=/workspace/pangram-data/attribution_heads_v1_status.json
write_status() {
    python - "$status" "$1" "$2" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1])
path.write_text(json.dumps({'phase':sys.argv[2],'exit_code':int(sys.argv[3]),
                            'updated_at_unix':time.time()},indent=2)+'\n')
PY
}
write_status bootstrapping 0
bash scripts/bootstrap_attribution_vast_v1.sh > /workspace/pangram-data/attribution_bootstrap.log 2>&1
bootstrap_rc=$?
if [ "$bootstrap_rc" -ne 0 ]; then
    write_status failed "$bootstrap_rc"
    exit "$bootstrap_rc"
fi
write_status training 0
python -u scripts/train_attribution_heads_v1.py > /workspace/pangram-data/attribution_train.log 2>&1
train_rc=$?
if [ "$train_rc" -ne 0 ]; then
    write_status failed "$train_rc"
    exit "$train_rc"
fi
write_status complete 0
