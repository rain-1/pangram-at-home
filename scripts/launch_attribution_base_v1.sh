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
status=/workspace/pangram-data/attribution_base_v1_status.json
write_status() {
    python - "$status" "$1" "$2" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'phase':sys.argv[2],
    'exit_code':int(sys.argv[3]),'updated_at_unix':time.time()},indent=2)+'\n')
PY
}
write_status frozen_features_and_heads 0
python -u scripts/train_attribution_heads_v1.py --init base \
    > /workspace/pangram-data/attribution_base_frozen.log 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    write_status failed "$rc"
    exit "$rc"
fi
write_status full_backbone_training 0
python -u scripts/train_attribution_unfrozen_v1.py --init base \
    > /workspace/pangram-data/attribution_base_unfrozen.log 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    write_status complete 0
else
    write_status failed "$rc"
fi
exit "$rc"
