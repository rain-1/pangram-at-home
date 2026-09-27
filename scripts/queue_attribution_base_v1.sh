#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pangram-at-home
while true; do
    phase=$(python - <<'PY'
import json
from pathlib import Path
path=Path('/workspace/pangram-data/attribution_unfrozen_v1_status.json')
print(json.loads(path.read_text())['phase'] if path.exists() else 'pending')
PY
)
    case "$phase" in
        complete) break ;;
        failed) echo 'v10 trainable run failed; base queue will not start' >&2; exit 1 ;;
        *) sleep 20 ;;
    esac
done
bash scripts/launch_attribution_base_v1.sh
