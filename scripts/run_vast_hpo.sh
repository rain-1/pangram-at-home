#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pangram-at-home
root="/workspace/pangram-data"
trap 'date -u > "$root/HPO_FAILED"' ERR
set -a
source /workspace/pangram-at-home/.env
set +a
export PANGRAM_DATA_ROOT="$root"
export WANDB_DIR="$root/wandb"
export WANDB_CONSOLE=off
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PANGRAM_TUNE_QUANTIZATION=none

python -u scripts/tune_diverse_ray.py --root "$root" --mode hpo --search random \
  --trials 24 --max-examples 25600 --eval-examples 3200 --report-to wandb \
  --name hpo_diverse_v2 > "$root/hpo_driver.log" 2>&1
python scripts/select_and_export_tuning.py --root "$root" --hpo-name hpo_diverse_v2 \
  --ablation-name ablation_diverse_v2 --select-only \
  > "$root/hpo_selection.log" 2>&1
python -u scripts/tune_diverse_ray.py --root "$root" --mode ablation \
  --best-config "$root/runs/hpo_diverse_v2_best_config.json" --report-to wandb \
  --name ablation_diverse_v2 > "$root/ablation_driver.log" 2>&1
python scripts/select_and_export_tuning.py --root "$root" \
  --hpo-name hpo_diverse_v2 --ablation-name ablation_diverse_v2 \
  > "$root/hpo_export.log" 2>&1
date -u > "$root/HPO_DONE"
