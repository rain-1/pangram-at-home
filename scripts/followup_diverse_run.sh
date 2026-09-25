#!/usr/bin/env bash
set -euo pipefail

training_pid="${1:?training PID required}"
wandb_run_id="${2:?W&B run ID required}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="/mnt/f/pangram-at-home/runs/qwen3_17b_diverse_v1"
cd "$repo_dir"

while kill -0 "$training_pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$run_dir/train_summary.json" || ! -f "$run_dir/best_adapter/adapter_model.safetensors" ]]; then
  echo "Training ended without a final adapter or summary" >&2
  exit 1
fi

set -a
source "$repo_dir/.env"
set +a
export WANDB_DIR="/mnt/f/pangram-at-home/wandb"
export WANDB_CONSOLE="off"
python -u scripts/evaluate_diverse_lora.py --wandb-run-id "$wandb_run_id"
if ! python -u scripts/run_editlens_reference.py --dataset diverse --tier full --model roberta > "$run_dir/reference_roberta_eval.log" 2>&1; then
  echo "RoBERTa reference evaluation failed; see $run_dir/reference_roberta_eval.log" >&2
fi
if ! python -u scripts/run_editlens_reference.py --dataset diverse --tier full --model llama > "$run_dir/reference_llama_eval.log" 2>&1; then
  echo "Llama reference evaluation failed; see $run_dir/reference_llama_eval.log" >&2
fi
MPLBACKEND=Agg python scripts/chart_diverse_results.py
git add reports/metrics/qwen3_17b_diverse_v1.json reports/diverse_full_results_v1.pdf
for model_name in roberta llama; do
  report_path="reports/metrics/reference_diverse_${model_name}_full.json"
  if [[ -f "$report_path" ]]; then
    git add "$report_path"
  fi
done
git commit -m "Report Qwen diverse evaluation against baselines"
git push origin main
