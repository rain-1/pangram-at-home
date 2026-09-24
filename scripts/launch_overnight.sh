#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/code/pangram-at-home
export PANGRAM_DATA_ROOT="${PANGRAM_DATA_ROOT:-/mnt/f/pangram-at-home}"
export TOKENIZERS_PARALLELISM=false

python -u scripts/train_segment_lora.py --run-name qwen3_17b_mixed_stage1_v1
python -u scripts/evaluate_segment_lora.py --run-name qwen3_17b_mixed_stage1_v1
