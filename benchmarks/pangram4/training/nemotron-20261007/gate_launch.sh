#!/bin/bash
# Start run_arm.sh ARM FRACTION once the arm's GPUs have been idle (<1 GB used) for 5 consecutive minutes and no ablation or
# EmbeddingGemma training/scoring process is alive (their score workers would otherwise load a model onto a GPU the MoE fills).
R=/tmp/pangram-nemotron-20261007; cd $R; ARM=$1; FR=$2
GPUS=$(python3 -c "import json;print(','.join(json.load(open('moe.json'))['runs']['$ARM']['gpus']))")
busy() { for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in *train_sweep.py*qwen35*|*train_sweep.py*embgemma*|*cross_model_eval.py*|*sweep_eval.py*) case "$c" in *pangram-nemotron*) ;; *) return 0;; esac;; esac; done; return 1; }
idle=0; echo "{\"t\": \"$(date -u +%H:%M:%S)\", \"arm\": \"$ARM\", \"event\": \"gate_wait\", \"gpus\": \"$GPUS\"}" >> pipeline-$ARM.log
while [ $idle -lt 5 ]; do
  used=$(nvidia-smi --id=$GPUS --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$used" -lt 1000 ] && ! busy; then idle=$((idle+1)); else idle=0; fi
  sleep 60
done
exec ./run_arm.sh $ARM $FR
