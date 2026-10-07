#!/bin/bash
# One arm end to end on its 4 GPUs: train (launch_moe.py, which then runs sweep_eval.py on the first GPU), benchmark-v3 scoring
# sharded over the 4 GPUs, woog's held-out battery (t21-heldout, heldout-writers, hetero v1.4.0 test split), then copy the final
# adapters and records to /data. Usage: setsid nohup ./run_arm.sh nemo 0.2 &
R=/tmp/pangram-nemotron-20261007; cd $R; ARM=$1; FR=$2
TAG=$(python3 -c "import json;print(json.load(open('moe.json'))['runs']['$ARM']['tag'])")
GPUS=($(python3 -c "import json;print(' '.join(json.load(open('moe.json'))['runs']['$ARM']['gpus']))"))
log() { echo "{\"t\": \"$(date -u +%H:%M:%S)\", \"arm\": \"$ARM\", \"event\": \"$1\"}" >> pipeline-$ARM.log; }
alert() { log "failed: $1"; echo "$1" > sweeps/ALERT-$ARM; exit 1; }
log start
python3 -u launch_moe.py $ARM --fraction $FR > launch-$ARM.out 2>&1 || alert "launch_moe exit $?"
log trained_and_sweep_evaluated
CK=sweeps/$TAG/stage2-epoch2-adapters.safetensors; [ -f $CK ] || alert "missing $CK"
mkdir -p benchmark/scores
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=${GPUS[$i]} python3 -u benchmark/score_benchmark.py $TAG $(basename $CK) benchmark/inputs/benchmark-v3.jsonl.gz benchmark/scores --shard $i --nshards 4 > benchmark/score-$TAG-$i.log 2>&1 & done
wait; log benchmark_scored
CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 -u cross_model_eval.py $TAG $CK inputs/t21-heldout-score-rows.jsonl.gz t21-heldout > sweeps/$TAG/score-t21.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPUS[1]} python3 -u cross_model_eval.py $TAG $CK inputs/heldout-score-rows.jsonl.gz heldout-writers > sweeps/$TAG/score-heldout.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPUS[2]} python3 -u cross_model_eval.py $TAG $CK hetero-test cross-test 0 > sweeps/$TAG/score-cross.log 2>&1 &
wait; log battery_scored
D=/data/workspace/pangram-nemotron-20261007; mkdir -p $D/sweeps/$TAG $D/benchmark/scores
cp $CK sweeps/$TAG/*.json $D/sweeps/$TAG/ ; cp -r sweeps/$TAG/eval $D/sweeps/$TAG/ ; cp benchmark/scores/*$TAG* $D/benchmark/scores/ 2>/dev/null
log persisted; echo done > sweeps/$TAG/DONE
