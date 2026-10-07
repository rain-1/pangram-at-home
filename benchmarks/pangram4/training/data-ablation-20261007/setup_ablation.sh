#!/bin/bash
# Build /tmp/pangram-ablation-20261007 from woog's T2.1 setup (read-only use of /tmp/pangram-t2-20261006), then the arms.
set -e
R=/tmp/pangram-ablation-20261007; W=/tmp/pangram-t2-20261006; cd $R
log() { echo "{\"t\": \"$(date -u +%H:%M:%S)\", \"event\": \"$1\"}" | tee -a setup.log; }
log start
for f in train_sweep.py queue_runner.py sweep_eval.py cross_model_eval.py modeling_sweep.py data.py modeling.py adapters_short.py persist_daemon.py models.json; do cp $W/$f .; done
[ -d vendor ] || cp -a $W/vendor .
mkdir -p assets runs sweeps
[ -d assets/qwen35-4b ] || cp -a $W/assets/qwen35-4b assets/
ln -sfn $R/assets/qwen35-4b assets/qwen35-4b-t21
[ -d runs/qwen35-4b-t21 ] || cp -a $W/runs/qwen35-4b-t21 runs/
[ -d inputs ] || cp -a $W/inputs .
cp $W/sweeps/sweep-eval-rows.jsonl.gz sweeps/
log copied
cat > runtime.py <<'PY'
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
sys.path.append(str(ROOT/'vendor-trackio'))  # trackio and its deps last, so they never shadow the training stack
SPACE_CACHE=str(ROOT/'assets')
TRACKIO_PROJECT='pangram-data-ablation-20261007'
def require_space():
 assert str(ROOT)=='/tmp/pangram-ablation-20261007'
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('TRACKIO_DIR',str(ROOT/'trackio'))
PY
python patch_trackio.py
[ -d vendor-trackio ] || python -m pip install -q --target vendor-trackio trackio==0.40.0
log patched
# woog's T2.1 seeds: link checkpoint + sweep-eval scores so they can be re-scored on the test-only cross set
for s in 1 2 3; do t=q4b-T21-s$s; mkdir -p sweeps/$t/eval; ln -sfn $W/sweeps/$t/stage2-epoch2-adapters.safetensors sweeps/$t/stage2-epoch2-adapters.safetensors
  cp $W/sweeps/$t/eval/stage2-epoch2-sentences.npz $W/sweeps/$t/eval/stage2-epoch2.json sweeps/$t/eval/; cp $W/sweeps/$t/run.json sweeps/$t/; done
python - <<'PY'
import json
v=json.load(open('specs/views.json'))
for arm in sorted(v):
    print(arm, v[arm])
PY
for spec in specs/A*.json; do arm=$(basename $spec .json); view=$(python -c "import json;print(json.load(open('specs/views.json'))['$arm'])")
  name=qwen35-4b-abl-$(echo $arm | tr A-Z a-z)
  [ -d runs/$name ] || nice python -u setup_mix.py --base qwen35-4b-t21 --name $name --spec $spec --drops leakage-drop-ids.json --additions additions-hetero-$view.jsonl.gz --extra-held inputs/t21-heldout-score-rows.jsonl.gz inputs/heldout-score-rows.jsonl.gz > setup-$arm.out 2>&1
  log "built-$arm"; done
python build_queue_abl.py
log done
