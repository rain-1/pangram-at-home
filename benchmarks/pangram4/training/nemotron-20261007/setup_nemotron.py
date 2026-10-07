"""Space-side setup for the Nemotron 3.5 Lightning run (run from /tmp/pangram-nemotron-20261007 with the code copied in).

Same stack as the Qwen3.6-35B-A3B MoE runs (system transformers 5.17 + classifications/vendor-fast10 PEFT 0.18.1, Trackio
last on the path). Weights were downloaded on the Space to /data/workspace/model-cache/nemotron35-lightning-30b-a3b and are
copied to local disk for loading speed. prepared-v2 is rebuilt with Nemotron's own tokenizer by woog's prepare_model.py
(same seeds and pools, so the same documents are drawn; only the 510-token crop points differ), exactly as each backbone
in backbone-launch-20261003 got its own. Evaluation rows are rebuilt with the sweep's build_eval_set.py.
"""
import gzip, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

R = Path('/tmp/pangram-nemotron-20261007'); W = Path('/data/workspace'); log = open(R / 'setup.log', 'a')
NAME = 'nemotron35-30b-a3b'; REPO = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'; REV = 'a9904d24bcc1d289a1950fa9d2b978c47cf903b9'
SRC = W / 'model-cache/nemotron35-lightning-30b-a3b'; BL = W / 'backbone-launch-20261003'


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush(); print(json.dumps(k), flush=True)


say(event='start')
if not (R / 'vendor').exists():
    shutil.copytree(W / 'classifications/vendor-fast10', R / 'vendor')
if not (R / 'vendor-trackio').exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', str(R / 'vendor-trackio'), 'trackio==0.40.0'], check=True, capture_output=True)
dst = R / 'assets' / NAME
if not dst.exists():
    shutil.copytree(SRC, dst.with_suffix('.partial'), ignore=shutil.ignore_patterns('.cache')); dst.with_suffix('.partial').rename(dst)
for p in SRC.iterdir():
    if p.is_file():
        assert (dst / p.name).stat().st_size == p.stat().st_size, p.name
say(event='assets_copied', files=len(list(dst.iterdir())))
(R / 'models.json').write_text(json.dumps([{'name': NAME, 'repo': REPO, 'kind': 'causal', 'revision': REV, 'host': 'space',
                                             'lora_rank': 128, 'expert_rank': 16}], indent=1))
for n in ('source-pools', 'reservation.json'):
    if not (R / n).exists():
        (R / n).symlink_to(BL / n)
subprocess.run([sys.executable, str(R / 'prepare_model.py'), NAME], check=True, cwd=R)
man = json.loads((R / 'runs' / NAME / 'prepared-v2/manifest.json').read_text())
say(event='prepared', counts=man['counts'].get('stage2-epoch0'), manifest_sha256=hashlib.sha256((R / 'runs' / NAME / 'prepared-v2/manifest.json').read_bytes()).hexdigest())
(R / 'sweeps').mkdir(exist_ok=True)
subprocess.run([sys.executable, str(R / 'build_eval_set.py')], check=True, cwd=R)
rows = R / 'sweeps/sweep-eval-rows.jsonl.gz'
say(event='eval_rows', rows=sum(1 for _ in gzip.open(rows, 'rt')), sha256=hashlib.sha256(gzip.decompress(rows.read_bytes())).hexdigest())
say(event='done')
