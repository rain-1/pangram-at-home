"""Ablation scoring worker for one GPU (after woog's score_worker_t21.py). Claims (run, step) tasks whose inputs exist:
t21-heldout (strict held-out set), heldout-writers, cross-test (heterogeneous-ai-spans v1.4.0 TEST split only, all rows;
the arms train on its train split). Needs the run's sweep-eval sentence scores (written by queue_runner's sweep_eval).
Usage: score_worker_abl.py GPU"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-ablation-20261007'); S = R / 'sweeps'; gpu = sys.argv[1]
log = open(S / f'score-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{R}/vendor', TRACKIO_DIR=str(R / 'trackio'))
CK = 'stage2-epoch2'
STEPS = ('t21', 'heldout', 'cross')
def tags():
    q = json.loads((S / 'queue.json').read_text())
    return [j['tag'] for j in q] + ['q4b-T21-s1', 'q4b-T21-s2', 'q4b-T21-s3']  # woog's T2.1 seeds, re-scored on cross-test only
def wanted(t, step): return step == 'cross' or not t.startswith('q4b-T21-s')
def ready(t, step):
    d = S / t
    return (d / 'eval' / f'{CK}-sentences.npz').exists() and (d / f'{CK}-adapters.safetensors').exists()
def cmd(t, step):
    ck = str(S / t / f'{CK}-adapters.safetensors')
    if step == 't21': return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs/t21-heldout-score-rows.jsonl.gz'), 't21-heldout']
    if step == 'heldout': return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs/heldout-score-rows.jsonl.gz'), 'heldout-writers']
    return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'hetero-test'), 'cross-test', '0']
def claim():
    with open(S / 'score.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        left = [(t, s) for t in tags() for s in STEPS if wanted(t, s) and not (S / t / f'score-{s}.claim').exists()]
        for t, step in left:
            if (S / t).exists() and ready(t, step):
                (S / t / f'score-{step}.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return (t, step), left
        return None, left
say(event='start')
while True:
    if (S / 'STOP-SCORING').exists(): say(event='stop'); break
    job, left = claim()
    if job is None:
        time.sleep(60); continue
    t, step = job; t0 = time.time(); say(event='begin', tag=t, step=step)
    with open(S / t / f'score-{step}.log', 'a') as f:
        rc = subprocess.run(cmd(t, step), cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='end', tag=t, step=step, rc=rc, seconds=round(time.time() - t0))
    if rc: (S / t / f'score-{step}.failed').write_text(str(rc))
