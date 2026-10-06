"""Launch the data-parallel MoE run from moe.json with torchrun (train_sweep.py reads WORLD_SIZE/RANK/LOCAL_RANK).

  python launch_moe.py {base,mix} --fraction F --preflight   # 30 steps, no validation, tag <tag>-preflight: check max_memory_gb <= ~76 and rank counts
  python launch_moe.py {base,mix} --fraction F               # the real run, then scores every saved checkpoint with sweep_eval.py on the first GPU
Resume state and rollback snapshots go to state/<tag> on local disk (optimizer state is ~4-5 GB per snapshot).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('run', choices=['base', 'mix']); p.add_argument('--fraction', type=float, required=True)
p.add_argument('--preflight', action='store_true'); p.add_argument('--resume', action='store_true'); a = p.parse_args()
all_ = json.loads((R / 'moe.json').read_text()); cfg = {**all_, **all_['runs'][a.run]}
tag = cfg['tag'] + ('-preflight' if a.preflight else '')
args = [*cfg['args'], '--fraction', str(a.fraction), '--state-dir', str(R / 'state'), *(cfg['preflight_args'] if a.preflight else []), *(['--resume'] if a.resume else [])]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(cfg['gpus']), 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
out = R / 'sweeps' / tag; out.mkdir(parents=True, exist_ok=True)
with open(out / 'train.log', 'a') as f:
    rc = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(cfg["gpus"])}',
                         str(R / 'train_sweep.py'), cfg['model'], tag, *args], env=env, cwd=R, stdout=f, stderr=f).returncode
if rc == 0 and not a.preflight:
    with open(out / 'eval.log', 'a') as f:
        rc = subprocess.run([sys.executable, '-u', str(R / 'sweep_eval.py'), tag], env={**env, 'CUDA_VISIBLE_DEVICES': cfg['gpus'][0]},
                            cwd=R, stdout=f, stderr=f).returncode
sys.exit(rc)
