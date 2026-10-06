"""Unattended chain for the matched MoE runs (one process per arm; runs on the Space, independent of any client session).

For each seed: wait for the arm's training to finish (launch_moe.py then scores its checkpoints on the arm's first GPU with
sweep_eval.py), meanwhile score the final checkpoint on the exported benchmark with benchmark/score_benchmark.py sharded over the
arm's other three GPUs, then start the next seed. Progress and failures go to pipeline-<arm>.log; ALERT-pipeline-<arm> on failure.
  setsid nohup python -u pipeline.py base --seeds 1 2 &
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parent; S = R / 'sweeps'
p = argparse.ArgumentParser(); p.add_argument('arm'); p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
p.add_argument('--fraction', default='0.5'); p.add_argument('--inputs', default=str(R / 'benchmark/v3-inputs.jsonl.gz'))
a = p.parse_args(); cfg = json.loads((R / 'moe.json').read_text())['runs'][a.arm]; log = open(R / f'pipeline-{a.arm}.log', 'a')


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'arm': a.arm, **k}) + '\n'); log.flush()


def alive(proc_pattern):
    for d in Path('/proc').iterdir():
        if d.name.isdigit():
            try:
                if proc_pattern in (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='ignore'):
                    return True
            except OSError:
                pass
    return False


def state(tag):
    f = S / tag / 'status.json'
    return json.loads(f.read_text()).get('state') if f.exists() else None


def fail(msg):
    say(event='failed', error=msg); (S / f'ALERT-pipeline-{a.arm}').write_text(msg); sys.exit(1)


for seed in a.seeds:
    tag = cfg['tag'].rsplit('-s', 1)[0] + f'-s{seed}'
    if not alive(f'launch_moe.py {a.arm}') and state(tag) not in ('training', 'trained', 'loading_model'):
        say(event='launch', tag=tag)
        subprocess.Popen([sys.executable, '-u', str(R / 'launch_moe.py'), a.arm, '--fraction', a.fraction, '--seed', str(seed)],
                         cwd=R, stdout=open(R / f'launch-{a.arm}-s{seed}.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(120)
    while state(tag) not in ('trained', 'failed', 'diverged'):
        if not alive(f'launch_moe.py {a.arm}') and state(tag) not in ('trained',):
            time.sleep(30)
            if state(tag) not in ('trained', 'failed', 'diverged'):
                fail(f'{tag}: launcher exited before training finished (state={state(tag)})')
        time.sleep(60)
    if state(tag) != 'trained':
        fail(f'{tag}: training ended in state {state(tag)}')
    say(event='trained', tag=tag)
    ck = sorted((S / tag).glob('stage2-epoch*-adapters.safetensors'))[-1].name
    shards = []
    for i, g in enumerate(cfg['gpus'][1:]):
        shards.append(subprocess.Popen([sys.executable, '-u', str(R / 'benchmark/score_benchmark.py'), tag, ck, a.inputs, str(R / 'benchmark/scores'),
                                        '--shard', str(i), '--nshards', str(len(cfg['gpus']) - 1)], cwd=R, env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': g},
                                       stdout=open(R / f'benchmark/score-{tag}-{i}.log', 'a'), stderr=subprocess.STDOUT))
    say(event='benchmark_scoring_started', tag=tag, checkpoint=ck)
    codes = [s.wait() for s in shards]
    say(event='benchmark_scoring_done', tag=tag, codes=codes)
    while alive(f'launch_moe.py {a.arm}'):  # sweep_eval of this seed still running on the first GPU
        time.sleep(60)
    say(event='seed_done', tag=tag, evals=sorted(p.name for p in (S / tag / 'eval').glob('*.json')) if (S / tag / 'eval').exists() else [])
    if any(codes):
        (S / f'ALERT-pipeline-{a.arm}').write_text(f'{tag}: benchmark scoring shard exit codes {codes}')
say(event='all_done')
