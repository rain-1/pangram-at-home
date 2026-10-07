"""Write sweeps/queue.json: control (1 seed) + each ablation arm x seeds 1-3, seed-major so every arm has seed 1 first.
Recipe = woog's T2.1 runs (run_pair_t21.sh): 20% length, micro-batch 8, LR 2e-4, head LR 2e-5, cosine, 6% warmup."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent
arms = sorted(p.stem for p in (R / 'specs').glob('A*.json'))
base = ['--fraction', '0.2', '--micro-batch', '8', '--lr', '2e-4', '--head-lr', '2e-5', '--schedule', 'cosine', '--warmup', '0.06']
q = [{'tag': 'q4b-ABL-C-s1', 'model': 'qwen35-4b-t21', 'arm': 'C', 'seed': 1, 'args': base + ['--seed', '1']}]
for seed in (1, 2, 3):
    for arm in arms:
        q.append({'tag': f'q4b-ABL-{arm}-s{seed}', 'model': f'qwen35-4b-abl-{arm.lower()}', 'arm': arm, 'seed': seed, 'args': base + ['--seed', str(seed)]})
(R / 'sweeps').mkdir(exist_ok=True); (R / 'sweeps/queue.json').write_text(json.dumps(q, indent=1)); print(len(q), 'jobs')
