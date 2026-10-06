"""Write sweeps/queue.json from arms.json: every arm of the chosen waves, in wave order, for the given seeds."""
import argparse, json
from pathlib import Path

R = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('--waves', nargs='*'); p.add_argument('--seeds', type=int, nargs='+', default=[1])
p.add_argument('--append', action='store_true'); a = p.parse_args()
arms = json.loads((R / 'arms.json').read_text()); q = json.loads((R / 'sweeps/queue.json').read_text()) if a.append else []
have = {j['tag'] for j in q}
for wave, group in arms['waves'].items():
    if a.waves and wave not in a.waves:
        continue
    for seed in a.seeds:
        for arm, extra in group.items():
            tag = f'q4b-{arm}-s{seed}'
            if tag not in have:
                q.append({'tag': tag, 'model': 'qwen35-4b', 'arm': arm, 'wave': wave, 'seed': seed,
                          'args': [*arms['_base'], *extra, '--seed', str(seed), '--arm', arm]})
(R / 'sweeps').mkdir(exist_ok=True); (R / 'sweeps/queue.json').write_text(json.dumps(q, indent=1)); print(len(q), 'jobs')
