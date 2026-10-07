#!/bin/bash
# After both MoE arms in /tmp/pangram-nemotron-20261007 are DONE: queue the typography arm (A2 + --typo-aug, seeds 1-2) and the
# deferred A8-double seed 3, then start queue runners on GPUs 0-2. score_worker_abl.py (still looping) scores them as usual.
R=/tmp/pangram-ablation-20261007; N=/tmp/pangram-nemotron-20261007; cd $R
until [ -f $N/sweeps/nemo-t21a2-s1/DONE -o -f $N/sweeps/ALERT-nemo ] && [ -f $N/sweeps/qwen-t21a2-s1/DONE -o -f $N/sweeps/ALERT-qwen ]; do sleep 120; done
python3 - <<'PY'
import json, fcntl
with open('sweeps/queue.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    q = json.load(open('sweeps/queue.json')); a2 = next(j for j in q if j['tag'] == 'q4b-ABL-A2-neutral-s1')
    new = [{**a2, 'tag': f'q4b-ABL-A2T-s{s}', 'arm': 'A2T-neutral-typo', 'seed': s,
            'args': [x if a2['args'][i - 1] != '--seed' else str(s) for i, x in enumerate(a2['args'])] + ['--typo-aug']} for s in (1, 2)]
    new += json.load(open('sweeps/queue-deferred.json'))
    json.dump(q + [j for j in new if j['tag'] not in {x['tag'] for x in q}], open('sweeps/queue.json', 'w'), indent=1)
    print(json.dumps([(j['tag'], j['args'][-3:]) for j in new]))
PY
rm -f sweeps/STOP
for g in 0 1 2; do setsid nohup python -u queue_runner.py $g < /dev/null >> sweeps/runner-gpu$g.out 2>&1 & done
echo "{\"t\": \"$(date -u +%H:%M:%S)\", \"event\": \"typo_and_a8s3_started\"}" >> setup.log
