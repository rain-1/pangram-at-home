"""Space-side setup for the hyperparameter sweep (run from /tmp/pangram-hparam-sweep-20261006 with the code already copied).

Follows splice-sweep-20261006/setup_splice.py: copies the qwen35-4b assets and prepared-v2 data from backbone-launch-20261003
(hash-checked), the fast10 vendor packages plus Trackio, and rebuilds the overnight sweep's evaluation rows from the frozen suite.
"""
import gzip, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

R = Path('/tmp/pangram-hparam-sweep-20261006'); W = Path('/data/workspace'); log = open(R / 'setup.log', 'a')
SUITE = {'workflow.jsonl.gz': 'c1b1ca20f59a5ae729b100c0ab548e99c79fe35ef2aca369c1791033b10a1da1',
         'comparison.jsonl.gz': '1c1d75d8f57b00762ba2798c11612662366de05a0bed56bf08a311b366a51fae'}


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush(); print(json.dumps(k), flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


say(event='start')
if not (R / 'vendor').exists():
    shutil.copytree(W / 'classifications/vendor-fast10', R / 'vendor')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', str(R / 'vendor-trackio'), 'trackio==0.40.0'], check=True, capture_output=True)
say(event='vendor_ready', packages=sorted(p.name for p in (R / 'vendor').iterdir() if p.is_dir() and not p.name.endswith('-info')))

src = W / 'backbone-launch-20261003/assets/qwen35-4b'; dst = R / 'assets/qwen35-4b'
if not dst.exists():
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('xet'))
for p in src.iterdir():
    if p.is_file():
        assert (dst / p.name).stat().st_size == p.stat().st_size, p.name
say(event='assets_copied', files=len(list(dst.iterdir())))

P0 = W / 'backbone-launch-20261003/runs/qwen35-4b/prepared-v2'; base = R / 'runs/qwen35-4b/prepared-v2'
if not base.exists():
    shutil.copytree(P0, base)
man = json.loads((base / 'manifest.json').read_text())
for k, v in man['files'].items():
    assert hashlib.sha256(gzip.decompress((base / f'{k}.jsonl.gz').read_bytes())).hexdigest() == v['sha256'], k
say(event='prepared_copied', manifest_sha256=sha(base / 'manifest.json'), counts=man['counts'].get('stage2-epoch0'))

models = json.loads((W / 'backbone-launch-20261003/models.json').read_text())
(R / 'models.json').write_text(json.dumps([m for m in models if m['name'] == 'qwen35-4b'], indent=1))

for name, h in SUITE.items():
    assert sha(W / 'evaluation-suite-v1-20261003' / name) == h, name
(R / 'sweeps').mkdir(exist_ok=True)
subprocess.run([sys.executable, str(R / 'build_eval_set.py')], check=True, cwd=R)
rows = R / 'sweeps/sweep-eval-rows.jsonl.gz'
say(event='eval_rows', rows=sum(1 for _ in gzip.open(rows, 'rt')), sha256=hashlib.sha256(gzip.decompress(rows.read_bytes())).hexdigest())
say(event='done')
