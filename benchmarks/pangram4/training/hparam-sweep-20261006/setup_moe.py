"""Stage Qwen3.6-35B-A3B in the sweep folder (run after setup_space.py; disk only, no GPU).

Copies the base assets and the MoE's own prepared-v2 (manifest f24335dc..., the data woog's moe-A-* runs used; windows are
tokenizer-specific, so it differs from the 4B's), registers the model with its LoRA ranks (128, experts 16), and stages
woog's never-evaluated moe-A-full-lr1e4 adapters as sweeps/moe-A-full-lr1e4-woog for sweep_eval.py. No status.json is
written there, so sweep_eval never prunes those (symlinked) checkpoints.
"""
import gzip, hashlib, json, shutil, time
from pathlib import Path

R = Path('/tmp/pangram-hparam-sweep-20261006'); W = Path('/data/workspace'); log = open(R / 'setup-moe.log', 'a')
NAME = 'qwen36-35b-a3b'; MANIFEST = 'f24335dc851b73dd1d7916d8fdfca2062de002c5eb5cef5275dd349a87ea4f5d'
WOOG = W / 'overnight-sweep-20261004/h200/moe-A-full-lr1e4'


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush(); print(json.dumps(k), flush=True)


say(event='start')
src = W / 'backbone-launch-20261003/assets' / NAME; dst = R / 'assets' / NAME
if not dst.exists():
    shutil.copytree(src, dst.with_suffix('.partial'), ignore=shutil.ignore_patterns('xet')); dst.with_suffix('.partial').rename(dst)
for p in src.iterdir():
    if p.is_file():
        assert (dst / p.name).stat().st_size == p.stat().st_size, p.name
say(event='assets_copied', files=len(list(dst.iterdir())))

P0 = W / 'backbone-fast10-20261003/h200' / NAME / 'prepared-v2'; base = R / 'runs' / NAME / 'prepared-v2'
assert hashlib.sha256((P0 / 'manifest.json').read_bytes()).hexdigest() == MANIFEST
if not base.exists():
    shutil.copytree(P0, base)
man = json.loads((base / 'manifest.json').read_text())
for k, v in man['files'].items():
    assert hashlib.sha256(gzip.decompress((base / f'{k}.jsonl.gz').read_bytes())).hexdigest() == v['sha256'], k
say(event='prepared_copied', counts=man['counts'].get('stage2-epoch0'))

models = json.loads((R / 'models.json').read_text())
spec = next(m for m in json.loads((W / 'backbone-launch-20261003/models.json').read_text()) if m['name'] == NAME)
spec = {**spec, 'lora_rank': 128, 'expert_rank': 16}
(R / 'models.json').write_text(json.dumps([m for m in models if m['name'] != NAME] + [spec], indent=1))

ev = R / 'sweeps' / 'moe-A-full-lr1e4-woog'; ev.mkdir(parents=True, exist_ok=True)
run = json.loads((WOOG / 'run.json').read_text()); assert run['data_manifest_sha256'] == MANIFEST
run['assets'] = str(dst); run['copied_from'] = str(WOOG)
(ev / 'run.json').write_text(json.dumps(run, indent=1))
for p in sorted(WOOG.glob('stage2-epoch*-adapters.safetensors')):
    if not (ev / p.name).exists():
        (ev / p.name).symlink_to(p)
say(event='woog_lr1e4_staged', checkpoints=sorted(p.name for p in ev.glob('*.safetensors')), lora=run['config']['lora'])
say(event='done')
