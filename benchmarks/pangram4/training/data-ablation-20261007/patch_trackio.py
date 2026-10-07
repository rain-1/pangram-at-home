"""Patch woog's train_sweep.py (T2.1 version) to log to Trackio instead of W&B; nothing else changes."""
from pathlib import Path
p = Path('train_sweep.py'); s = p.read_text()
start = s.index("    wr = _NoWandb()\n    if is0:\n        import wandb")
end = s.index("    selection = rows(root, 'selection-windows'")
s = s[:start] + "    wr = _NoWandb()\n    if is0:\n        wr = _Trackio(a, cfg, out)\n" + s[end:]
s = s.replace("class _NoWandb:", '''class _Trackio:
    """W&B-shaped wrapper around trackio: log(), summary[...] and finish(exit_code=...)."""

    def __init__(self, a, cfg, out):
        import trackio
        from runtime import TRACKIO_PROJECT
        self.t = trackio; self.summary = _Summary(self); self.url = None; self.id = a.tag
        trackio.init(project=TRACKIO_PROJECT, name=a.tag, group=a.tag.rsplit('-s', 1)[0], config={**cfg, 'tag': a.tag, 'model': a.model}, resume='allow', embed=False)
        save(out / 'trackio-tracking.json', {'project': TRACKIO_PROJECT, 'run': a.tag})

    def log(self, d):
        try:
            self.t.log({k: v for k, v in d.items() if v is not None})
        except Exception as e:  # tracking must never interrupt training
            print(json.dumps({'trackio_error': repr(e)}), flush=True)

    def finish(self, exit_code=0):
        self.log({'summary/exit_code': exit_code}); self.t.finish()


class _Summary(dict):
    def __init__(self, w):
        super().__init__(); self.w = w

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if isinstance(v, (int, float)):
            self.w.log({f'summary/{k}': v})


class _NoWandb:''', 1)
p.write_text(s)
q = Path('queue_runner.py'); t = q.read_text()
t = t.replace("if not env.get('WANDB_API_KEY'):\n    say(event='abort', reason='WANDB_API_KEY missing'); sys.exit(1)\n", "env.setdefault('TRACKIO_DIR', str(R / 'trackio'))\n")
q.write_text(t)
print('patched', 'wandb.init' not in s, 'WANDB_API_KEY missing' not in t)
