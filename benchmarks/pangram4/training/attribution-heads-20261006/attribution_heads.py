"""Frozen-detector attribution heads on the workbench Qwen3.5-4B (port of rain1's train_attribution_heads_v1).

rain1's v1 finding (research/span-detection-20260928/reports/attribution_comparison_v1.md): linear heads on a frozen
AI-text detector backbone beat the same heads on the raw backbone (Arena 50-model top-1 48.7% vs 42.7%, Qwen3-1.7B).
This repeats it on the workbench detector: features are the mean final hidden state over the second Repeat2 copy of
each window (data.layout, stage 2), averaged over up to 8 windows per document; the heads are class-weighted linear
probes with early stopping on validation macro-F1.

Run from the sweep root so runtime/modeling/adapters resolve (it shares the hparam sweep's assets and vendor packages):
  python attribution_heads.py --init fast10|base|<sweep tag> --task arena|authors [--features-only]
Data: DATA/<task>/{train,val,test}.jsonl with {id, text, label}; prompt-disjoint (arena) and work-disjoint (authors).
"""
from runtime import ROOT, TRACKIO_PROJECT, require_space
import argparse, hashlib, json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from data import layout
from modeling import Detector
from adapters_short import attach_lora

DATA = Path('/data/workspace/attribution-heads-20261006/data')
OUT = ROOT / 'attribution'
FAST10 = Path('/data/workspace/backbone-fast10-20261003/runs/qwen35-4b/run')
SPLITS = ('train', 'val', 'test')
WIDTH, STRIDE, MAX_WINDOWS, SEED = 510, 256, 8, 42


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(task, split):
    return [json.loads(l) for l in open(DATA / task / f'{split}.jsonl')]


def window_starts(n):
    if n <= WIDTH:
        return [0]
    pts = sorted(set(list(range(0, n - WIDTH + 1, STRIDE)) + [n - WIDTH]))
    if len(pts) > MAX_WINDOWS:
        pts = [pts[i] for i in np.linspace(0, len(pts) - 1, MAX_WINDOWS).round().astype(int)]
    return pts


def load(init):
    """Frozen Detector: base weights only, the fast10 Atlas adapter, or a finished sweep run's final adapter."""
    assets = ROOT / 'assets/qwen35-4b'
    if init == 'base':
        cfg, ckpt = {'kind': 'causal', 'lora': {'rank': 128, 'alpha': 32}}, None
    elif init == 'fast10':
        cfg, ckpt = json.loads((FAST10 / 'run.json').read_text())['config'], FAST10 / 'stage2-epoch0-adapters.safetensors'
    else:
        d = ROOT / 'sweeps' / init; cfg = json.loads((d / 'run.json').read_text())['config']
        ckpt = sorted(d.glob('stage2-epoch*-adapters.safetensors'))[-1]
    model = Detector.load_base({'repo': str(assets), 'revision': None}, cfg['kind'])
    if ckpt:
        attach_lora(model, cfg); w = load_file(str(ckpt))
        assert set(w) == {n for n, p in model.named_parameters() if p.requires_grad}, 'adapter keys differ'
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in w:
                    p.copy_(w[n].to(p.dtype))
    tok = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    return model.to(dtype=torch.bfloat16, device='cuda').eval(), tok, (sha(ckpt) if ckpt else None)


@torch.inference_mode()
def feature(text, model, tok):
    ids = tok(text, add_special_tokens=False)['input_ids']
    if not ids:
        raise ValueError('empty text')
    vecs = []
    for s in window_starts(len(ids)):
        lay = layout(ids[s:s + WIDTH], [-100] * len(ids[s:s + WIDTH]), 'causal', 2)
        x = torch.tensor([lay['ids']], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.backbone(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
        vecs.append(h[lay['source_positions']].float().mean(0).cpu().numpy())
    return np.mean(vecs, 0).astype(np.float32)


def encode(task, init, model, tok):
    d = OUT / init / task; d.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        f = d / f'{split}_features.npy'
        if f.exists():
            continue
        rs = rows(task, split); t0 = time.time()
        m = np.stack([feature(r['text'], model, tok) for r in rs])
        np.save(f.with_suffix('.tmp.npy'), m); f.with_suffix('.tmp.npy').replace(f)
        print(json.dumps({'task': task, 'split': split, 'rows': len(rs), 'seconds': round(time.time() - t0)}), flush=True)


def score(y, logits, labels):
    pred = logits.argmax(1); top5 = np.argsort(-logits, 1)[:, :5]
    f1 = []
    for i in range(len(labels)):
        tp = np.sum((pred == i) & (y == i)); fp = np.sum((pred == i) & (y != i)); fn = np.sum((pred != i) & (y == i))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.)
    return {'rows': int(len(y)), 'accuracy': float(np.mean(pred == y)), 'top5_accuracy': float(np.mean([y[i] in top5[i] for i in range(len(y))])),
            'macro_f1': float(np.mean(f1)), 'balanced_accuracy': float(np.mean([np.mean(pred[y == i] == i) for i in range(len(labels)) if np.any(y == i)]))}


def fit(task, init, ckpt_sha):
    import trackio
    d = OUT / init / task; rs = {s: rows(task, s) for s in SPLITS}
    labels = sorted({r['label'] for s in SPLITS for r in rs[s]}); index = {l: i for i, l in enumerate(labels)}
    X = {s: np.load(d / f'{s}_features.npy') for s in SPLITS}
    mean, std = X['train'].mean(0), X['train'].std(0)
    xs = {s: torch.from_numpy(np.clip((X[s] - mean) / (std + 1e-4), -5, 5)).float().cuda() for s in SPLITS}
    ys = {s: torch.tensor([index[r['label']] for r in rs[s]], device='cuda') for s in SPLITS}
    counts = Counter(ys['train'].tolist()); weights = torch.tensor([1 / counts[i] for i in range(len(labels))], device='cuda')
    torch.manual_seed(SEED); head = torch.nn.Linear(xs['train'].shape[1], len(labels)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01); lossf = torch.nn.CrossEntropyLoss(weight=weights / weights.mean())
    gen = torch.Generator().manual_seed(SEED); batch = 256 if task == 'arena' else 64; best = None; wait = 0
    trackio.init(project=TRACKIO_PROJECT, name=f'attribution-{task}-{init}', group='attribution-heads', resume='allow', embed=False,
                 config={'task': task, 'init': init, 'adapter_sha256': ckpt_sha, 'labels': len(labels), 'frozen_backbone': True})
    for epoch in range(1, 151):
        head.train(); perm = torch.randperm(len(xs['train']), generator=gen).cuda(); tl = []
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]; opt.zero_grad(set_to_none=True); l = lossf(head(xs['train'][b]), ys['train'][b]); l.backward(); opt.step(); tl.append(float(l))
        head.eval()
        with torch.inference_mode():
            vl = head(xs['val']); vloss = float(lossf(vl, ys['val'])); vs = score(ys['val'].cpu().numpy(), vl.cpu().numpy(), labels)
        trackio.log({'attr/epoch': epoch, 'attr/train_loss': float(np.mean(tl)), 'attr/val_loss': vloss, 'attr/val_macro_f1': vs['macro_f1'], 'attr/val_accuracy': vs['accuracy']})
        key = (vs['macro_f1'], -vloss)
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().clone() for k, v in head.state_dict().items()}); wait = 0
        else:
            wait += 1
        if wait >= 20:
            break
    head.load_state_dict(best[2]); head.eval(); res = {}
    with torch.inference_mode():
        for s in ('val', 'test'):
            res[s] = score(ys[s].cpu().numpy(), head(xs[s]).cpu().numpy(), labels)
    save_file({k: v.contiguous() for k, v in best[2].items()}, str(d / 'linear_head.safetensors'))
    np.savez_compressed(d / 'normalization.npz', mean=mean, std=std)
    report = {'task': task, 'init': init, 'labels': labels, 'best_epoch': best[1], 'validation': res['val'], 'test': res['test'],
              'adapter_sha256': ckpt_sha, 'data_sha256': {s: sha(DATA / task / f'{s}.jsonl') for s in SPLITS},
              'feature_recipe': {'repeat2': True, 'source_tokens': WIDTH, 'stride': STRIDE, 'max_windows': MAX_WINDOWS,
                                 'pooling': 'mean second-copy final hidden state per window, then document mean'}}
    (d / 'report.json').write_text(json.dumps(report, indent=1))
    trackio.log({f'attr_test/{k}': v for k, v in res['test'].items() if k != 'rows'}); trackio.finish()
    print(json.dumps({'task': task, 'init': init, 'test': res['test']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--init', default='fast10'); p.add_argument('--task', choices=['arena', 'authors'], nargs='+', default=['arena', 'authors'])
    p.add_argument('--features-only', action='store_true'); a = p.parse_args()
    require_space(); torch.manual_seed(SEED); torch.set_num_threads(4)
    model, tok, ck = load(a.init)
    for t in a.task:
        encode(t, a.init, model, tok)
    del model; torch.cuda.empty_cache()
    if not a.features_only:
        for t in a.task:
            fit(t, a.init, ck)
