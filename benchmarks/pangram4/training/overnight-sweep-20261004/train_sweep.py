"""Overnight sweep trainer: train_short.py recipe with command-line knobs. Writes ROOT/sweeps/<tag>/."""
from runtime import ROOT, require_space
import argparse, json, gzip, hashlib, random, time, math, traceback
from collections import Counter
import numpy as np
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from safetensors.torch import save_file, load_file
from data import encode_example
import modeling_sweep
from modeling_sweep import Detector, collate, loss, training_loss
from adapters_short import attach_lora, parameter_groups


SAVE_STAGE1 = not str(ROOT).startswith('/workspace/')  # H200 volume quota is tight; the single stage-1 checkpoint is never evaluated


def save(p, d):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)


def rows(root, key, manifest, fraction, seed):
    b = gzip.decompress((root / 'prepared-v2' / f'{key}.jsonl.gz').read_bytes())
    assert hashlib.sha256(b).hexdigest() == manifest['files'][key]['sha256']
    rs = [json.loads(l) for l in b.splitlines()]
    if not key.startswith('stage'):
        return rs
    target = {k: round(v * fraction) for k, v in manifest['counts'][key].items()}; seen = Counter(); short = []
    for row in rs:
        if seen[row['dataset']] < target[row['dataset']]:
            short.append(row); seen[row['dataset']] += 1
    assert dict(seen) == target
    random.Random(seed * 1000 + sum(map(ord, key))).shuffle(short)
    return short


def batches(rs, tok, kind, stage, size):
    for i in range(0, len(rs), size):
        yield collate([encode_example(r, tok, kind, stage) for r in rs[i:i + size]], tok.pad_token_id)


def checkpoint(model, path):
    tensors = {n: p.detach().to(device='cpu', dtype=torch.bfloat16).contiguous() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path.with_suffix('.tmp'); save_file(tensors, str(tmp)); tmp.replace(path)


def restore(model, path):
    d = load_file(str(path)); assert set(d) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in d:
                p.copy_(d[n])


@torch.inference_mode()
def validate(model, tok, rs, stage, size):
    model.eval(); total = 0; n = 0
    for b in batches(rs, tok, model.kind, stage, size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(b); v, _ = loss(o, b, stage)
        assert torch.isfinite(v); total += float(v) * b['logical_size']; n += b['logical_size']
    return total / n


def main(a):
    require_space(); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    spec = next(m for m in json.loads((ROOT / 'models.json').read_text()) if m['name'] == a.model)
    root = ROOT / 'runs' / a.model; out = ROOT / 'sweeps' / a.tag; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'run.json').exists(), 'Existing sweep run: do not restart'
    modeling_sweep.W['sentence'] = a.sentence_weight
    manifest = json.loads((root / 'prepared-v2/manifest.json').read_text())
    total_rows = sum(round(v * a.fraction) for k in ['stage1-epoch0', 'stage2-epoch0', 'stage2-epoch1', 'stage2-epoch2'] for v in manifest['counts'][k].values())
    cfg = {'kind': spec['kind'], 'training_fraction': a.fraction, 'total_training_examples': total_rows, 'seed': a.seed,
           'learning_rate': a.lr, 'head_learning_rate': a.head_lr, 'schedule': a.schedule, 'warmup_fraction': a.warmup,
           'micro_batch': a.micro_batch, 'physical_microbatch': a.micro_batch, 'effective_batch': 32,
           'stages': {'1': {'epochs': 1}, '2': {'epochs': 3}}, 'loss_weights': dict(modeling_sweep.W),
           'lora': {'rank': spec.get('lora_rank', 128), 'expert_rank': spec.get('expert_rank', 128), 'alpha': 32},
           'checkpoint_every_steps': a.ckpt_every, 'source_tokens_max': 510, 'precision': 'BF16',
           'trainable_dtype': 'bfloat16', 'checkpoint_dtype': 'bfloat16', 'optimizer': a.optimizer}
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    asset = ROOT / 'assets' / a.model; tok = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    save(out / 'status.json', {'state': 'loading_model'})
    model = Detector.load_base({'repo': str(asset), 'revision': None}, cfg['kind']); coverage = attach_lora(model, cfg)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.requires_grad)
    model = model.cuda(); model.physical_microbatch = a.micro_batch
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model.backbone, 'enable_input_require_grads'):
        model.backbone.enable_input_require_grads()
    save(out / 'run.json', {'config': cfg, 'model': spec, 'assets': str(asset), 'tag': a.tag,
                            'data_manifest_sha256': hashlib.sha256((root / 'prepared-v2/manifest.json').read_bytes()).hexdigest(),
                            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}})
    import wandb
    settings = wandb.Settings(console='off', disable_code=True, disable_git=True, save_code=False, disable_job_creation=True,
                              x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True)
    wr = wandb.init(entity='rigg-alice0', project='pangram-text-classifiers', group='text-classifiers',
                    name=f'{a.model}-{a.tag}-20261004', config={}, settings=settings, dir=str(out))
    save(out / 'wandb-tracking.json', {'run_id': wr.id, 'url': wr.url})
    selection = rows(root, 'selection-windows', manifest, a.fraction, a.seed)[::3]  # fixed third keeps validation cheap
    history = []; started = time.time(); global_step = 0; processed = 0; done_rows = 0
    try:
        for stage in [1, 2]:
            epochs = cfg['stages'][str(stage)]['epochs']; size = a.micro_batch; accum = 32 // size; best = float('inf')
            opt = (bnb.optim.AdamW8bit(parameter_groups(model, cfg), weight_decay=.01) if a.optimizer == 'adamw8bit'
                   else torch.optim.AdamW(parameter_groups(model, cfg), weight_decay=.01))
            total_steps = sum(math.ceil(round(manifest['files'][f'stage{stage}-epoch{e}']['rows'] * a.fraction) / 32) for e in range(epochs))
            warm = math.ceil(a.warmup * total_steps)
            sched = (get_cosine_schedule_with_warmup(opt, warm, total_steps) if a.schedule == 'cosine'
                     else get_constant_schedule_with_warmup(opt, warm))
            for epoch in range(epochs):
                rs = rows(root, f'stage{stage}-epoch{epoch}', manifest, a.fraction, a.seed)
                model.train(); opt.zero_grad(set_to_none=True); n = math.ceil(len(rs) / size); running = 0; epochstart = time.time()
                for i, b in enumerate(batches(rs, tok, cfg['kind'], stage, size)):
                    count = min(accum, n - (i // accum) * accum)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(b); v, parts = training_loss(pred, b, stage)
                    if not torch.isfinite(v):
                        raise RuntimeError('Nonfinite training loss')
                    (v / count).backward(); running += float(v.detach()); processed += int(b['attention_mask'].sum()); done_rows += b['logical_size']
                    if (i + 1) % accum == 0 or i + 1 == n:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
                        if a.ckpt_every and stage == 2 and global_step % a.ckpt_every == 0:
                            checkpoint(model, out / f'step{global_step:05d}-adapters.safetensors')
                            save(out / 'step-checkpoints.json', sorted(p.name for p in out.glob('step*-adapters.safetensors')))
                    if i % max(1, 32 // size) == 0:
                        status = {'state': 'training', 'stage': stage, 'epoch': epoch, 'batch': i, 'batches': n, 'loss': float(v.detach()),
                                  'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'logical_rows': done_rows,
                                  'total_rows': total_rows, 'processed_tokens': processed,
                                  'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                        save(out / 'status.json', status); print(json.dumps(status), flush=True)
                        wr.log({'train/loss': status['loss'], 'progress/stage': stage, 'progress/epoch': epoch + 1,
                                'progress/percent': 100 * done_rows / total_rows, 'progress/optimizer_step': global_step,
                                'train/elapsed_seconds': status['elapsed_seconds'],
                                'train/tokens_per_second': processed / max(1, status['elapsed_seconds'])})
                    del b, pred, v
                score = validate(model, tok, selection, stage, size)
                rec = {'stage': stage, 'epoch': epoch, 'train_loss': running / n, 'selection_loss': score,
                       'epoch_seconds': time.time() - epochstart, 'optimizer_steps': global_step}
                history.append(rec); save(out / 'history.json', history)
                wr.log({'epoch/index': len(history), 'epoch/train_loss': rec['train_loss'], 'epoch/validation_loss': score, 'epoch/seconds': rec['epoch_seconds']})
                cp = out / f'stage{stage}-epoch{epoch}-adapters.safetensors'
                if stage == 2 or SAVE_STAGE1:
                    checkpoint(model, cp)
                if score < best:
                    best = score; bestpath = cp; save(out / f'stage{stage}-selection.json', {**rec, 'checkpoint': cp.name})
            if stage == 1 and epochs > 1:
                restore(model, bestpath)
            del opt, sched; torch.cuda.empty_cache()
        save(out / 'status.json', {'state': 'trained', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'processed_tokens': processed, 'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        wr.summary['training_state'] = 'finished'; wr.log({'progress/percent': 100}); wr.finish()
    except Exception as e:
        save(out / 'status.json', {'state': 'failed', 'error': repr(e), 'elapsed_seconds': time.time() - started})
        wr.summary['training_state'] = 'failed'; wr.finish(exit_code=1); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('tag')
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--head-lr', type=float, default=2e-5)
    p.add_argument('--schedule', choices=['cosine', 'constant'], default='cosine'); p.add_argument('--warmup', type=float, default=.06)
    p.add_argument('--sentence-weight', type=float, default=.2); p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--fraction', type=float, default=.2); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-every', type=int, default=0)
    p.add_argument('--optimizer', choices=['adamw8bit', 'adamw'], default='adamw8bit')
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        d = ROOT / 'sweeps' / args.tag; d.mkdir(parents=True, exist_ok=True)
        save(d / 'status.json', {'state': 'failed', 'error': repr(e)}); traceback.print_exc(); raise
