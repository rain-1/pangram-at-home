"""Hyperparameter sweep trainer (overnight-sweep-20261004/train_sweep.py plus LoRA, optimiser, batch and loss-weight knobs).

Writes ROOT/sweeps/<tag>/ and logs to Trackio (local database under ROOT/trackio, synced to a Hub Space off-host)."""
from runtime import ROOT, TRACKIO_PROJECT, require_space
from pathlib import Path
import os
import argparse, json, gzip, hashlib, random, shutil, time, math, traceback
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


import typo_aug
TYPO_AUG = False  # set by --typo-aug
SHORT_SPAN_OVERSAMPLE = False  # set by --short-span-oversample
TOK = None  # set in main; used to drop the few rows this tokenizer version puts over the 510-token budget
REDUCE_BUCKET = 64 * 2**20  # FP32 elements per gradient all-reduce (256 MiB)
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
    if SHORT_SPAN_OVERSAMPLE and key.startswith('stage2'):
        short = oversample_short_spans(short, random.Random(seed * 7919 + sum(map(ord, key))))
    if TOK is not None:
        # Windows were cut to 510 tokens with the Qwen tokenizer. Under another tokenizer some exceed the budget (Nemotron: 6-16%
        # of rows), so dropping them would bias the data. Instead each overlong row is re-cut to 510 of this tokenizer's tokens,
        # centred on its target span; regions and offsets are clipped. Document-only rows are chunked by the encoder and kept.
        n = [len(x) for x in TOK([r['text'] for r in short], add_special_tokens=False)['input_ids']]
        cut = 0
        for i, (r, k) in enumerate(zip(short, n)):
            if r.get('supervision') != 'document_only' and k > 510:
                short[i] = fit_tokens(r, TOK); cut += 1
        if cut:
            print(json.dumps({'event': 'recut_overlong_rows', 'file': key, 'recut': cut}), flush=True)
    if TYPO_AUG:
        short = [typo_aug.render(r, seed) for r in short]
    random.Random(seed * 1000 + sum(map(ord, key))).shuffle(short)
    return short


def fit_tokens(row, tok, limit=510):
    """Re-cut a row to at most `limit` tokens of `tok`, centred on its target span, keeping every other field."""
    text = row['text']; off = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    ts, te = row.get('target_start', 0), row.get('target_end', len(text)); centre = (ts + te) // 2
    ci = next((i for i, (_, b) in enumerate(off) if b > centre), len(off) - 1)
    s = max(0, min(len(off) - limit, ci - limit // 2)); e = s + limit
    a = off[s][0]
    while e > s and len(tok(text[a:off[e - 1][1]], add_special_tokens=False)['input_ids']) > limit:
        e -= 1
    b = off[e - 1][1]; out = dict(row); out['text'] = text[a:b]
    out['regions'] = [{**g, 'start': max(g['start'], a) - a, 'end': min(g['end'], b) - a} for g in row['regions'] if min(g['end'], b) > max(g['start'], a)]
    out['target_start'] = max(0, min(b - a, ts - a)); out['target_end'] = max(0, min(b - a, te - a))
    if 'source_start' in row:
        out['source_start'] = row['source_start'] + a; out['source_end'] = row['source_start'] + b
    return out


def ai_span_chars(row):
    regions = row.get('regions') or []
    ai = [r['end'] - r['start'] for r in regions if r['label'] == 1]
    human = sum(r['end'] - r['start'] for r in regions if r['label'] == 0)
    return min(ai) if ai and human else None


def oversample_short_spans(rows, rng):
    """Swap paper windows whose AI span is >=600 chars for copies of existing windows with shorter spans.

    Same row count and dataset mix; no new text is created."""
    papers = [i for i, r in enumerate(rows) if r['dataset'] == 'papers' and r.get('supervision') != 'document_only']
    short = [rows[i] for i in papers if (ai_span_chars(rows[i]) or 10**9) < 600]
    long_ = [i for i in papers if (ai_span_chars(rows[i]) or 0) >= 600]
    if not short:
        return rows
    out = list(rows)
    for i in long_:
        out[i] = dict(rng.choice(short)); out[i]['oversampled_short_span'] = True
    return out


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


class DivergenceGuard:
    """Flags a training-loss blow-up within ~20-30 steps of its onset.

    Per stage: exponential moving average of step loss (half-life 50 steps); after 150 steps it tracks the
    lowest EMA seen and trips once the EMA stays above 2x that minimum + 0.1 for 30 consecutive steps.
    Replayed on every H200 sweep log (Oct 4-5): it caught all four runs that blew up (moe-A-full, curve-4b-B,
    curve-9b-B, curve-9b-B2) and raised no alarm on the ~30 healthy ones.
    """
    HALF_LIFE, ARM, RATIO, ADD, PATIENCE = 50, 150, 2., .1, 30

    def __init__(self):
        self.stage = None

    def update(self, stage, loss):
        if stage != self.stage:
            self.stage, self.ema, self.min, self.n, self.over = stage, None, None, 0, 0
        a = 1 - .5 ** (1 / self.HALF_LIFE); self.n += 1
        self.ema = loss if self.ema is None else self.ema + a * (loss - self.ema)
        if self.n < self.ARM:
            return False
        self.min = self.ema if self.min is None else min(self.min, self.ema)
        self.over = self.over + 1 if self.ema > self.RATIO * self.min + self.ADD else 0
        return self.over >= self.PATIENCE

    def state(self):
        return {k: getattr(self, k, None) for k in ['stage', 'ema', 'min', 'n', 'over']}

    def load(self, s):
        for k, v in (s or {}).items():
            setattr(self, k, v)


class _Trackio:
    """W&B-shaped wrapper around trackio: log(), summary[...] and finish(exit_code=...)."""

    def __init__(self, a, cfg, out):
        import trackio
        self.t = trackio; self.summary = _Summary(self); self.url = None; self.id = a.tag
        self.run = trackio.init(project=TRACKIO_PROJECT, name=a.tag, group=a.arm or a.tag.rsplit('-s', 1)[0],
                                config={**cfg, 'tag': a.tag, 'arm': a.arm, 'model': a.model}, resume='allow', embed=False)
        save(out / 'trackio-tracking.json', {'project': TRACKIO_PROJECT, 'run': a.tag, 'dir': os.environ['TRACKIO_DIR']})

    def log(self, d):
        try:
            self.t.log({k: v for k, v in d.items() if v is not None})
        except Exception as e:  # tracking must never interrupt training
            print(json.dumps({'trackio_error': repr(e)}), flush=True)

    def finish(self, exit_code=0):
        self.t.log({'summary/exit_code': exit_code}); self.t.finish()


class _Summary(dict):
    def __init__(self, w):
        super().__init__(); self.w = w

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if isinstance(v, (int, float)):
            self.w.log({f'summary/{k}': v})


class _NoWandb:
    summary = {}
    url = None
    id = None

    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


def main(a):
    require_space(); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    import torch.distributed as dist
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0')); local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group('nccl')
    is0 = rank == 0
    put = save if is0 else (lambda *x, **k: None)
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    spec = next(m for m in json.loads((ROOT / 'models.json').read_text()) if m['name'] == a.model)
    root = ROOT / 'runs' / a.model; out = ROOT / 'sweeps' / a.tag; out.mkdir(parents=True, exist_ok=True)
    assert a.resume or not (out / 'run.json').exists(), 'Existing sweep run: do not restart (use --resume)'
    global SHORT_SPAN_OVERSAMPLE
    SHORT_SPAN_OVERSAMPLE = a.short_span_oversample
    global TYPO_AUG
    TYPO_AUG = a.typo_aug
    modeling_sweep.W.update(sentence=a.sentence_weight, segment=a.segment_weight, mixed=a.mixed_weight, document=a.document_weight)
    manifest = json.loads((root / 'prepared-v2/manifest.json').read_text())
    total_rows = sum(round(v * a.fraction) for k in (['stage1-epoch0'] if a.start_stage == 1 else []) + ['stage2-epoch0', 'stage2-epoch1', 'stage2-epoch2'] for v in manifest['counts'][k].values())
    EB = a.effective_batch; size = min(a.micro_batch, EB); A = EB // size
    assert EB % size == 0 and A % world == 0, f'effective batch {EB} must split evenly into micro-batches across GPUs'
    cfg = {'kind': spec['kind'], 'training_fraction': a.fraction, 'total_training_examples': total_rows, 'seed': a.seed,
           'learning_rate': a.lr, 'head_learning_rate': a.head_lr, 'schedule': a.schedule, 'warmup_fraction': a.warmup,
           'micro_batch': size, 'physical_microbatch': size, 'effective_batch': EB, 'data_parallel_gpus': world,
           'stages': {'1': {'epochs': 1}, '2': {'epochs': 3}}, 'loss_weights': dict(modeling_sweep.W),
           'lora': {'rank': a.lora_rank, 'expert_rank': a.expert_rank or spec.get('expert_rank', a.lora_rank), 'alpha': a.lora_alpha, 'dropout': a.lora_dropout}, 'weight_decay': a.weight_decay,
           'checkpoint_every_steps': a.ckpt_every, 'short_span_oversample': a.short_span_oversample, 'typo_aug': a.typo_aug, 'source_tokens_max': 510, 'precision': 'BF16',
           'trainable_dtype': 'bfloat16', 'checkpoint_dtype': 'bfloat16', 'optimizer': a.optimizer}
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    asset = ROOT / 'assets' / a.model; tok = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    typo_aug.TOK = tok
    global TOK
    TOK = tok
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    put(out / 'status.json', {'state': 'loading_model'})
    model = Detector.load_base({'repo': str(asset), 'revision': None}, cfg['kind']); coverage = attach_lora(model, cfg)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.requires_grad)
    model = model.cuda(); model.physical_microbatch = size
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model.backbone, 'enable_input_require_grads'):
        model.backbone.enable_input_require_grads()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]

    def sync_params():
        if world > 1:
            with torch.no_grad():
                for p_ in trainable:
                    dist.broadcast(p_.data, 0)

    def reduce_grads():
        """Sum micro-batch gradients across GPUs in FP32; parameters unused on a rank (idle experts) contribute zeros."""
        if world == 1:
            return
        # Bucketed so the FP32 staging copy stays near REDUCE_BUCKET elements instead of every trainable gradient at once;
        # one flat buffer pushed the 35B-A3B MoE to 78 GiB on two GPUs regardless of micro-batch.
        bucket = []; size_ = 0
        for i_, p_ in enumerate(trainable):
            bucket.append(p_); size_ += p_.numel()
            if size_ >= REDUCE_BUCKET or i_ == len(trainable) - 1:
                flat = torch.cat([(q_.grad if q_.grad is not None else torch.zeros_like(q_)).float().reshape(-1) for q_ in bucket])
                dist.all_reduce(flat); o = 0
                for q_ in bucket:
                    n_ = q_.numel(); q_.grad = flat[o:o + n_].view_as(q_).to(q_.dtype); o += n_
                del flat; bucket = []; size_ = 0

    def total(*vals):
        t = torch.tensor(vals, dtype=torch.float64, device='cuda')
        if world > 1:
            dist.all_reduce(t)
        return t.tolist()

    sync_params()
    put(out / 'run.json', {'config': cfg, 'model': spec, 'assets': str(asset), 'tag': a.tag,
                           'data_manifest_sha256': hashlib.sha256((root / 'prepared-v2/manifest.json').read_bytes()).hexdigest(),
                           'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}})
    wr = _NoWandb()
    if is0:
        wr = _Trackio(a, cfg, out)
    selection = rows(root, 'selection-windows', manifest, a.fraction, a.seed)[::3]  # fixed third keeps validation cheap

    @torch.inference_mode()
    def validate_shared(stage):
        model.eval(); tot = 0.; cnt = 0
        for b in batches(selection[rank::world], tok, model.kind, stage, size):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o_ = model(b); v_, _ = loss(o_, b, stage)
            assert torch.isfinite(v_); tot += float(v_) * b['logical_size']; cnt += b['logical_size']
        tot, cnt = total(tot, cnt)
        return tot / cnt

    history = []; started = time.time(); global_step = 0; processed = 0; done_rows = 0; bench = {}; step_losses = []
    sd = Path(a.state_dir) / a.tag if a.state_dir else out  # resume state can live off the quota-limited disk
    sd.mkdir(parents=True, exist_ok=True)
    rp = sd / 'resume-state.pt'; state = None; last_save = time.time()
    guard = DivergenceGuard(); snaps = sd / 'snapshots'  # rolling adapter snapshots for rollback after a blow-up
    skip = set(); lr_scale = 1.; rollbacks = 0; steplog = []; snap_steps = []
    full_snaps = a.on_diverge == 'rollback'  # rollback needs optimizer state and data position, not just adapters

    def apply_state(st):
        nonlocal history, global_step, processed, done_rows, started, skip, lr_scale, rollbacks
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                if n_ in st['trainable']:
                    p_.copy_(st['trainable'][n_].to(p_.device, p_.dtype))
        history = st['history']; global_step = st['global_step']; processed = st['processed']; done_rows = st['done_rows']
        started = time.time() - st['elapsed']; guard.load(st.get('guard')); guard.over = 0
        skip = {tuple(x) for x in st.get('skip', [])}; lr_scale = st.get('lr_scale', 1.); rollbacks = st.get('rollbacks', 0)
        random.setstate(st['rng']['python']); np.random.set_state(st['rng']['numpy']); torch.set_rng_state(st['rng']['torch'])

    if a.init_adapters and not (a.resume and rp.exists()):
        # Start from another run's adapters (e.g. a finished stage 1 when only stage-2 data differs between arms).
        restore(model, Path(a.init_adapters)); sync_params()
        put(out / 'init-adapters.json', {'path': a.init_adapters, 'sha256': hashlib.sha256(Path(a.init_adapters).read_bytes()).hexdigest(), 'start_stage': a.start_stage})
    if a.resume and rp.exists():
        state = torch.load(rp, map_location='cpu', weights_only=False); apply_state(state)
        if snaps.exists():
            snap_steps = sorted(int(p_.name[4:9]) for p_ in snaps.glob('step*.pt'))
        if is0:
            print(json.dumps({'event': 'resumed', 'stage': state['stage'], 'epoch': state['epoch'], 'batch': state['batch'], 'optimizer_steps': global_step}), flush=True)

    def save_resume(stage, epoch, batch, running, best, bestpath, opt, sched, path=None):
        if not is0:
            return
        path = path or rp
        snap = {'trainable': {n_: p_.detach().to('cpu', torch.bfloat16) for n_, p_ in model.named_parameters() if p_.requires_grad},
                'opt': opt.state_dict(), 'sched': sched.state_dict(), 'stage': stage, 'epoch': epoch, 'batch': batch, 'running': running,
                'best': best, 'bestpath': str(bestpath) if bestpath else None, 'history': history, 'global_step': global_step,
                'processed': processed, 'done_rows': done_rows, 'elapsed': time.time() - started, 'guard': guard.state(),
                'skip': sorted(skip), 'lr_scale': lr_scale, 'rollbacks': rollbacks,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}}
        tmp = path.with_suffix('.tmp'); torch.save(snap, tmp); tmp.replace(path)

    class _Stop(Exception):
        pass

    class _Diverged(Exception):
        pass

    class _Rollback(Exception):
        pass
    try:
        while True:
            try:
                for stage in ([2] if a.start_stage == 2 else [1, 2]):
                    if state is not None and stage < state['stage']:
                        continue
                    epochs = cfg['stages'][str(stage)]['epochs']; best = float('inf'); bestpath = None
                    opt = (bnb.optim.AdamW8bit(parameter_groups(model, cfg), weight_decay=a.weight_decay) if a.optimizer == 'adamw8bit'
                           else torch.optim.AdamW(parameter_groups(model, cfg), weight_decay=a.weight_decay))
                    total_steps = sum(math.ceil(round(manifest['files'][f'stage{stage}-epoch{e}']['rows'] * a.fraction) / EB) for e in range(epochs))
                    warm = math.ceil(a.warmup * total_steps)
                    sched = (get_cosine_schedule_with_warmup(opt, warm, total_steps) if a.schedule == 'cosine'
                             else get_constant_schedule_with_warmup(opt, warm))
                    base0 = list(sched.base_lrs)
                    if state is not None and stage == state['stage']:
                        opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
                        best = state['best']; bestpath = Path(state['bestpath']) if state['bestpath'] else None
                    sched.base_lrs = [b_ * lr_scale for b_ in base0]  # learning rate cut after repeated rollbacks
                    for g_, b_, f_ in zip(opt.param_groups, sched.base_lrs, sched.lr_lambdas):
                        g_['lr'] = b_ * f_(sched.last_epoch)
                    stage_t0 = time.time(); stage_rows0 = done_rows
                    for epoch in range(epochs):
                        if state is not None and stage == state['stage'] and epoch < state['epoch']:
                            continue
                        start_i, running = 0, 0
                        if state is not None:
                            start_i, running = state['batch'], state['running']; state = None
                        rs = rows(root, f'stage{stage}-epoch{epoch}', manifest, a.fraction, a.seed)
                        model.train(); opt.zero_grad(set_to_none=True); n = math.ceil(len(rs) / size); epochstart = time.time()
                        for g0 in range(start_i, n, A):
                            if (stage, epoch, g0) in skip:  # batches dropped by an auto-rollback
                                continue
                            members = list(range(g0, min(g0 + A, n))); gn = len(members); loss_sum = 0.; toks = 0
                            group = rs[g0 * size:(g0 + gn) * size]
                            if world > 1 and a.balance:
                                # Deal the group's rows to GPUs by estimated cost (snake order), equal row counts per GPU.
                                order = sorted(range(len(group)), key=lambda j: -len(group[j]['text'])); bins = [[] for _ in range(world)]
                                for pos, j in enumerate(order):
                                    r_ = pos % (2 * world); bins[r_ if r_ < world else 2 * world - 1 - r_].append(group[j])
                                mine = bins[rank]; micro = [mine[j:j + size] for j in range(0, len(mine), size)]
                            else:
                                micro = [rs[k * size:(k + 1) * size] for k in members if (k - g0) % world == rank]
                            for mb in micro:
                                if not mb:
                                    continue
                                b = collate([encode_example(r, tok, cfg['kind'], stage) for r in mb], tok.pad_token_id)
                                with torch.autocast('cuda', dtype=torch.bfloat16):
                                    pred = model(b); v, parts = training_loss(pred, b, stage)
                                if not torch.isfinite(v):
                                    raise RuntimeError('Nonfinite training loss')
                                w_ = len(mb) / len(group)  # equals 1/gn when micro-batches are full: same objective as single-GPU accumulation
                                (v * w_).backward(); loss_sum += float(v.detach()) * w_ * gn; toks += int(b['attention_mask'].sum())
                                del b, pred, v
                            reduce_grads()
                            loss_sum, toks = total(loss_sum, toks)
                            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.)); lr_now = max(g_['lr'] for g_ in opt.param_groups)  # rate this step actually uses
                            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
                            step_losses.append(round(loss_sum / gn, 6)); running += loss_sum; processed += int(toks); done_rows += len(rs[g0 * size:(g0 + gn) * size])
                            tripped = guard.update(stage, loss_sum / gn)  # identical on every rank: the loss is all-reduced
                            steplog.append((global_step, stage, epoch, g0, guard.ema)); del steplog[:-4000]
                            if a.snapshot_every and global_step % a.snapshot_every == 0 and not tripped and (not full_snaps or g0 + A < n):
                                snaps.mkdir(exist_ok=True)
                                if full_snaps:
                                    save_resume(stage, epoch, g0 + A, running, best, bestpath, opt, sched, path=snaps / f'step{global_step:05d}.pt')
                                    snap_steps.append(global_step)
                                    for old in snap_steps[:-a.snapshot_keep]:
                                        if is0:
                                            (snaps / f'step{old:05d}.pt').unlink(missing_ok=True)
                                    del snap_steps[:-a.snapshot_keep]
                                elif is0:
                                    checkpoint(model, snaps / f'step{global_step:05d}-adapters.safetensors')
                                    for old in sorted(snaps.glob('step*-adapters.safetensors'))[:-a.snapshot_keep]:
                                        old.unlink()
                            if tripped and a.on_diverge == 'rollback' and rollbacks < a.max_rollbacks:
                                # Onset = first step after the loss EMA last sat near its minimum; drop a small window of
                                # batches around it and replay from the newest snapshot taken before that window.
                                thr = 1.25 * guard.min + .02; cur = [e for e in steplog if e[1] == stage]; onset = cur[0][0]
                                for e in reversed(cur):
                                    if e[4] is not None and e[4] <= thr:
                                        onset = e[0] + 1; break
                                onset = min(onset, global_step); lo_, hi_ = onset - a.skip_before, onset + a.skip_after
                                usable = [s_ for s_ in snap_steps if s_ < lo_]
                                if usable:
                                    target = usable[-1]; new_skip = {(e[1], e[2], e[3]) for e in steplog if lo_ <= e[0] < hi_}
                                    new_scale = lr_scale * (a.rollback_lr_factor if rollbacks >= 1 else 1.)
                                    info = {'event': 'rollback', 'rollback': rollbacks + 1, 'tripped_at_step': global_step, 'onset_step': onset,
                                            'skipped_steps': [lo_, hi_], 'skipped_batches': len(new_skip), 'restored_snapshot_step': target,
                                            'lr_scale': new_scale, 'loss_ema': round(guard.ema, 4), 'loss_ema_min': round(guard.min, 4)}
                                    if is0:
                                        print(json.dumps(info), flush=True)
                                        log_ = json.loads((out / 'rollbacks.json').read_text()) if (out / 'rollbacks.json').exists() else []
                                        save(out / 'rollbacks.json', log_ + [info]); wr.summary['rollbacks'] = rollbacks + 1
                                        if hasattr(wr, 'alert'):
                                            wr.alert(title=f'{a.tag} rolled back', text=json.dumps(info)[:900])
                                        for s_ in snap_steps:
                                            if s_ > target:
                                                (snaps / f'step{s_:05d}.pt').unlink(missing_ok=True)
                                    if world > 1:
                                        dist.barrier()
                                    st = torch.load(snaps / f'step{target:05d}.pt', map_location='cpu', weights_only=False)
                                    keep_skip = skip | new_skip; n_rb = rollbacks + 1
                                    apply_state(st); skip = keep_skip; lr_scale = new_scale; rollbacks = n_rb
                                    steplog[:] = [e for e in steplog if e[0] <= target]; snap_steps[:] = [s_ for s_ in snap_steps if s_ <= target]
                                    state = st; sync_params()
                                    raise _Rollback()
                            if tripped:
                                info = {'event': 'diverged', 'stage': stage, 'epoch': epoch, 'batch': g0 + gn, 'optimizer_steps': global_step,
                                        'loss_ema': round(guard.ema, 4), 'loss_ema_min': round(guard.min, 4), 'learning_rate': lr_now,
                                        'rollbacks': rollbacks, 'snapshots': sorted(p.name for p in snaps.glob('step*')), 'snapshot_dir': str(snaps)}
                                if is0:
                                    print(json.dumps(info), flush=True); save(out / 'divergence.json', info)
                                    wr.summary['diverged_at_step'] = global_step
                                    if hasattr(wr, 'alert'):
                                        wr.alert(title=f'{a.tag} diverged', text=json.dumps(info)[:900])
                                if a.on_diverge != 'warn':  # stop, or rollback with no usable snapshot / rollbacks used up
                                    raise _Diverged()
                                guard.over = -10 ** 9  # warn mode: report once per stage, keep training
                            if a.resume_every and time.time() - last_save > a.resume_every * 60 and g0 + A < n:
                                save_resume(stage, epoch, g0 + A, running, best, bestpath, opt, sched); last_save = time.time()
                            if a.ckpt_every and stage == 2 and global_step % a.ckpt_every == 0 and is0:
                                checkpoint(model, out / f'step{global_step:05d}-adapters.safetensors')
                                save(out / 'step-checkpoints.json', sorted(p.name for p in out.glob('step*-adapters.safetensors')))
                            if is0:
                                status = {'state': 'training', 'stage': stage, 'epoch': epoch, 'batch': g0 + gn, 'batches': n, 'loss': loss_sum / gn,
                                          'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'logical_rows': done_rows,
                                          'total_rows': total_rows, 'processed_tokens': processed, 'gpus': world,
                                          'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                          'grad_norm': round(grad_norm, 4), 'learning_rate': lr_now,
                                          'loss_ema': guard.ema and round(guard.ema, 4), 'loss_ema_min': guard.min and round(guard.min, 4),
                                          'rollbacks': rollbacks, 'lr_scale': lr_scale, 'skipped_batches': len(skip)}
                                save(out / 'status.json', status)
                                if global_step % 8 == 0:
                                    print(json.dumps(status), flush=True)
                                wr.log({'train/loss': status['loss'], 'progress/stage': stage, 'progress/epoch': epoch + 1,
                                        'progress/percent': 100 * done_rows / total_rows, 'progress/optimizer_step': global_step,
                                        'train/elapsed_seconds': status['elapsed_seconds'], 'train/grad_norm': grad_norm,
                                        'train/learning_rate': lr_now, 'train/loss_ema': guard.ema, 'train/rollbacks': rollbacks,
                                        'train/tokens_per_second': processed / max(1, status['elapsed_seconds'])})
                            if a.max_steps and global_step >= a.max_steps:
                                bench[f'stage{stage}'] = {'rows': done_rows - stage_rows0, 'seconds': time.time() - stage_t0}
                                raise _Stop()
                        sync_params()
                        score = validate_shared(stage) if not a.no_validate else float('nan')
                        rec = {'stage': stage, 'epoch': epoch, 'train_loss': running / n, 'selection_loss': score,
                               'epoch_seconds': time.time() - epochstart, 'optimizer_steps': global_step}
                        history.append(rec); put(out / 'history.json', history)
                        wr.log({'epoch/index': len(history), 'epoch/train_loss': rec['train_loss'], 'epoch/validation_loss': score, 'epoch/seconds': rec['epoch_seconds']})
                        cp = out / f'stage{stage}-epoch{epoch}-adapters.safetensors'
                        if (stage == 2 or SAVE_STAGE1) and is0:
                            checkpoint(model, cp)
                        if score < best or best == float('inf'):
                            best = score; bestpath = cp; put(out / f'stage{stage}-selection.json', {**rec, 'checkpoint': cp.name})
                    bench[f'stage{stage}'] = {'rows': done_rows - stage_rows0, 'seconds': time.time() - stage_t0}
                    if stage == 1 and epochs > 1:
                        restore(model, bestpath)
                    del opt, sched; torch.cuda.empty_cache()
                break
            except _Rollback:
                opt = sched = None; torch.cuda.empty_cache()
        if is0:
            rp.unlink(missing_ok=True); shutil.rmtree(snaps, ignore_errors=True)
        put(out / 'status.json', {'state': 'trained', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'processed_tokens': processed,
                                  'gpus': world, 'stage_throughput': bench, 'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        wr.summary['training_state'] = 'finished'; wr.log({'progress/percent': 100}); wr.finish()
    except _Diverged:
        # Clean exit (status 0) so wrappers do not auto-resume into the same blow-up; snapshots and resume state are kept.
        put(out / 'status.json', {'state': 'diverged', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step,
                                  'detail': 'divergence.json', 'gpus': world})
        wr.summary['training_state'] = 'diverged'; wr.finish(exit_code=0)
    except _Stop:
        put(out / 'status.json', {'state': 'benchmark_done', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'step_losses': step_losses,
                                  'gpus': world, 'stage_throughput': bench, 'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        wr.finish()
    except Exception as e:
        put(out / 'status.json', {'state': 'failed', 'error': repr(e), 'elapsed_seconds': time.time() - started})
        wr.summary['training_state'] = 'failed'; wr.finish(exit_code=1); raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('tag')
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--head-lr', type=float, default=2e-5)
    p.add_argument('--schedule', choices=['cosine', 'constant'], default='cosine'); p.add_argument('--warmup', type=float, default=.06)
    p.add_argument('--sentence-weight', type=float, default=.2); p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--fraction', type=float, default=.2); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-every', type=int, default=0)
    p.add_argument('--arm', default='')
    p.add_argument('--init-adapters', default='', help='adapter safetensors to start from (must match this LoRA config)')
    p.add_argument('--start-stage', type=int, choices=[1, 2], default=1, help='2 = skip stage 1 (use with --init-adapters from a finished stage 1)')
    p.add_argument('--lora-rank', type=int, default=128); p.add_argument('--lora-alpha', type=float, default=32)
    p.add_argument('--expert-rank', type=int, default=0, help='MoE expert LoRA rank; 0 = models.json value (16 for qwen36-35b-a3b)')
    p.add_argument('--lora-dropout', type=float, default=0.); p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--effective-batch', type=int, default=32)
    p.add_argument('--segment-weight', type=float, default=.2); p.add_argument('--mixed-weight', type=float, default=.1)
    p.add_argument('--document-weight', type=float, default=.1)
    p.add_argument('--optimizer', choices=['adamw8bit', 'adamw'], default='adamw8bit')
    p.add_argument('--short-span-oversample', action='store_true')
    p.add_argument('--typo-aug', action='store_true', help='per-row quote-style/space randomisation (typo_aug.py)')
    p.add_argument('--resume', action='store_true'); p.add_argument('--resume-every', type=float, default=20.0)
    p.add_argument('--max-steps', type=int, default=0); p.add_argument('--no-validate', action='store_true')
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--state-dir', default='')
    p.add_argument('--snapshot-every', type=int, default=100, help='rolling adapter snapshots (in the state dir) every N optimizer steps; 0 disables')
    p.add_argument('--snapshot-keep', type=int, default=3)
    p.add_argument('--on-diverge', choices=['stop', 'warn', 'rollback'], default='stop',
                   help='rollback: restore the newest snapshot before the blow-up, skip the batches around its onset and continue')
    p.add_argument('--max-rollbacks', type=int, default=3)
    p.add_argument('--skip-before', type=int, default=4, help='optimizer steps skipped before the detected onset')
    p.add_argument('--skip-after', type=int, default=12, help='optimizer steps skipped from the onset on')
    p.add_argument('--rollback-lr-factor', type=float, default=.5, help='learning-rate multiplier applied from the second rollback on')
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        d = ROOT / 'sweeps' / args.tag; d.mkdir(parents=True, exist_ok=True)
        save(d / 'status.json', {'state': 'failed', 'error': repr(e)}); traceback.print_exc(); raise
