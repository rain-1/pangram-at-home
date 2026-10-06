# Qwen3.5-4B hyperparameter sweep (October 6, 2026)

A one-factor sweep of the hyperparameters that `overnight-sweep-20261004` held fixed. That sweep varied learning rate, schedule, sentence-loss weight, optimizer, short-span oversampling and length. With warmup and cosine decay, every learning rate from 1e-4 to 5e-4 reached about 0.77 all-edit recall, and small edits stayed at 0.21–0.27 (`../overnight-sweep-20261004/NOTES.md`).

Every arm is recipe A (LR 2e-4, head LR 2e-5, cosine, 6% warmup, 20% length, micro-batch 8) with one change. Recipe A fixes LoRA at rank 128 with alpha 32, an adapter scale of only 0.25. It also uses dropout 0, weight decay 0.01, effective batch 32, and loss weights sentence 0.2, segment 0.2, mixed 0.1, document 0.1.

| Wave | Arms |
|---|---|
| 1 · LoRA | `A-ref` (seed-1 anchor on this hardware); alpha 128 and 256; rank/alpha 16/32, 32/64 and 256/512 (scale 2); alpha 128 at LR 5e-5 (same effective step as A); dropout 0.05 |
| 2 · optimiser | weight decay 0 and 0.1; effective batch 16 and 64; head LR 2e-4 and 1e-3; warmup 0.15; rank/alpha 64/128 |
| 3 · loss weights | segment 0 and 1.0; mixed 0 and 0.5; document 0 and 0.5; sentence 0; token-only (all auxiliary weights 0) |

Arms are listed in `arms.json`; `build_queue.py --seeds 1` writes `sweeps/queue.json`. Combinations of the winners and repeat seeds follow once wave results are in.

## Code

The code is a copy of the overnight sweep, with only these changes:
- `train_sweep.py`: adds `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--weight-decay`, `--effective-batch`, `--segment-weight`, `--mixed-weight`, `--document-weight` and `--arm`. Trackio replaces W&B.
- `adapters_short.py`: LoRA dropout is read from the config (default 0).
- `sweep_eval.py`: logs headline dev/test metrics into the same Trackio run.
- `queue_runner.py`: no longer requires a W&B key.

`data.py`, `modeling.py`, `modeling_sweep.py` and the rest of the trainer match the code hashes recorded in `splice-sweep-20261006` runs.

## Running on the training Space

```sh
# copy this folder to /tmp/pangram-hparam-sweep-20261006 on the Space, then there:
python setup_space.py            # assets, prepared-v2 (hash-checked), vendor packages, Trackio, evaluation rows
python build_queue.py --seeds 1
for g in 0 1 2 3 4 5 6 7; do setsid nohup python -u queue_runner.py $g < /dev/null > sweeps/runner-gpu$g.out 2>&1 & done
```

Setup copies the qwen35-4b assets and `prepared-v2` from `backbone-launch-20261003` (manifest `166a4171…`, the same data as the splice sweep). It takes `peft`, `bitsandbytes` and `fla` from `classifications/vendor-fast10`, with Trackio appended last on `sys.path`.

It also rebuilds the overnight sweep's 3,511 evaluation rows with that sweep's `build_eval_set.py` (same seed). The inputs are the frozen suite in `evaluation-suite-v1-20261003`, hash-checked against `research/evaluation/space-suite-v1/package/manifest.json`; the rows' sha256 is `c2cff164…`. Like the splice sweep, these runs use the reference `causal_conv1d` fallback, so speed matches.

## Tracking

Runs log to Trackio project `pangram-hparam-sweep-20261006`, in a local SQLite database under `trackio/` on the Space. The Space has no Hub token, so `sync_trackio.sh` runs on a logged-in machine. It snapshots the database over `hf spaces ssh` and syncs it to the private Space `eac123/pangram-hparam-sweep-trackio` every five minutes.

## Metrics

`sweep_eval.py` reports recall at 1% FPR with the cutoff fitted on the same half it reports. That is fine for ranking arms but optimistic. For held-out operating points, use `calib_score.py` and `../overnight-sweep-20261004/calibrated_report.py`, which also reports the document any-highlight rate.
