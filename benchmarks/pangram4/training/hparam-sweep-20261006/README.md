# Qwen3.5-4B hyperparameter sweep (October 6, 2026)

A one-factor sweep of the hyperparameters that `overnight-sweep-20261004` held fixed. That sweep varied learning rate, schedule, sentence-loss weight, optimizer, short-span oversampling and length. With warmup and cosine decay, every learning rate from 1e-4 to 5e-4 reached about 0.77 all-edit recall, and small edits stayed at 0.21–0.27 (`../overnight-sweep-20261004/NOTES.md`).

Every arm is recipe A (LR 2e-4, head LR 2e-5, cosine, 6% warmup, 20% length, micro-batch 8) with one change. Recipe A fixes LoRA at rank 128 with alpha 32, an adapter scale of only 0.25. It also uses dropout 0, weight decay 0.01, effective batch 32, and loss weights sentence 0.2, segment 0.2, mixed 0.1, document 0.1.

| Wave | Arms |
|---|---|
| 1 · LoRA | `A-ref` (seed-1 anchor on this hardware); alpha 128 and 256; rank/alpha 16/32, 32/64 and 256/512 (scale 2); alpha 128 at LR 5e-5 (same effective step as A); dropout 0.05 |
| 2 · optimiser | weight decay 0 and 0.1; effective batch 16 and 64; head LR 2e-4 and 1e-3; warmup 0.15; rank/alpha 64/128 |
| 3 · loss weights | segment 0 and 1.0; mixed 0 and 0.5; document 0 and 0.5; sentence 0; token-only (all auxiliary weights 0) |

Arms are listed in `arms.json`; `build_queue.py --seeds 1` writes `sweeps/queue.json`. Combinations of the winners and repeat seeds follow once wave results are in.


**Terminology: "epoch" in this trainer means a data shard, not a pass.** Stage 1 is one warm-up shard (`stage1-epoch0`).
- It trains the segment and document heads on single-copy input.
- It forms a two-stage curriculum with stage 2.

Stage 2 reads three **different** files, `stage2-epoch{0,1,2}`.
- Each is a fresh draw with the same per-source mix, under one cosine schedule.
- Repetition across them depends on pool size: `papers` about 1.6×, `fullpapers` heavy (different crops of 377 papers), `human` and GRADTEX essentially none. rain1's additions never repeat.
- A checkpoint and validation pass are taken at each shard boundary.

Identifiers keep woog's names (file names, `stage2-epochN-adapters.safetensors`, the `epoch` field). Reports say "stage-2 shard N".

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

Runs log to Trackio project `pangram-hparam-sweep-20261006`, in a local SQLite database under `trackio/` on the Space. The Space has no Hub token, so `sync_trackio.sh` runs on a logged-in machine. It snapshots the database over `hf spaces ssh` and syncs it to the private Space `eac123/pangram-hparam-sweep-trackio` every ten minutes. The Gradio dashboard reads its bucket only at startup, so the script uploads the database to the Space's bucket and then restarts the Space. `trackio sync` itself would time out waiting for the live dashboard to catch up.

## Metrics

`sweep_eval.py` reports recall at 1% FPR with the cutoff fitted on the same half it reports. That is fine for ranking arms but optimistic. For held-out operating points, use `calib_score.py` and `../overnight-sweep-20261004/calibrated_report.py`, which also reports the document any-highlight rate.

## MoE run

`moe.json` and `launch_moe.py` (torchrun, 4×A100) train Qwen3.6-35B-A3B. The recipe is woog's `moe-A-full-lr1e4` with four changes, decided after reviewing her MoE runs in `/data/workspace/overnight-sweep-20261004/h200/` (an independent second review reached the same conclusions):

| Setting | `moe-A-full` | `moe-A-full-lr1e4` | This run | Why |
|---|---|---|---|---|
| LR | 2e-4 | 1e-4 | 1e-4 | At 2e-4 the full-length run diverged near step 950: training loss jumped from about 0.05 to 1.05 and never recovered, and test all-edit recall finished at 0.31. At 1e-4 it trained cleanly (loss 0.108 → 0.041 → 0.023). The 4B tolerated even 5e-4, so its LR tolerance does not carry over. |
| Warmup | 0.06 | 0.06 | 0.10 | The blow-up came just after peak LR, so a gentler ramp is the cheapest safety margin. |
| Micro-batch (effective 32) | 8 | 8 | 4 | Fits 80 GB A100s; the averaged gradient is the same. |
| Divergence handling | none | none | `--on-diverge rollback` (snapshots every 100 steps, at most 3 rollbacks, LR × 0.5 from the second) | `moe-A-full` ran on for about 2 hours after collapsing. |
| Checkpoint selection | lowest selection loss | lowest selection loss | best dev-half all-edit recall (`sweep_eval.prune` keeps it plus the final epoch) | Selection loss was lowest for the collapsed run. |

**Unchanged on purpose:**
- **LoRA rank 128, expert rank 16, alpha 32 (shared).** With 1/r scaling, a fixed alpha keeps the effective step similar across ranks. Equalising the expert scale (alpha 4 on rank 16) would shrink expert updates about 8×. Raising alpha hurt the 4B at epoch 0: all-edit recall 0.545 at alpha 256 against 0.761 for the reference. The MoE is already at the edge of stability, so its alpha × LR product is not raised.
- **Head LR, weight decay, dropout:** 2e-5, 0.01, 0.

A lower separate LR for the expert adapters was considered and not used: it needs an extra parameter group, and the gains are uncertain.

**Mechanics:** `reduce_grads` now all-reduces in 256 MiB buckets. The single flat FP32 buffer copied all ~1.1B trainable gradients and kept the 2-GPU runs at 78 GiB however small the micro-batch. `--expert-rank` defaults to the models.json value (16); the sweep copy had briefly overridden it with the general rank.

**Before launch:**
1. Score `sweeps/moe-A-full-lr1e4-woog` (staged by `setup_moe.py`), whose three epochs were never evaluated. If none beats the 20%-length `moe-A-s1` (test all-edit 0.815), set `--fraction` to 0.2–0.5 instead of a full-length run.
2. `python launch_moe.py --preflight`: peak memory must be about 76 GB or less, with the expected rank counts and a sane gradient norm.

**4B results that would change this.** Act only on a gain of at least 0.03 dev all-edit recall, or a gain on every subset:
- effective batch 64 wins or ties → adopt it;
- warmup 0.15 wins or ties → use 0.15;
- head LR 2e-4 wins → adopt it;
- higher alpha or LR wins → do not transfer to the MoE.

## Unattended chain and benchmark scoring

`pipeline.py <arm> --seeds 1 2 --fraction 0.5` runs one chain per arm on the Space. For each seed it:
1. trains through `launch_moe.py`, which then scores the checkpoints with `sweep_eval.py` on the arm's first GPU;
2. meanwhile scores the final checkpoint on rain1's aidet_eval benchmark-v3 with `benchmark/score_benchmark.py`, sharded over the arm's other three GPUs;
3. starts the next seed.

The benchmark inputs come from `benchmark/export_inputs.py`, run against `pangram-at-home-eval/artifacts/benchmark-v3`: 55,109 inputs (40,787 shared windows and 14,322 native documents), sha256 `02b6fdb5…` gzipped. They contain benchmark text, so they stay on the Space and out of git.

Score files land in `benchmark/scores/`. Copy them to `pangram-at-home-eval/artifacts/external_scores/`, where that repo's `models/precomputed.py` adapter (model ids `moe_{base,mix}_s{1,2}_{mean,maxsent}`) serves them to the normal `predict`/`calibrate`/`evaluate` steps. The adapter checks each input's text hash against the scored text.

Failures write `sweeps/ALERT-pipeline-<arm>`; progress goes to `pipeline-<arm>.log`.

## Published checkpoint

The base MoE (`moe-base-lr1e4-w10-s1`, stage-2 shard 3) is on the Hub as the private model repo `open-text-detector/moe-span-detector-base-20261006` (commit `61507a5`). The repo holds the adapter (sha256 `1612b75d…`), its `run.json`, the loading code, an `inference.py` and a model card with the evaluation and benchmark-v3 results. It needs `peft==0.18.1`, because the fused-expert LoRA layout is version-specific.
