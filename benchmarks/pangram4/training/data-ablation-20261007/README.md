# Data ablation of heterogeneous-ai-spans on top of T2.1 (October 7, 2026)

**Question:** does rain1's `heterogeneous-ai-spans` v1.4.0 add anything on top of woog's T2.1 mix, and which part of it helps or hurts? On the old `prepared-v2` base, the mix MoE runs lost recall at 1% FPR and flagged human fiction.

**Setup:** Qwen3.5-4B with woog's T2.1 recipe (`run_pair_t21.sh`): 20% length, LR 2e-4 cosine with 6% warmup, head LR 2e-5, micro-batch 8, effective batch 32. Each run uses one A100. The base data is T2.1's `prepared-v2` (manifest `536ab38a…`). Each arm adds whole sets to stage 2 only, through `setup_mix.py`. The leakage screen checks the sweep evaluation, selection and calibration windows, woog's two held-out writer sets and the leakage id list.

| Arm | Added to every stage-2 shard (full-length counts; 20% is used) |
|---|---|
| C | nothing (1 seed here; woog's 3 T2.1 seeds are the main control) |
| A1-mixv2 | all windows of Gutenberg, JMLR and Standard Ebooks mixed documents + JMLR controls (2,130 + 300) |
| A2-neutral | A1 from the `layout_neutral` view (whitespace collapsed; no paragraph-length cue) |
| A3-controls | human controls only (2,130) |
| A4-gutenberg | Gutenberg mixed documents only (1,000) |
| A5-jmlr | JMLR mixed documents + controls (730 + 300) |
| A6-stdebooks | Standard Ebooks mixed documents only (500) |
| A7-half | A1 at half dose |
| A8-double | A1 at double caps (3,006 + 600) |

Beige Book and WikiText are excluded from every arm. Seeds 1–3 run in seed-major order, so each arm gets seed 1 before any arm gets seed 2.

**Scoring:** woog's battery:
- the sweep evaluation (dev/test halves);
- `t21-heldout`, her strict held-out writers;
- `heldout-writers`;
- cross-model on the heterogeneous-ai-spans v1.4.0 **test split only**, all 1,188 rows.

Her cross-model sample draws from every split, and these arms train on the train split, so her three T2.1 seeds are re-scored on the same test-only set.

**Files:**
- `setup_ablation.sh` builds `/tmp/pangram-ablation-20261007` from her T2.1 folder (read-only) and builds the arms.
- `patch_trackio.py` switches her trainer to Trackio (project `pangram-data-ablation-20261007`).
- `score_worker_abl.py`, `build_queue_abl.py`, `specs/`.

Outputs mirror to `/data/workspace/pangram-ablation-20261007` through her `persist_daemon.py`.
