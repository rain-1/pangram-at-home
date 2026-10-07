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

## Results (test half, recall at 1% FPR; mean over seeds)

| Arm | Seeds | All edits | Small | paper_v3 | Standalone | Strict held-out (FP) | Held-out writers (FP) | Hetero test (control FP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| woog T2.1 | 3 | 0.784 | 0.449 | 0.760 | 0.793 | – | – | 0.847 (0.4%) |
| C | 1 | 0.779 | 0.447 | 0.769 | 0.806 | 0.772 (2.3%) | 0.646 (3.0%) | 0.778 (0.3%) |
| A1-mixv2 | 3 | 0.775 | 0.471 | 0.769 | 0.799 | 0.756 (2.1%) | 0.635 (2.9%) | 0.986 (1.0%) |
| **A2-neutral** | 3 | **0.793** | **0.491** | 0.773 | 0.799 | 0.762 (2.3%) | 0.633 (2.9%) | 0.988 (1.4%) |
| A3-controls | 3 | 0.788 | 0.491 | **0.728** | 0.799 | 0.745 (2.0%) | 0.601 (2.5%) | 0.618 (0.7%) |
| A4-gutenberg | 3 | 0.775 | 0.462 | 0.753 | 0.786 | 0.736 (1.8%) | 0.592 (2.3%) | 0.974 (1.1%) |
| A5-jmlr | 3 | 0.789 | 0.489 | 0.793 | 0.799 | 0.738 (1.8%) | 0.587 (2.2%) | 0.815 (0.6%) |
| A6-stdebooks | 3 | 0.776 | 0.458 | 0.770 | 0.789 | 0.740 (1.9%) | 0.587 (2.3%) | 0.930 (0.6%) |
| A7-half | 3 | 0.779 | 0.471 | 0.733 | 0.772 | 0.754 (2.1%) | 0.608 (2.6%) | 0.977 (1.2%) |
| A8-double | 2 | 0.796 | 0.513 | 0.782 | 0.786 | 0.774 (2.5%) | 0.660 (3.4%) | 0.990 (1.4%) |

- Every arm adds +0.01 to +0.06 on small edits over T2.1 alone; A2 and A8 are best overall.
- Human controls alone (A3) cost paper_v3 recall and held-out writers; controls only help paired with their mixed documents.
- Single-source arms (A4–A6) lose 0.04–0.06 on held-out writers; the full mixes do not.
- The cost: false positives on untouched hetero test controls rise from 0.3–0.4% to 1.0–1.4%. Held-out FP rates do not move.
- The hetero test column is in-distribution for every arm and is not evidence of generalisation.
- A8 seed 3 and the typography arm (A2 + `--typo-aug`, `typo_aug.py`, `patch_typo.py`, `gate_typo.sh`) crashed out of memory at 19:44 BST on 10-07: woog's `score_moe.py` held six GPUs. Not yet rerun.
