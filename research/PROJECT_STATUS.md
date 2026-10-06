# Project status: open AI-text span detection

Snapshot at 2026-10-06 21:40 BST. Built from:
- the Claude Code transcripts for this repo, `pangram-at-home-eval` and `generate-heterogeneous` (the last has only a greeting);
- git history of all four repos (`pangram-at-home` main and the integration branch, `pangram-at-home-eval`, `generate-heterogeneous`, `pretraining-datawork`);
- run logs on the training Space.

`pretraining-datawork` has no transcript; it is summarised from git only. All times are BST.

Status words used below:
- **accepted**: the user agreed or acted on it.
- **rejected**: the user declined.
- **ignored**: no reply.
- **pending**: asked, no decision yet.

## 1. What you are trying to solve

1. **An open detector that localises AI-written spans inside documents**, not just document-level AI/human.
   - It should beat the open Pangram EditLens baselines at a calibrated low false-positive rate (1%).
2. **Training data that teaches the right signal.**
   - Coherent human/AI mixtures from many modern writers (`heterogeneous-ai-spans`).
   - Human hard negatives.
   - No formatting or source shortcuts.
3. **A benchmark you trust** (`aidet_eval` benchmark-v3).
   - Domain × authorship × mixture-mechanism panels.
   - Thresholds frozen on separate human calibration data; no single leaderboard number.
4. **Speed:** "we do want to operate quickly in this project" (11:44). Prefer something good over something optimal (10:33).
5. **Combining with woog's workbench**, with a bias to her code (08:26, 08:36). You deferred decisions about her checkpoints and environment until you can talk to her (12:02).

## 2. Where things stand

**Best model: MoE base**, private on the Hub as `open-text-detector/moe-span-detector-base-20261006`.
- Qwen3.6-35B-A3B LoRA trained on woog's `prepared-v2`, at half length.
- Benchmark-v3 at 1% FPR: beats both EditLens models on every AI and mixed panel.

| Panel | MoE base | Mix v1 | Mix v2 | EditLens Llama |
|---|---:|---:|---:|---:|
| fully-AI core recall | **87.1%** | 82.7% | 84.5% | 82.2% |
| controlled replacement recall | **60.0%** | 47.6% | 46.8% | 30.8% |
| Beemo editing AI / mixed recall | **89.0 / 60.0%** | 55.0 / 33.5% | 61.5 / 35.5% | 71.0 / 50.0% |
| audited human core FPR | 1.2% | 1.4% | 0.78% | **0.56%** |
| HAP-E-2 human fiction flagged | **0/120** | 110/120 | 33/120 | 5/120 |
| hetero human controls FPR (long documents) | 47% | 34.7% | 33% | **19.7%** |
| Standard Ebooks humans flagged | not checked | 100% | 100% | not checked |

**Your heterogeneous data:**
- **Helps on its own task:** standalone rewrites go from 0.49 to 0.80 on woog's evaluation (mix v2, shard 2).
- **Costs recall elsewhere at 1% FPR**, because its threshold is about 5× base's.
- **v1's 110/120 false positives came mainly from formatting** (CRLF, hard wraps, curly quotes on the human side only). v1.4.0 fixed that.
- **Remaining suspected cues:**
  - paragraph length: human fiction paragraphs are 145 chars at the median against 555 for AI;
  - Standard Ebooks typography and diacritics;
  - the contrast between pre-1919 sources and modern writers.

**Hyperparameters are not the lever.** The 15-run 4B sweep stayed at about 0.78 all-edit recall, matching woog's plateau. Only alpha 128 beat the reference on every subset, at one seed, and was never confirmed.

## 3. Timeline

**09-24 to 09-28: `pangram-at-home` main (189 commits)**
- Data pyramids, baselines, and a Qwen3-1.7B passage model: 93.4% recall at 2.2% FPR; 37% on paraphrased AI.
- Ray Tune search, Repeat2 token model, span datasets v4–v14, attribution probes.
- v10 was kept as the conservative default; v14 was not a clear win.

**10-03 to 10-06 morning: `generate-heterogeneous`**
- Releases v1.0.0 (10-05 13:35), v1.1.0 (14:36), v1.2.0 (15:39), v1.3.0 (10-06 08:25).
- Overnight batches 01–04 finished at 00:58, 03:40, 07:51 and 11:39.

**10-05 to 10-06: `pretraining-datawork`**
- human-core-v1 (1.1M words), Google Translate controls, Twitter controls, GrokSet queue.

**10-06, this repo** (merging with woog's workbench, the 4B sweep and the MoE runs):
- 08:26–08:40 Woog's branch reviewed. A first subfolder-dump merge was replaced by a curated integration (`integrate/woog-workbench-plus-span`).
- 08:46–09:14 Your data uploaded (`open-text-detector/span-detection-rain1-v14`, 23 configs). Attribution heads ported. 4B sweep wave 1 launched on 8 A100s (09:08).
- 09:47 Trackio sync fixed by uploading to the bucket and restarting the dashboard.
- 10:30 Attribution result: fast10 detector 53.3% vs raw base 43.3% (Arena 50-model top-1).
- 10:36–10:49 Independent Opus review of the MoE hyperparameters; MoE config committed.
- 11:47–12:07 Data-mix analysis: 397 leakage ids; MAGE, LLMTrace, Beige Book and WikiText dropped. Mix v1 built.
- 12:31 MoE base and mix v1 launched. **12:47 both crashed** on a 512-token row. Relaunched at 13:27.
- 16:17 / 16:37 Base and mix v1 trained. v3 results at 17:02 (base) and 17:42 (mix).
- 17:43 Seed 2 cancelled at your request.
- 18:05 Formatting audit found the v1.3.0 leak. You published v1.4.0 at 18:46.
- 18:59–20:40 Mix v2: balanced windows, v1.4.0, stage 2 only, 8 GPUs. v3 results at 21:05.
- 21:06 Base model uploaded privately to the Hub.

**10-06, `pangram-at-home-eval`:**
- 09:19 `aidet_eval` spec; all 16 sources acquired (38 GB).
- 10:35 Benchmark v1; 11:27 v2; 11:45 v3.
- 12:26 First commit. 17:44 MoE models imported; the line-wrap shortcut found independently. 17:54 `.gitignore` fix.
- 21:12 Mix v2 imported. Its results were never reported in that session.

## 4. Experiments

| # | When | Experiment | Result | Status |
|---|---|---|---|---|
| 1 | 09-24..28 | v1–v14 span models, Qwen3-1.7B (`main`) | v10 conservative default; v8 failed; v12/v14 traded recall for human FPR | done |
| 2 | 10-05..06 | heterogeneous-ai-spans generation v1.0–v1.4 | 4,400 mixed + 4,400 controls released; batch 04 (1,000 docs) unreleased | batch 04 pending |
| 3 | 10-06 09:08–11:00 | Qwen3.5-4B one-factor sweep, waves 1–2 (15 runs + 1 stopped) | Plateau 0.78–0.79; alpha 128 the only every-subset win (1 seed); scale-2 adapters, dropout, batch 16/64 and head LR 1e-3 hurt | done; alpha 128 unconfirmed |
| 4 | 10-06 09:45–10:30 | Attribution heads on fast10 4B vs raw base | 53.3% vs 43.3% Arena top-1; writers 11/12 both | done |
| 5 | 10-06 11:55 | Score woog's `moe-A-full-lr1e4` | Failed: PEFT expert-layout mismatch | abandoned |
| 6 | 10-06 12:31 | MoE base, seed 1 (half length) | Best on v3; woog-eval test 0.808 all-edit | done, published |
| 7 | 10-06 12:31 | MoE mix v1, seed 1 | HAP-E-2 110/120 false positives; caused by the formatting leak | done |
| 8 | 10-06 16:16–17:43 | MoE base and mix, seed 2 | Cancelled after about 1 h | cancelled |
| 9 | 10-06 18:59 | MoE mix v2 (v1.4.0, balanced, stage 2 only) | HAP-E-2 33/120; core human FPR 0.78%; recall still below base | done; woog-eval shard 3 pending |
| 10 | 10-06 (other session) | Sparse pilot: Qwen3.5-4B + 25% sparse windows | Sparse F1 +0.37, AITDNA F1 −0.057; unnormalised data, 1 seed | needs rerun |
| 11 | 10-06 | aidet_eval benchmark v1 → v3 | v3: 11,322 eval + 3,000 calibration documents; No Robots and Beemo humans excluded | done |

## 5. Open issues

1. **Your data's net value is not yet positive at 1% FPR.**
   - Mix v2 raises the threshold about 5× and loses about 25 points on Beemo and 13 on controlled replacement.
   - Standard Ebooks humans are still 100% flagged, HAP-E-2 33/120.
   - Suspects: paragraph length, Standard Ebooks typography/diacritics, era contrast.
2. **No short-span training data.**
   - Small-edit recall stays at about 0.25 in every run.
   - Woog's data and hetero v1.0–v1.4 have no in-context AI spans under 200 chars. v14's GRADTEX "span" rows turned out to be human-only.
   - The uncommitted sparse pilot (1–3-sentence AI) is the start of a fix.
3. **Long-document false positives.**
   - Max-over-windows scoring flags 33–47% of long human controls (MoE) and 14–20% (EditLens).
   - No length-aware calibration exists.
4. **Woog's base data has a reverse formatting cue:** Luna mirrors use curly quotes, the human pool straight quotes and line breaks. Not normalised; affects every model trained on it.
5. **Woog's MoE checkpoints cannot be loaded** under the Space's PEFT 0.18.1 (transposed expert factors). Her best MoE (`moe-A-s1`, test all-edit 0.815, AUROC 0.987) has never been scored on v3.
6. **Single seeds everywhere.**
   - The mix v2 comparison also mixes seeds: its stage 1 came from base seed 2; base is seed 1.
7. **Benchmark caveats.**
   - 173 CNN/DailyMail and 2 EditLens items overlap Pangram training. 27 v3 documents overlap MoE base training windows.
   - 19% of mixtures have an unconfirmed generator.
   - Insertion panels use gemini-2.5-flash only.
   - The 500-per-domain calibration makes 1% thresholds noisy.
   - Choosing mean vs maxsent after seeing results risks test selection.
8. **Public-repo risk.**
   - `rain-1/pangram-at-home` is public. The integration branch contains woog's whole private workbench history, so it has not been pushed.
   - `rain-1/pangram-generate-heterogenous` and `pangram-pretraining-datawork` are also public.

## 6. Things Claude flagged, and your response

| Time | Flag or recommendation | Response |
|---|---|---|
| 08:4x | A pure hyperparameter sweep will likely reproduce woog's plateau; sweep data instead | rejected ("Pure hyperparameter sweep"). The sweep confirmed the plateau |
| 10:32 | Ray Tune / per-module LRs | rejected ("multiple lrs may not be required", "just something good") |
| 10:33 | Drop wave 3 (loss weights), combine winners | accepted |
| 10:47 | MoE config: LR 1e-4, warmup 10%, rollback, dev-based checkpoint choice | accepted, committed |
| 11:37 | Keep Beige Book out of calibration; report it as a stress set | accepted for training (dropped from the mix); eval-side not acted on |
| 11:43 | Integrate your data as a separate experiment after the sweep | accepted |
| 11:47 | Analysed, non-blind mix | accepted ("thank you for the data analysis") |
| 12:0x | Stop the rank 64 run to free GPUs | accepted |
| 12:0x | Half vs full length | default half (no explicit answer) |
| 12:3x | Seed-2 replication | rejected at 17:43 ("i dont care about this") |
| 14:3x | Score woog's checkpoints via transpose conversion | pending ("forget about woogs"; deferred to woog) |
| 16:14 | "Epoch" means shard; document it | accepted |
| 16:14 | Stage-2 span-length curriculum | accepted as a note only (in `EXPERIMENT_BACKLOG.md`) |
| 16:17 | Rebuild the Trackio runs merged with the crashed attempts | ignored |
| 17:48 | Max-window aggregation inflates long-document FPR; add length-aware calibration | pending (you asked why; no decision) |
| 18:05 | Formatting leak in v1.3.0 | accepted; v1.4.0 published 18:46 |
| 18:58 | Paragraph-length cue / `layout_neutral` / era contrast are next | pending |
| 18:58 | Woog's base quote cue | pending (for woog) |
| 21:0x | Queue lower-dose mix, `layout_neutral` run, modern human fiction controls | ignored (you moved to the upload) |
| 21:09 | Score woog's `moe-A-s1` on v3 | pending |
| 08:4x | Dataset upload: PERSUADE included despite "local evaluation only"; Qwen2.5-3B licence unverified | ignored |
| 08:3x / 08:4x | Push the integration branch or open a PR; delete the superseded merge branch | ignored (asked twice) |
| eval 12:25 | "Known traps" list for training | ignored |
| eval 17:51 | Unwrap fix in builder; JMLR/unwrapped-fiction panels; shortcut diagnostic | accepted; superseded by v1.4.0 |
| eval 18:04 | Run the line-break diagnostic on the MoE models | pending |
| eval 18:04 | Commit eval repo changes (precomputed adapter etc.) | ignored ("let me see") |
| eval 18:04 | Ask writers to keep the source paragraph count | ignored; but prompt v4 `match_source_paragraph_structure` exists uncommitted |

## 7. Errors and time lost

| Time | What happened | Cost | Fix |
|---|---|---|---|
| 08:31 | First merge just dumped main into a subfolder; you wanted real integration | ~15 min | New curated branch |
| 09:00–09:47 | Trackio dashboard stale: it reads the bucket only at startup | ~45 min of no dashboard | Upload to bucket + restart Space every 10 min |
| 11:46 | Prepped the wrong dataset (v14 instead of heterogeneous-ai-spans) | ~10 min | Asked; switched |
| 11:55 | Woog's MoE checkpoints unloadable (PEFT transposed expert layout) | ~20 min | Abandoned; documented as a hazard |
| 12:47–13:27 | Both MoE runs crashed on one 512-token row of woog's data (newer tokenizer); 8 GPUs idle 40 min until the hourly check | 40 min × 8 A100 | Trainer drops over-length rows; 2-minute failure watcher added |
| 12:0x | A stop command matched its own shell | minutes | Self-excluding stop script |
| 16:17–17:40 | Step-checkpoint scoring delayed seed 2; GPUs 5–7 idle ~25 min | ~25 min × 3 GPUs | `--ckpt-every 0` for later runs |
| 16:16–17:43 | Seed-2 runs started, then cancelled | ~1 h × 8 GPUs | Stopped on request |
| 17:02 | eval validator assumes EditLens bucket probabilities | ~10 min | Adapter puts extra scores in `coverage` |
| 17:5x | Misdiagnosed mix v1 failure as window-ratio skew; real cause was the formatting leak | ~30 min of reasoning | Corrected after the balance and audit checks |
| 18:5x | Mix analysis assumed v14 GRADTEX rows contained AI spans; they are human-only | one build cycle | Dropped from mix v2 |
| 10:5x–11:00 | Watcher false alarms (empty output read as SSH failure; duplicate watchers) | minor | Liveness marker; old watchers stopped |
| 11:36 | Times reported in UTC while you were on BST | confusion | Switched to BST |
| 17:54 | Session restart killed background loops (Trackio sync) | stale dashboard until 17:55 | Manual final sync |
| eval 09:54–10:15 | Builder NaN/empty-pool bugs, MinHash test, parity harness | ~25 min | Fixed |
| eval 11:57–12:10 | Viewer CPU hunt; four real causes found | ~13 min | Fixed; not confirmed by you |
| eval 12:26–17:54 | `.gitignore` hid the model adapters from the commit | 5.5 h undetected | Anchored `.gitignore` |
| eval 12:17 | Smoke run shown as the default; looked like a tiny benchmark | minutes | Newest full run shown first |
| hetero 07:48 | "et al." broke sentence checks in batch 03 | 1 retry | Citation fix commits |
| main 09-25..27 | Microbatch/precision confounds; Vast credit ran out; v14 rental CUDA mismatch | hours | Local 4080 runs; new rental |

## 8. Unfinished threads

Ages are as of 21:40 BST on 10-06.

| Thread | Last touched | Age |
|---|---|---|
| Mix v2 shard-3 score on woog's evaluation (scoring on the Space) | 20:45 | 1 h |
| Mix v2 weights only in Space `/tmp`; not archived to the bucket | 20:40 | 1 h |
| `pangram-at-home-eval` uncommitted: `models.yaml`, `precomputed.py`, viewer, diagnostics | 21:07 | 0.5 h |
| Mix v2 results not reported in the eval session | 21:12 | 0.5 h |
| `generate-heterogeneous` uncommitted: v1.4.0 formatting scripts, sparse pilot, everyday pool, prompt v4 | 18:46 | 3 h |
| Hetero batch 04 (1,000 documents) not released | 11:39 | 10 h |
| Hetero v1.4.0 README still cites v1.3.0 in places | 18:46 | 3 h |
| Sparse pilot rerun on normalised data | 18:41 | 3 h |
| Line-break diagnostic on MoE models | 18:04 | 3.5 h |
| Score woog's `moe-A-s1` on v3 | 21:09 | 0.5 h |
| Next mix iteration (`layout_neutral`, lower dose, modern fiction controls) | 21:0x | 0.5 h |
| Length-aware calibration / aggregation fix in aidet_eval | 17:48 | 4 h |
| Talk to woog: PEFT checkpoint hazard, base-data quote cue, branch/PR location | 12:02 | 9.5 h |
| Integration branch not pushed (public-repo risk); superseded `merge/workbench-span-research` not deleted | 08:40 | 13 h |
| Alpha 128 4B confirmation (3 seeds) | 10:40 | 11 h |
| Attribution heads on the best sweep checkpoint | 10:30 | 11 h |
| Doc-any calibration in `calibrated_report.py` never run on real sweep data | 08:40 | 13 h |
| Dataset upload questions (PERSUADE inclusion, Qwen2.5 licence) | 08:55 | 13 h |
| Trackio merged-run cleanup | 16:17 | 5.5 h |
| pretraining-datawork: GrokSet text, NEWSROOM, licence questions | 10-06 16:43 | 5 h |
| `main`: fresh blind test groups, cluster uncertainty, larger attribution test, unscored 300-article modern-generator eval | 09-28 | 8 days |

## 9. Where things live

| What | Where |
|---|---|
| Integration branch (code, sweep, MoE, docs) | `pangram-at-home`, `integrate/woog-workbench-plus-span`, local only |
| Data decisions registry | `research/DATA_SOURCES.md` |
| Mix analysis, specs, leakage ids | `research/data-mix-20261006/` |
| Sweep / MoE code and README | `benchmarks/pangram4/training/hparam-sweep-20261006/` |
| Final weights + records (base, mix v1, 4B sweep) | Space bucket `/data/workspace/hparam-sweep-20261006/` |
| Published base MoE | HF `open-text-detector/moe-span-detector-base-20261006` (private) |
| Your v14 data | HF `open-text-detector/span-detection-rain1-v14` (private) |
| Heterogeneous data | HF `open-text-detector/heterogeneous-ai-spans` v1.4.0 (public) |
| Training dashboard | Trackio Space `eac123/pangram-hparam-sweep-trackio` (private) |
| Benchmark results | `pangram-at-home-eval/artifacts/runs/ai-detector-panels-v3__577af7a44f/report.md` |
