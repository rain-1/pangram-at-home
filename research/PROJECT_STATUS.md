# Project status: open AI-text span detection

Snapshot at 2026-10-07 21:45 BST. Covers 10-06 and 10-07 in detail and earlier work in summary. Built from:
- the Claude Code transcripts for this repo, `pangram-at-home-eval` and `generate-heterogeneous` (the last has only a greeting);
- git history of all four repos (`pangram-at-home` main and the integration branch, `pangram-at-home-eval`, `generate-heterogeneous`, `pretraining-datawork`);
- run logs on the training Space.

Only this repo's session was active on 10-07. `pretraining-datawork` has no transcript; it is summarised from git only. All times are BST.

**New on 10-07** (rows and entries dated 10-07 below are new; earlier entries are carried over from the 10-06 snapshot):
- Data ablation of your hetero sets on top of woog's T2.1 (25 runs): your data helps small edits; A2 layout-neutral and A8 double dose best.
- Qwen3.6 MoE and Nemotron 3.5 Lightning on T2.1 + A2: Qwen best on woog's eval; both lose AI recall on benchmark-v3 vs the base MoE, with far fewer false alarms.
- EmbeddingGemma 2 tried (6 recipes) and ruled out.
- Typography audit of T2.1: two cues found; `--typo-aug` fix written, its test crashed out of memory (woog's jobs on 6 GPUs).
- Mix v2 weights lost in a Space restart.
- New decisions pending: union mix at 50% length, full-length final run, sentence-scale edit pilot, typography rerun.

Status words used below:
- **accepted**: the user agreed or acted on it.
- **rejected**: the user declined.
- **ignored**: no reply.
- **pending**: asked, no decision yet.

## 1. What you are trying to solve

1. **An open detector that localises AI-written spans inside documents**, not just document-level AI/human.
   - It should beat the open Pangram EditLens baselines at a calibrated low false-positive rate (1%).
   - 10-07: "try to make it the best possible model" (15:03).
2. **Training data that teaches the right signal.**
   - Coherent human/AI mixtures from many modern writers (`heterogeneous-ai-spans`).
   - Human hard negatives.
   - No formatting or source shortcuts.
   - 10-07: ablate whole sets to find out which parts of your data help (10:43, 10:58).
3. **A benchmark you trust** (`aidet_eval` benchmark-v3).
   - Domain × authorship × mixture-mechanism panels.
   - Thresholds frozen on separate human calibration data; no single leaderboard number.
4. **Speed:** "we do want to operate quickly in this project" (10-06 11:44). Prefer something good over something optimal (10-06 10:33).
5. **Use the GPUs fully** while they are free ("GPUs are ours all day", 10-07 10:46).
6. **Combining with woog's workbench**, with a bias to her code. Decisions about her checkpoints and environment wait until you talk to her.
7. **Cheaper backbones and new ones:** EmbeddingGemma 2 (10-07 12:01), Nemotron 3.5 Lightning (10-07 15:02). Small models should go on smaller GPUs (15:02).

## 2. Where things stand

**No single best model.** Two Qwen3.6-35B-A3B MoEs lead on different things.

| | Base MoE (10-06) | T2.1 + A2 MoE (10-07) |
|---|---|---|
| Data | woog `prepared-v2` | woog T2.1 + your layout-neutral hetero (arm A2) |
| Length | 50% | 20% |
| Woog's eval, test all-edit / small-edit recall | 0.808 / about 0.25 | **0.803 / 0.493** |
| v3 fully-AI core recall (native, mean) | **88.6%** | 82.6% |
| v3 controlled replacement recall | **69.9%** | 32.5% |
| v3 Beemo AI / human-edited AI recall | **95.5 / 77.5%** | 60.0 / 27.5% |
| v3 hetero human controls flagged | 31.3% | **7.7%** |
| v3 AITDNA human documents flagged | 11.6% | **0%** |
| v3 source-matched human references flagged | 11.2% | **2.8%** |
| Where | HF `open-text-detector/moe-span-detector-base-20261006` (private) | Space `/data/workspace/pangram-nemotron-20261007` |

- T2.1 is built around edits to papers. It cuts fully-AI mirrors from 6,000 to 1,400 per shard and generic human text from 6,000 to 1,700.
  - The model gets precise on papers and on human text, but misses more AI text elsewhere.
  - The length difference (50% vs 20%) is a confound.
- **Candidate next model:** a union mix (T2.1 edit sets + `prepared-v2` fully-AI/generic-human shares + A2) at 50% length. Not run.

**Your heterogeneous data, after the 10-07 ablation** (Qwen3.5-4B on T2.1, 3 seeds per arm):
- **Adds +0.01 to +0.06 small-edit recall** on top of T2.1. The layout-neutral view (A2) and double dose (A8) are best.
- **Human controls alone hurt:** paper_v3 −0.04, held-out writers −0.05.
- **Single sources hurt held-out writers** (−0.04 to −0.06). Use the full mix.
- **Cost:** untouched hetero test controls flagged 0.3–0.4% → 1.0–1.4%. Held-out false positives unchanged.

**Backbones:**
- **Qwen3.6-35B-A3B stays the best backbone.**
- **Nemotron 3.5 Lightning 30B-A3B:** 1.7× faster to train. Behind on woog's paper_v3 (0.56 vs 0.77) and standalone rewrites (0.62 vs 0.85). Roughly tied on benchmark-v3.
- **EmbeddingGemma 2 (271M):** ruled out for span detection. The best of 6 recipes reached 0.28 all-edit recall, against 0.78 for the 4B.

**Hyperparameters are not the lever** (10-06 4B sweep). Data composition is.

**Typography:** the 10-07 audit found two remaining cues in T2.1.
- Mirrors (AI) are almost all curly quotes; the generic human pool is mostly straight quotes, with hard wraps and double spaces.
- Claude/Luna edits put straight apostrophes into curly-quote papers (6.5% of mixed rows).
- Woog's evaluation rows do not have the second cue, so scores are not inflated.
- A fix (`--typo-aug`: per-row random quote style, special spaces to spaces) is implemented. **Its test run failed (out of memory) and has not been rerun.**

## 3. Timeline

**09-24 to 09-28: `pangram-at-home` main (189 commits)**
- Data pyramids, baselines, and a Qwen3-1.7B passage model: 93.4% recall at 2.2% FPR; 37% on paraphrased AI.
- Ray Tune search, Repeat2 token model, span datasets v4–v14, attribution probes.
- v10 was kept as the conservative default; v14 was not a clear win.

**10-03 to 10-06 morning: `generate-heterogeneous`**
- Releases v1.0.0 (10-05 13:35), v1.1.0 (14:36), v1.2.0 (15:39), v1.3.0 (10-06 08:25).

**10-05 to 10-06: `pretraining-datawork`**
- human-core-v1 (1.1M words), Google Translate controls, Twitter controls, GrokSet queue.

**10-06, this repo:**
- 08:26–08:40 Woog's branch reviewed; curated integration branch `integrate/woog-workbench-plus-span`.
- 08:46–09:14 Your v14 data uploaded; attribution heads ported; 4B sweep wave 1 on 8 A100s (09:08).
- 10:30 Attribution heads: fast10 detector 53.3% vs raw base 43.3% Arena top-1.
- 10:47 MoE config committed after an independent Opus review.
- 11:47–12:07 Data-mix analysis; mix v1 built.
- 12:31 MoE base and mix v1 launched; 12:47 both crashed on a 512-token row; relaunched 13:27.
- 17:02 / 17:42 v3 results for base and mix v1. 17:43 seed 2 cancelled.
- 18:05 Formatting leak found in hetero v1.3.0; you published v1.4.0 at 18:46.
- 18:59–21:05 Mix v2 (v1.4.0, stage 2 only) and its v3 results.
- 21:06 Base MoE uploaded privately. 21:16 first status document.

**10-06, `pangram-at-home-eval`:** benchmark v1 (10:35), v2 (11:27), v3 (11:45); MoE models imported 17:44; committed 21:21.

**10-07 night: woog** pushed T2/T2.1 mixes, Claude/Fable/OpenAI-writer edit data, held-out lists, human calibration (2,403 papers), detector comparison page (05:53).

**10-07, this repo:**
- 10:43 Took stock of woog's commits. GPUs free; she was asleep.
- 10:56–10:58 You corrected the plan: ablate whole sets, not leakage drops.
- ~11:00 Ablation launched: 9 arms × up to 3 seeds on 8 A100s (commit cc735bd).
- 12:01 EmbeddingGemma 2 requested. ~12:30–14:40 three seeds trained and scored: all-edit 0.15.
- 14:42 One manual Trackio sync; the sync loop stayed off (local memory).
- 14:47–14:51 EmbeddingGemma diagnosis (output norm about 600); 5-arm recipe sweep launched.
- 15:02–15:40 Nemotron requested. Downloaded on the Space; LoRA, Triton scan and re-cut fixes tested; run gated on idle GPUs.
- 15:03 "make it the best possible model". Both MoEs switched to T2.1 + A2.
- 15:58–16:08 Typography audit of T2.1; `typo_aug` written; post-MoE test gated.
- 15:5x EmbeddingGemma sweep results: best 0.28. Ruled out ("fair enough", 15:56).
- 16:55 Ablation done. Nemotron and Qwen MoE started.
- 18:53 Nemotron finished all scoring; 19:43 Qwen MoE finished.
- 19:44 Typography pair and A8 seed 3 crashed out of memory: woog's `score_moe.py` held 6 GPUs from about 19:25.
- 21:30 Benchmark-v3 evaluated for both MoEs.

## 4. Experiments

| # | When | Experiment | Result | Status |
|---|---|---|---|---|
| 1 | 09-24..28 | v1–v14 span models, Qwen3-1.7B (`main`) | v10 conservative default; v12/v14 traded recall for human FPR | done |
| 2 | 10-05..06 | heterogeneous-ai-spans v1.0–v1.4 | 4,400 mixed + 4,400 controls; v1.4.0 formatting fix | batch 04 unreleased |
| 3 | 10-06 09:08 | Qwen3.5-4B one-factor sweep (15 runs) | Plateau 0.78–0.79; alpha 128 only every-subset win (1 seed) | done; alpha 128 unconfirmed |
| 4 | 10-06 09:45 | Attribution heads, fast10 4B vs raw base | 53.3% vs 43.3% Arena top-1 | done |
| 5 | 10-06 11:55 | Score woog's `moe-A-full-lr1e4` | PEFT expert-layout mismatch | abandoned |
| 6 | 10-06 12:31 | MoE base, `prepared-v2`, 50% length | Best v3 recall; woog-eval 0.808 | done, published |
| 7 | 10-06 12:31 | MoE mix v1 (hetero v1.3.0) | HAP-E-2 110/120 false positives (formatting leak) | done |
| 8 | 10-06 16:16 | MoE base + mix seed 2 | Cancelled after about 1 h | cancelled |
| 9 | 10-06 18:59 | MoE mix v2 (v1.4.0, stage 2 only) | HAP-E-2 33/120; recall below base | done; **weights lost** in a Space restart |
| 10 | 10-06 | Sparse pilot (other session) | Sparse F1 +0.37, AITDNA F1 −0.057 | needs rerun |
| 11 | 10-06 | aidet_eval benchmark v1 → v3 | 11,322 eval + 3,000 calibration documents | done |
| 12 | 10-07 11:00 | Data ablation: 9 arms on Qwen3.5-4B + T2.1, 25 runs | A2 and A8 best; controls-only and single sources hurt (§2) | done; A8 seed 3 failed |
| 13 | 10-07 12:30 | EmbeddingGemma 2, T2.1 recipe, 3 seeds | All-edit 0.15, small 0.05 | done |
| 14 | 10-07 14:51 | EmbeddingGemma 2 recipe sweep, 5 arms | Feature normalisation fixed the start; best 0.28 (rank 32, LR 5e-4) | done; ruled out |
| 15 | 10-07 16:55 | Qwen3.6 MoE, T2.1 + A2, 20% | Best on woog's eval; lower v3 recall, far fewer false alarms | done |
| 16 | 10-07 16:55 | Nemotron 3.5 Lightning, T2.1 + A2, 20% | Behind Qwen on woog's eval; about tied on v3; 1.7× faster | done |
| 17 | 10-07 15:58 | Typography audit of T2.1 | Two cues (§2); eval rows clean | done; fix implemented |
| 18 | 10-07 19:44 | Typography pair (A2 ± `--typo-aug`) + A8 seed 3 | Out of memory: woog's jobs on the GPUs | failed; not rerun |

## 5. Open issues

1. **Recall vs precision trade-off between data mixes** (§2). The best model needs a mix that keeps both. Union-mix run proposed, not run. The length confound is unresolved.
2. **Small edits are still the weakest task.**
   - Recall about 0.49 with T2.1 + A2 (up from about 0.25 on `prepared-v2`).
   - All small-edit data is in papers, and 15.5k of 21.5k edits come from Luna. Your hetero spans are paragraph-sized.
   - Proposed: 1–3-sentence replacements in hetero human controls (fiction, JMLR, Hansard) by your six writers. Pilot not approved (needs API spend).
3. **Typography cues in T2.1.** Fix written, untested.
4. **Long-document false positives.**
   - Max-over-windows scoring still flags long human documents: base MoE 31–47% of hetero controls, T2.1 + A2 MoE 8–23%.
   - No length-aware calibration exists.
5. **We train on about a fifth of the available data** (20% length). Full-length training on the winning backbone is untried, about 6 hours on 4 GPUs.
6. **Shared GPUs.** Woog's jobs took 6 of 8 GPUs at about 19:25 on 10-07 without notice to this session; our jobs crashed. No coordination channel.
7. **Woog's MoE checkpoints cannot be loaded** under the Space's PEFT 0.18.1 (transposed expert factors). Her best MoE has never been scored on v3.
8. **Single seeds** for every MoE result.
9. **Benchmark caveats.**
   - 173 CNN/DailyMail and 2 EditLens items overlap Pangram training.
   - 27 v3 documents overlap base MoE training windows.
   - 19% of mixtures have an unconfirmed generator; insertions come from gemini-2.5-flash only.
   - 1% thresholds are noisy at 500 calibration documents per domain.
   - Mean vs maxsent chosen after seeing results risks test selection.
10. **Public-repo risk.** `rain-1/pangram-at-home` is public, and the integration branch holds woog's private history. It has not been pushed.
11. **Trackio is incomplete.**
    - The ablation project was synced once (14:42).
    - The EmbeddingGemma and Nemotron/MoE projects are local to the Space only.
    - Local memory (2 GB free) kills the sync loop.

## 6. Things Claude flagged, and your response

| When | Flag or recommendation | Response |
|---|---|---|
| 10-06 08:4x | A pure hyperparameter sweep will likely reproduce woog's plateau | rejected; the sweep confirmed the plateau |
| 10-06 10:32 | Ray Tune / per-module LRs | rejected ("just something good") |
| 10-06 10:47 | MoE config: LR 1e-4, warmup 10%, rollback, dev-based checkpoint choice | accepted |
| 10-06 11:37 | Keep Beige Book out of training and calibration | accepted for training; eval side not acted on |
| 10-06 11:47 | Analysed, non-blind mix | accepted |
| 10-06 12:3x | Seed-2 replication | rejected ("i dont care about this") |
| 10-06 14:3x | Score woog's checkpoints via transpose conversion | pending (deferred to woog) |
| 10-06 16:14 | Stage-2 span-length curriculum | accepted as a note only |
| 10-06 16:17 | Rebuild Trackio runs merged with crashed attempts | ignored |
| 10-06 17:48 | Length-aware calibration for long documents | pending |
| 10-06 18:05 | Formatting leak in v1.3.0 | accepted; v1.4.0 |
| 10-06 18:58 | `layout_neutral` / lower dose next | accepted; done as ablation arms A2 and A7 on 10-07 |
| 10-06 18:58 | Woog's base-data quote cue | accepted on 10-07 (audit + fix) |
| 10-06 21:09 | Score woog's `moe-A-s1` on v3 | pending |
| 10-06 08:3x | Push the branch / open a PR; delete the superseded merge branch | ignored (asked twice) |
| 10-06 eval 18:04 | Line-break diagnostic on MoE models | pending |
| 10-07 10:5x | Ablation plan including leakage-drop arms | corrected by you: whole sets only |
| 10-07 11:xx | Leave-one-out arms | ignored |
| 10-07 14:45 | EmbeddingGemma recipe sweep before ruling it out | accepted ("in theory yes, if it works it would be nice", 15:02) |
| 10-07 15:5x | Rule out EmbeddingGemma 2 for span detection | accepted ("fair enough") |
| 10-07 14:4x | Hold A8 seed 3 to free 4 GPUs together | Claude's decision; not asked |
| 10-07 15:4x | Final model at full length on the winning backbone | pending |
| 10-07 15:4x | Data priority 1: typography audit of T2.1 | accepted |
| 10-07 15:4x | Data priority 2: sentence-scale edit pilot outside papers (API spend) | pending |
| 10-07 15:4x | Data priority 3: hard human negatives in long fiction | pending |
| 10-07 16:06 | Run the typography test after the MoEs, not before | accepted ("yeah fair") |
| 10-07 16:08 | Commit today's code | done now, with this document |
| 10-07 21:3x | Typography pair on free GPUs 4/6; union-mix MoE at 50%; T2.1+A2 50% control | pending |

## 7. Errors and time lost

| When | What happened | Cost | Fix |
|---|---|---|---|
| 10-06 12:47–13:27 | Both MoE runs crashed on one 512-token row; idle until the hourly check | 40 min × 8 A100 | Drop over-length rows; 2-minute watcher |
| 10-06 16:16–17:43 | Seed-2 runs started, then cancelled | about 1 h × 8 GPUs | Stopped on request |
| 10-06 eval 12:26–17:54 | `.gitignore` hid the model adapters from the commit | 5.5 h undetected | Anchored `.gitignore` |
| 10-06 09:00–09:47 | Trackio dashboard stale (reads the bucket only at startup) | about 45 min | Upload + restart |
| 10-06 17:5x | Mix v1 failure misdiagnosed as window skew; real cause was formatting | about 30 min | Balance and format audits |
| 10-06 11:55 | Woog's MoE checkpoints unloadable (PEFT layout) | about 20 min | Documented hazard |
| 10-06 night | Space restart wiped `/tmp`: mix v2 weights and its shard-3 score lost | one finished run | Archive to `/data` at run end (now in `run_arm.sh` and the persist daemon) |
| 10-07 morning | Local scratch wiped overnight | minutes | Helpers recreated |
| 10-07 ~11:00 | Setup wait loop matched a pip "ERROR" warning as failure | minutes | Fixed match |
| 10-07 12:0x–12:30 | EmbeddingGemma setup: transformers 5.17 lacks the model; no CLS/SEP; head width 768 vs 512 (first patched the wrong file); guard patch missed a combined line | about 30 min | transformers 5.19 isolated; bos/eos; width from `embedding_dim` |
| 10-07 12:30–14:40 | EmbeddingGemma trained without feature normalisation (output norm about 600) | 3 runs × about 1 h | LayerNorm before heads; recipe sweep |
| 10-07 13:4x | Trackio sync loop killed by low local memory | dashboard stale | One manual sync |
| 10-07 15:1x | Copying 62 GB from the Space's `/data` mount ran under 1 MB/s | about 15 min | Re-downloaded the pinned revision from the Hub to the Space's local disk (30 s) |
| 10-07 15:2x | Stop loop matched its own shell again | minutes | Rerun |
| 10-07 15:4x | Reported times in PDT (from `AGENTS.md`) while you are on BST | confusion | BST from now on |
| 10-07 19:44 | Typography pair and A8 seed 3 crashed out of memory twice each (woog's jobs on 6 GPUs) | test not run (about 2 h planned) | Not rerun; GPUs 4 and 6 free at 21:18 |
| 10-07 design | MoE comparison at 20% vs the base at 50% length | confounded comparison | Proposed 50% control |

## 8. Unfinished threads

Ages are as of 21:45 BST on 10-07.

| Thread | Last touched | Age |
|---|---|---|
| Typography test (A2 ± `--typo-aug`) and A8 seed 3 | 10-07 19:44 | 2 h |
| Union-mix MoE at 50% length; T2.1 + A2 50% control | 10-07 21:30 | 0.3 h |
| Full-length run of the final model | 10-07 15:4x | 6 h |
| Sentence-scale edit pilot outside papers | 10-07 15:4x | 6 h |
| Hard human negatives for long fiction | 10-07 15:4x | 6 h |
| `pangram-at-home-eval/configs/models.yaml`: new MoE entries uncommitted | 10-07 21:20 | 0.5 h |
| Trackio: ablation, EmbeddingGemma and MoE projects not synced | 10-07 14:42 | 7 h |
| Talk to woog: GPU sharing, PEFT hazard, T2.1 typography cue, branch location | 10-06 12:02 | 34 h |
| Score woog's `moe-A-s1` on v3 | 10-06 21:09 | 24.5 h |
| Length-aware calibration in aidet_eval | 10-06 17:48 | 28 h |
| Line-break diagnostic on MoE models | 10-06 18:04 | 27.5 h |
| `generate-heterogeneous` uncommitted: v1.4.0 formatting scripts, sparse pilot, prompt v4 | 10-06 18:46 | 27 h |
| Sparse pilot rerun on normalised data | 10-06 18:41 | 27 h |
| Hetero batch 04 (1,000 documents) unreleased | 10-06 11:39 | 34 h |
| Alpha 128 4B confirmation | 10-06 10:40 | 35 h |
| Attribution heads on the best checkpoint | 10-06 10:30 | 35 h |
| Superseded merge branch not deleted (integration branch pushed 10-07 22:0x) | 10-06 08:40 | 37 h |
| Dataset upload questions (PERSUADE, Qwen2.5 licence) | 10-06 08:55 | 37 h |
| pretraining-datawork: GrokSet text, NEWSROOM, licence questions | 10-06 16:43 | 29 h |
| `main`: blind test groups, cluster CIs, larger attribution test, 300-article eval | 09-28 | 9 days |

Closed since the last snapshot: eval-repo commit (10-06 21:21), next mix iteration (ablation arms A2/A7), woog's quote cue (audited), mix v2 shard-3 score (lost with the weights).

## 9. Where things live

| What | Where |
|---|---|
| Integration branch (code, sweeps, MoE, docs) | `pangram-at-home`, `integrate/woog-workbench-plus-span`, local only |
| Data decisions registry | `research/DATA_SOURCES.md` |
| Mix analysis, specs, leakage ids, format audits | `research/data-mix-20261006/` |
| 10-06 sweep / MoE code | `benchmarks/pangram4/training/hparam-sweep-20261006/` |
| 10-07 data ablation code and results | `benchmarks/pangram4/training/data-ablation-20261007/` |
| 10-07 MoE backbone comparison code and results | `benchmarks/pangram4/training/nemotron-20261007/` |
| 10-06 weights + records | Space `/data/workspace/hparam-sweep-20261006/` |
| 10-07 ablation outputs | Space `/data/workspace/pangram-ablation-20261007/` |
| 10-07 MoE adapters, evals, benchmark scores | Space `/data/workspace/pangram-nemotron-20261007/` |
| 10-07 EmbeddingGemma runs | Space `/data/workspace/pangram-embgemma-20261007/` |
| Nemotron weights | Space `/data/workspace/model-cache/nemotron35-lightning-30b-a3b` |
| Published base MoE | HF `open-text-detector/moe-span-detector-base-20261006` (private) |
| Published T2.1 + A2 MoE (10-08) | HF `open-text-detector/moe-span-detector-t21a2-20261007` (private; adapter sha256 `3d9f2543…`) |
| Published T2.1 + A2 Qwen3.5-4B, seed 2 (10-08) | HF `open-text-detector/qwen35-4b-span-detector-t21a2-20261007` (private; adapter sha256 `722942b4…`) |
| Heterogeneous data | HF `open-text-detector/heterogeneous-ai-spans` v1.4.0 (public) |
| Training dashboard | Trackio Space `eac123/pangram-hparam-sweep-trackio` (private) |
| Benchmark results | `pangram-at-home-eval/artifacts/runs/ai-detector-panels-v3__577af7a44f/report.md` |
