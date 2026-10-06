# Pangram project ledger

As of Mon Oct 5, 23:32 PDT. Synthesized from all 8 Claude Code sessions on this project (Oct 3–6, 2026). Times are PDT. "Age" is time since a thread was last touched. This file contains no credentials, infrastructure addresses, paper text or per-paper scores.

## Contents
- [What you are trying to solve](#what-you-are-trying-to-solve)
- [What we have learned](#what-we-have-learned)
- [Waiting on you (project owner)](#waiting-on-you-(project-owner))
- [Unfinished threads](#unfinished-threads)
- [Experiments](#experiments)
- [Decisions](#decisions)
- [Flags raised but not answered](#flags-raised-but-not-answered)
- [Errors, time lost and fixes](#errors-time-lost-and-fixes)
- [Standing rules](#standing-rules)
- [Sessions](#sessions)
- [Links](#links)

## What you are trying to solve
- **Core research.** Detect AI-written text in research papers at document, sentence and token level while keeping false positives on human writing very low. The hardest case is catching one- or two-sentence AI edits inside otherwise human papers.
- **Training data.** Close the data gap that limits small-edit detection (only 2 of 24,000 stage-2 windows had an AI span under 200 characters), and grow trustworthy human paper text without teaching a formatting shortcut.
- **Text extraction.** Produce clean, positioned paper text from PDFs (the 'baseline': text-layer extraction plus cleanup), with figures, tables, garbled math and references removed and every omission marked.
- **Evaluation.** Report recall at a fixed 1% human false-positive rate, with ROC curves, held-out sets and enough seeds to beat noise, and test against writers other than the training generator.
- **Model and recipe.** Pick a backbone and LoRA recipe that is stable and cheap to iterate on; test whether larger models (MoE) or longer training help.
- **Paper Atlas.** Publish classifications on the Paper Atlas website with PDF overlays, calibration context and a reader that is pleasant to use, as a public version for collecting feedback before the announcement.
- **Infrastructure.** Run reliably across the HF Space (8×A100), the shared H200 node and Cloudflare R2 without losing work to restarts, quotas or rate limits.
- **Openness.** Keep the repo public-ready (no credentials, addresses, paper text or per-paper scores) and share work with a collaborator.

## What we have learned
1. **Training data was the main limit on small-edit detection.** Splicing single AI sentences into human passages lifted small-edit recall at 1% FPR from 0.25 to 0.58–0.63 (Qwen3.5-4B, 3 seeds). Recipe changes had stayed at 0.21–0.25. _(Oct 5 22:52)_
2. **The learning-rate schedule caused the collapses, not the recipe.** The 5e-4 constant schedule without warmup collapsed around step 1,000 (4B and 9B); the same peak with warmup and cosine was fine. Recipe A (2e-4 cosine, 6% warmup) became the default. _(Oct 5 06:22)_
3. **Longer training helps paper rewrites, not small edits.** Full-length 4B reached 0.89 on paper_v3 vs 0.77 at 20% length; small-edit recall stayed at 0.21–0.24. _(Oct 5 12:47)_
4. **MoE: short run best, long run unstable.** The 20%-length Qwen3.6-35B-A3B scored all-edit 0.84 and small-edit 0.30; the full-length run at 2e-4 diverged, and the stable 1e-4 rerun was worse than the 20% run. _(Oct 5 22:31)_
5. **No evidence of an extraction-artifact shortcut.** A shortcut here means a feature that predicts the label but is unrelated to authorship. Cleaning human text did not change FPR; adding extraction artifacts to AI text lowered recall only slightly (−0.05 with 10× the artifacts). 0 of 1,074 math sentences were flagged. _(Oct 5 21:45)_
6. **Thresholds set on ordinary human papers do not hold on harder human text.** Cutoffs set on ordinary human papers flag 4.5–7% of human sentences next to edits and 16–29% of human paper_v3 sentences. _(Oct 5 11:13)_
7. **Fast10 4B calibrated at 1% FPR.** Sentence threshold 0.0283 (held-out FPR 0.83%); catches 84% of one-sentence, 86% of two-sentence and 97% of paragraph edits. Rests on 215 pre-2023 papers; document-level threshold rests on 2. _(Oct 5 17:10)_
8. **ICLR flag rates rise sharply by year.** Median share of sentences flagged: ≤2022 0.5%, 2023–2024 ≈0.7%, ICLR 2025 1.5–1.8%, ICLR 2026 14.5%, ICLR 2027 68%. Whether 2027 reflects AI use or a difference in submission format, or in topic and writing style over time, is unresolved. _(Oct 5 18:55)_
9. **Baseline omissions now remove most noise.** On 100 random pages, noise left fell from 175 to 22 per 1k words (references now removed) with prose loss ~3 per 1k; clean pages rose from 16% to 45%. _(Oct 5 21:29)_
10. **Bulk transfers must be bundled.** Per-file uploads stayed at ~8 MB/s regardless of parallelism; one stream reached 47 MB/s and four reached 88 MB/s. Bundles moved 27k files in 126 s. _(Oct 5 20:15)_

## Waiting on you (project owner)

Items that need a decision, approval or action from you as the project owner. Claude or a collaborator can do the work for most of them once you decide; rotating credentials needs your own account access.

- [decision] **Publishing per-paper AI scores, PDF redistribution licensing, unverified-permission passages, open weights.** Raised before the repo existed; per-paper ICLR 2027 scores are now public on the Atlas. _(raised Oct 3 18:29)_
- [decision] **Project name close to Pangram Labs; workbench imitates their dashboard and brand.** Matters before the announcement. _(raised Oct 3 18:41)_
- [decision] **Is the AI-side text run through the same omission step?** If only human text gets markers, the model can learn a formatting shortcut. _(raised Oct 5 20:22)_
- [action] **Credential rotation: the Jupyter token (Oct 5, 00:44 PDT) and HF token (Oct 5, 17:50 PDT) were printed in session output.** Rotation never confirmed. _(raised Oct 5 17:50)_
- [approval] **ICLR 2027 drift checks: read top-flagged sentences; compare ICLR 2026 submission versions.** Needed before treating 68% as AI use. _(raised Oct 5 18:55)_
- [action] **Manual 'good enough' review of omissions.** Save verdicts on ~10 random pages in the review dashboard.
- [approval] **FSDP MoE training on A100s.** Approve building --fsdp (8-bit optimizer wrapper, sharded save and validation).
- [approval] **Storage cleanup approvals.** Approve or decline the bucket, H200 and Space /tmp deletions.

## Unfinished threads

| Thread | Area | State | Last touched | Age | Priority | Next step |
|---|---|---|---|---|---|---|
| Credential rotation | Infra | waiting on you | Oct 5 17:50 | 6 h | high | Rotate the HF token and the Jupyter token. |
| Is ICLR 2027's 68% real AI use? | Evaluation | open | Oct 5 18:55 | 5 h | high | Read top-flagged sentences in 20 papers; score ICLR 2026 submission versions (~20 batch requests). |
| Same omission treatment for AI-side text | Extraction | open | Oct 5 20:22 | 3 h | high | Decide whether AI text goes through the omission step before using omitted human text in training. |
| Wave 2 results (LLE, MIX, Arep) | Data | running | Oct 5 23:10 | 24 min | high | Compare against splice wave 1 at 1% FPR; decide the next data mix. |
| Republish the comparison artifact | Evaluation | open | Oct 5 23:15 | 18 min | high | Add corrected cross-model numbers, wave 2 and per-writer held-out results. |
| Multi-writer held-out scoring | Evaluation | running | Oct 5 23:26 | 6 min | high | Score SPG, SPH, wave 2 and both MoE runs; report per writer. |
| Atlas search latency | Site | stale | Oct 3 17:17 | 54 h | medium | Stream the 80 MB index or move search out of the Worker; live search is 2.5–5 s. |
| Add human calibration data | Evaluation | open | Oct 5 19:40 | 4 h | medium | Extract ~2,000 of the 2019–2022 Atlas PDFs; recalibrate document and 0.1% thresholds. |
| Revisit overnight A vs B conclusions | Models | open | Oct 5 19:55 | 4 h | medium | Re-read arm comparisons excluding diverged 5e-4 seeds. |
| Calibrated colour bands in the reader | Site | open | Oct 5 20:35 | 3 h | medium | Per-model bands and an FPR-labelled threshold slider. |
| Apply references removal and omissions to the corpus | Extraction | open | Oct 5 21:29 | 2 h | medium | Regenerate clean datasets; run the full-corpus omission pass (~2 h CPU). |
| Omission regressions and leaks | Extraction | open | Oct 5 21:29 | 2 h | medium | Fix the small-caps heading regression, pseudocode-as-math losses, table cell leaks, caption cut. |
| Manual 'good enough' review of omissions | Extraction | waiting on you | Oct 5 22:06 | 1 h | medium | Save verdicts on ~10 random pages in the review dashboard. |
| False positives next to AI spans | Evaluation | open | Oct 5 23:05 | 30 min | medium | Measure how often human sentences are flagged at each distance from an AI span. |
| Commit and consolidate code | Infra | open | Oct 5 23:20 | 12 min | medium | Land today's work on woog97/paingram main; one pipeline tool, shared helpers, tests. |
| Calibration-window overlap with the eval suite | Evaluation | stale | Oct 3 18:31 | 53 h | low | Probably handled by calibration-exclude-papers.json (Oct 5); verify and close. |
| Ettin sentence curves and MoE vs 9B bootstrap | Evaluation | stale | Oct 4 21:17 | 26 h | low | Low value now; close unless needed for a write-up. |
| 9B scoring of ICLR 2027 | Classification | deferred | Oct 5 03:20 | 20 h | low | You skipped it; revisit with the splice-trained model instead of fast10. |
| Small-caps stray space and NeurIPS checklist spot-check | Extraction | stale | Oct 5 12:24 | 11 h | low | Fix 'LOW -RANK' joins; spot-check ~18.7k removed checklist words. |
| Exact 10%-length runs | Models | stale | Oct 5 13:01 | 10 h | low | Run if length vs recipe still matters; otherwise close. |
| Code-block and algorithm policy | Extraction | open | Oct 5 19:31 | 4 h | low | Decide whether code listings get a ⟦code omitted⟧ marker. |
| FSDP MoE training on A100s | Infra | waiting on you | Oct 5 22:32 | 1 h | low | Approve building --fsdp (8-bit optimizer wrapper, sharded save and validation). |
| Storage cleanup approvals | Infra | waiting on you | Oct 5 22:31 | 1 h | low | Approve or decline the bucket, H200 and Space /tmp deletions. |
| Clause-labeling v3 | Data | running | Oct 5 23:24 | 6 min | low | Score v3 with 0.6B embeddings; rerun spaCy units; 4B embeddings when a GPU frees. |

## Experiments

| Finished / last update | Area | Experiment | Setup | Result | Status |
|---|---|---|---|---|---|
| Oct 3 17:17 | Site | Atlas redesign and performance | Cloudflare Workers | Redesign and mobile fixes shipped; live search still 2.5–5 s. | partial |
| Oct 3 19:46 | Models | Six-backbone comparison (fast10) | ModernBERT, Ettin-1B, Qwen3.5-4B/9B, Gemma-4-12B, Qwen3.6-35B-A3B; five at 10% length | Decoders beat encoders on public text by ~0.16 AUROC; paper AUROC tied (MoE 0.970). Sentence recall@1% FPR (1-sent/2-sent/para): MoE 32/46/95%, 9B 25/37/91%, 4B 18/18/44%. | done |
| Oct 4 22:02 | Extraction | Extraction artifact scan | 31,211 papers | 11 artifact classes (e.g. ICLR review line numbers in 100% of ICLR 2027). | done |
| Oct 4 22:05 | Data | Training-window audit | 24,000 stage-2 windows | Only 2 windows contain an AI span under 200 characters. | done |
| Oct 5 00:03 | Extraction | positioned-clean-v2 + backfill | 41,004 + 1,414 ICLR 2027 papers | Cleaned in ~28 min; 59.5M line-number words removed; 4 papers failed extraction. | done |
| Oct 5 03:15 | Classification | Inference throughput benchmarks | A100, 256 windows | 9B 9.7 → 11.3 win/s (merged LoRA); 4B 14.4 → 23.0 win/s (merged + torch.compile). Batch size had no effect. | done |
| Oct 5 11:13 | Evaluation | Calibration transfer test | 16 checkpoints on calibration windows | Cutoffs flag 0.2–1.6% of ordinary human papers but 4.5–7% of human sentences beside edits. | done |
| Oct 5 12:47 | Models | Overnight LoRA sweep (arms A–L) | Qwen3.5-4B/9B, 20% length, up to 3 seeds; A100 + H200 | A (2e-4 cosine) all-edit 0.770, small 0.249. 5e-4 constant collapses; standard AdamW (G) 0.515; short-span oversampling (L) no help. Small-edit recall stays at 0.21–0.25. | done |
| Oct 5 13:01 | Evaluation | Held-out 1% FPR, matched length | cutoff fit on dev, applied to test | Best now vs Oct 3: all-edit 0.78→0.80, small 0.22→0.26, paper_v3 0.59→0.87. | done |
| Oct 5 13:22 | Extraction | MinerU2.5-Pro | A100, vLLM | First full run lost ~10 h to OOM; coverage check 98.3% median; judged too slow; kept a 300-paper, 9-page subset. | stopped |
| Oct 5 14:39 | Models | MoE 20% length (moe-A-s1) | 1 H200, 62 min | All-edit 0.84, paragraph 0.98, small 0.30 (1 seed). | done |
| Oct 5 16:30 | Classification | ICLR 2027 scoring (fast10 4B) | 42,419 papers; 1 GPU, then 8 | Done after a Space restart; 8-GPU claim queue finished the last 24k papers in ~2 h. | done |
| Oct 5 16:40 | Data | N-gram dashboard | all human vs AI pools, hashing counter | 11 groups in under 2 min; soft-label explainer added. | done |
| Oct 5 17:10 | Evaluation | 1%-FPR calibration and ROC (fast10 4B) | 215 human papers, eval suite | Threshold 0.0283, held-out FPR 0.83%; 0.1% threshold unstable; document level rests on 2 papers. | done |
| Oct 5 18:12 | Extraction | OpenReview ICLR 2024/2025 sample | 71 batch requests, Space extraction | 3,477 new papers extracted and cleaned with positioned-clean-v2. | done |
| Oct 5 18:55 | Evaluation | Year diagnostic | 7,102 papers, 2023–2026 + downloads | Flag rates flat through 2024, rising from 2025; ICLR 2026 14.5%, ICLR 2027 68%. | done |
| Oct 5 19:25 | Models | MoE full length at 2e-4 | 2 H200s, data parallel | Diverged near step 950 (loss peak 1.4); small-edit 0.14. ~3 h of H200 time lost. | failed |
| Oct 5 19:55 | Infra | Divergence guard replay | all past H200 loss logs | Flagged all 5 blown-up runs within 20–60 steps; 0 false alarms on 30 healthy runs. | done |
| Oct 5 20:30 | Site | Atlas publication | 42,418 detail files, catalogue update | ICLR 2027 classified as 'Qwen3.5-4B · Experimental'; Calibration page live. | done |
| Oct 5 21:29 | Extraction | Baseline omissions hill-climb (v4→v6→math policy→references) | Sonnet judges, 176 pages | Noise left 175 → 22 per 1k; prose loss ~3 per 1k; clean pages 16% → 45%. | done |
| Oct 5 21:45 | Evaluation | Artifact-shortcut test | 4B and 9B; human text cleaned, AI text given extraction artifacts | No FPR change from cleaning human text; adding artifacts to AI text lowers small-edit recall by up to 0.05. | done |
| Oct 5 22:21 | Data | Luna LLM sentence edits v1 | 5,370 calls, Flex tier | 5,000 accepted for $0.46; median span 131 characters. | done |
| Oct 5 22:31 | Models | MoE full length at 1e-4 | 2 H200s, guard on | Stable, but small-edit 0.16 dev / 0.21–0.23 test, worse than the 20% run. | done |
| Oct 5 22:31 | Infra | A100 MoE benchmark (FSDP2) | 2 A100s | 7.4 s/step at micro-batch 16; full run ≈5.1 h on 2 A100s vs 2.6 h on 2 H200s. | done |
| Oct 5 22:52 | Data | Splice wave 1 (SPH, SPG) | 4B, 20% length, 3 seeds each | Small-edit recall 0.249 → 0.582 (SPH) / 0.629 (SPG); costs: standalone rewrites 0.51 → 0.36 / 0.26, public AUROC (SPG) 0.911. | done |
| Oct 5 23:05 | Evaluation | Cross-model eval (heterogeneous-ai-spans) | both MoE runs | Sentence AUROC 0.87–0.97 across six writers; recall@1% understated because 8% of human sentences inside mixed documents are flagged, mostly next to AI spans. | rerunning |
| Oct 5 23:10 | Data | Wave 2 (LLE, MIX, Arep) | 4B, 20% length, 1 seed each | Results pending. | running |
| Oct 5 23:20 | Data | Clause-labeling test (soft n-gram labels) | Luna vs spaCy splitters, synthetic edits | Synthetic v2 macro-F1: Luna clauses 0.953, sentences 0.971 (v2 biased to sentences); v3 with clause edits running. | running |
| Oct 5 23:26 | Evaluation | Multi-writer held-out edit set | Claude subagents + Luna | 1,185 Claude edits (Opus 349, Sonnet 335, Haiku 201) + 300 Luna; scoring of all models queued. | running |

## Decisions

### Recommendations you accepted
- Git repo, public-ready but private until announcement; scrub from first commit _(Oct 3 18:36, 2f2a)_
- Sentence-level ROC curves on the results page _(Oct 4 21:08, 9068)_
- Pause runs immediately to free A100 GPUs 2 and 3 _(Oct 5 00:39, 9068)_
- Stride 510 and merged LoRA for ICLR scoring _(Oct 5 03:05, 3b26)_
- Monitor runs every 15 min; one-minute sanity check after launch _(Oct 5 11:55, 25a9)_
- Checkpoint policy: one resumable checkpoint during a run, final weights after scoring _(Oct 5 13:58, 9068)_
- Parallelize ICLR scoring across all 8 GPUs with an interruptible queue _(Oct 5 14:20, 3b26)_
- Flag at 1% human FPR with ROC curves as standard practice _(Oct 5 17:05, 3b26)_
- RAM disk (/dev/shm) as H200 storage for re-downloadable files _(Oct 5 17:19, 9068)_
- Rerun MoE at LR 1e-4 with divergence guard; auto-rollback option _(Oct 5 19:43, 9068)_
- Raw probabilities plus calibration metadata instead of rescaled scores _(Oct 5 19:55, 3b26)_
- Bundle bulk transfers (your idea); downloads run on the Space _(Oct 5 20:20, 3b26)_
- Math policy for omissions; remove references entirely; span-anchored judge metrics _(Oct 5 20:22, 25a9)_
- Data priorities: splice edits, artifact-shortcut test, LLM sentence edits _(Oct 5 21:07, 9068)_

### Where you went a different way

| When | Claude's position | What you decided |
|---|---|---|
| Oct 5 00:23 | Claude proposed re-adding the 9B curve on the baseline recipe | "i dont trust u, i trust john schulman": stayed on LoRA Without Regret (which later collapsed on its schedule) |
| Oct 5 13:48 | Claude recommended pausing or shrinking the Space to save cost | GPUs are a community grant; only storage costs money |
| Oct 5 02:00 | Claude launched a full classification run before benchmarking | You asked to benchmark small batches first |
| Oct 5 22:53 | Claude proposed an API budget for the multi-writer eval set | You chose Claude subagents on your account |
| Oct 5 13:18 | Claude proposed exact 10%-length runs to separate length from recipe | You chose a 20% MoE run instead |
| Oct 5 13:19 | Claude suggested more GPUs for MinerU | You cut scope to 9 pages × 300 papers |
| Oct 5 23:09 | Claude wanted 3 seeds for wave 2 | You cut to 1 seed to get the comparison artifact sooner |
| Oct 5 14:56 | Claude deleted MoE base weights after a run | You objected; weights restored; delete base weights only on explicit request |
| Oct 5 19:35 | Claude recommended keeping ICLR 2027 flags private until drift was understood | You published them publicly as 'Experimental' to iterate on the UX |
| Oct 5 23:20 | Claude flagged that pushing to a public repo publishes before the announcement | You pushed the branch woog97/workbench to rain-1/pangram-at-home |
| Oct 5 19:55 | Claude suggested rescaling scores to the human-FPR scale | You questioned it as possibly post-hoc; raw scores were kept |
| Oct 4 23:53 | Claude argued against a VLM parser (MinerU) as a text source | You ran MinerU anyway, then judged it too slow and cut it to 300 papers |

## Flags raised but not answered

| Severity | Flag | Why it matters | Raised |
|---|---|---|---|
| high | Publishing per-paper AI scores, PDF redistribution licensing, unverified-permission passages, open weights | Raised before the repo existed; per-paper ICLR 2027 scores are now public on the Atlas. | Oct 3 18:29 |
| high | Project name close to Pangram Labs; workbench imitates their dashboard and brand | Matters before the announcement. | Oct 3 18:41 |
| high | Credential rotation: the Jupyter token (Oct 5, 00:44 PDT) and HF token (Oct 5, 17:50 PDT) were printed in session output | Rotation never confirmed. | Oct 5 17:50 |
| high | ICLR 2027 drift checks: read top-flagged sentences; compare ICLR 2026 submission versions | Needed before treating 68% as AI use. | Oct 5 18:55 |
| high | Is the AI-side text run through the same omission step? | If only human text gets markers, the model can learn a formatting shortcut. | Oct 5 20:22 |
| medium | Run one agent at a time on shared files and GPUs | Parallel sessions interfered with each other several times (duplicate GPU runs, outdated CLAUDE.md, uncommitted files from several sessions in one working tree). | Oct 3 16:54 |
| medium | PyMuPDF AGPL licence risk for the omissions code | Raised twice. | Oct 5 17:03 |
| medium | Do paired passages label the whole rewritten span as AI? Compare to soft labels | data.py not checked. | Oct 5 19:03 |
| medium | Add to the human calibration set from ~20k 2019–2022 Atlas PDFs | Document-level threshold rests on 2 papers; 0.1% threshold unstable. | Oct 5 19:40 |
| medium | Manual review of omission pages in the review dashboard | No verdicts saved yet; 'good enough' decision pending. | Oct 5 21:08 |
| medium | Pre-2022 arXiv LaTeX / PMC as the top human data source | Proposed a 10k-paper first batch. | Oct 5 22:11 |
| medium | Build the --fsdp A100 MoE training path | Benchmark done; ~2 h of work. | Oct 5 22:32 |
| low | Retry the ICLR round-3 loader (stale lock on the Space) | Likely superseded by the complete ICLR 2027 dataset; confirm and close. | Oct 3 17:34 |
| low | fp32 master weights if instability persists | Trainable weights are BF16 without an fp32 copy. | Oct 5 00:24 |
| low | What is the n-gram work for: exploration or detector features? | — | Oct 5 16:45 |
| low | Calibrated colour bands per model in the reader and list | Raw-score bands colour most flagged sentences red and some green. | Oct 5 20:35 |
| low | Storage cleanup approvals: ~79 GB old sweep checkpoints in the bucket, 9.3 GB on the H200, 67 GB MoE copy in Space /tmp | — | Oct 5 22:31 |

## Errors, time lost and fixes

| Category | Problem | Times | Cost | Fix |
|---|---|---|---|---|
| Storage | Space restarts wiped /tmp | 3 | All A100 sweep checkpoints and Space tools lost; ~75 min of baseline reruns; classification runtime rebuilt | Persistent helpers in ~/.config, vendored packages on /data, checkpoints to the bucket as each run finishes |
| Storage | Shared H200 quota exhausted | 2 | curve-9b-B killed at 36%; moe-A-full crashed mid-run (~45 min) | RAM disk for re-downloadable files; measure real free space before large writes |
| Training | Loss divergence detected late | 5 | ~3 h of H200 time on a damaged MoE run; detection ~1.5 h late; pre-spike weights lost | Divergence guard, rolling snapshots, loss-aware health checks, auto-rollback |
| Transfer | Per-file transfers and polling hit rate limits | 3 | ~30 min Space 429 lockout; ~25 min slow R2 uploads; 18 GB downloaded to the Mac and 19 GB uploaded again instead of transferring directly | 500 requests / 5 min budget; bundle everything; run downloads on the Space |
| Extraction | MinerU settings changed after calibration | 1 | ~10 h of failed GPU work (21,001 OOM errors) | Never change settings between calibration and the full run; failure-rate breaker |
| Environment | Missing tools after restarts | 8 | Repeated stalls: poppler, tesseract, pypdf, Pillow, nvcc, OpenCV, transformers versions, W&B key, HF token | Pinned micromamba envs, vendored packages, keys passed via env at launch |
| Security | Credentials printed in session output | 2 | Jupyter token and HF token exposed in transcripts | Redaction and compile-check-before-send rules; rotation still pending |
| Analysis | Wrong claims later corrected | 6 | Misleading numbers on the results page and in CLAUDE.md for hours | Corrected: edit-catch rate, MinerU text loss, coverage counts, '8–12 pt seed noise', cause of slow uploads |
| Coordination | Parallel sessions interfered with each other | 4 | Duplicate GPU runs, misread GPU assignments, outdated CLAUDE.md reads, uncommitted files from several sessions in one working tree | Session handoff messages; per-thread scope; separate git index for commits |
| Tooling | Blocked tool calls and permission prompts | 9 | Blocked publishes, deletes, pushes and launches needed manual approval; sleep blocked; browser extension unavailable | Manual mode for approvals; background watchers; headless browser via puppeteer |
| Process | Jobs launched before checks were in place | 3 | Unbenchmarked launch; prototype without a progress counter; watchers that only checked completion | Benchmark first; 1/2/4/…/60-min backoff checks; loss-aware monitoring |
| Process | Annotation effort abandoned | 1 | 300-sentence gold set stopped after ~15 | Synthetic known-truth edits instead |
| Site | Reader layout bugs | 3 | Page jumps broken, highlights hidden on Discover, page-strip marks collapsed | Bounded reader height, full reader on Discover, gradient strips, follow-reading panel |

Recurring patterns: ephemeral `/tmp` on the Space (3 wipes), shared storage quotas on the H200 (2 crashes), late detection of loss divergence, per-file transfers, and parallel sessions interfering with each other. Most fixes are now standing rules (below) or in `CLAUDE.md`.

## Standing rules

**Communication**

- Keep answers short and direct; when asked for status, give status, not results.
- Report times in PDT.

**Evaluation**

- Flag at 1% human FPR with ROC curves; never present 0.5-cutoff labels as classifications.
- Papers from 2022 or earlier are human-written; don't argue it.
- Distrust aggregates; validate visually and with frozen, random held-out sets.

**Training**

- Follow LoRA Without Regret where possible; present evidence when data disagrees.
- BF16 for inference, trainable weights and checkpoints; no fp32 copies.
- During a run keep one resumable checkpoint; after scoring keep final weights only; delete base weights only on explicit request; never delete the MoE base weights.

**Operations**

- Benchmark throughput on small batches before launching large runs.
- Check every job at 1, 2, 4, 8, 16, 32 minutes, then hourly; watch loss, not just liveness; fix and reset on stalls.
- Bundle bulk transfers; stay under 500 HF requests per 5 minutes; run downloads on the Space.
- Download new model weights only on the Space (exceptions recorded in AGENTS.md); use /dev/shm on the H200 for re-downloadable files.
- Ask before deleting anything on the Space or launching on shared hardware beyond what was agreed.

**Security**

- Never print or commit credentials; compile-check credential-bearing remote code; keep helpers and keys in ~/.config/pangram.

**Openness**

- Commits must be public-ready: no credentials, infrastructure addresses, third-party paper text, per-paper scores or brand assets.

**Data**

- OCR only as a fallback for garbled pages; omit figure text and keep captions; mark every omission; remove references from classification input.
- LLM labelling on Luna (Flex tier) with spend caps and a pilot first; eval edits via Claude subagents.

## Sessions

| Session | From | To | Focus |
|---|---|---|---|
| 9068 | Oct 3 15:39 | Oct 5 23:26 | Backbone evals, results page, overnight LoRA sweep, MoE runs, divergence guard, splice/LLM edit data, cross-model and held-out evals |
| f0fb | Oct 3 15:42 | Oct 3 17:17 | Paper Atlas redesign, mobile layout, search performance, Remote Control setup |
| 2f2a | Oct 3 17:21 | Oct 3 18:42 | State survey of Space/R2/H200; git repo creation and publication rules |
| d22e | Oct 4 20:44 | Oct 4 20:58 | ROC/FPR explainer; Pangram report's AUROC/FPR inconsistency |
| 67fb | Oct 4 21:17 | Oct 4 21:38 | H200 storage, Space helper recovery out of /tmp |
| 25a9 | Oct 4 21:33 | Oct 5 22:10 | Text extraction artifacts, positioned-clean-v2, MinerU, baseline omissions and LLM-judge hill-climb |
| 3b26 | Oct 5 01:06 | Oct 5 23:32 | ICLR 2027 scoring at scale, 1%-FPR calibration, year diagnostic, OpenReview sample, Atlas publication and reader UX, sharing |
| 33a3 | Oct 5 19:01 | Oct 5 23:27 | Soft n-gram / clause labeling test (Luna splitter, spaCy, synthetic edits) |

## Links
- [Paper Atlas (live)](https://pangram-paper-atlas.woog09.workers.dev)
- [Calibration page](https://pangram-paper-atlas.woog09.workers.dev/?view=calibration)
- [Backbone and sweep results](https://claude.ai/artifact/QJJUjcHwHNmXuqFNKX28QQ)
- [Model comparison (latest)](https://claude.ai/artifact/1Wey5VmYi8yN2gYgZJ7Zjy)
- [Fast10 4B calibration](https://claude.ai/artifact/4FNSVgD1vH8i87w4XYpq7D)
- [Omissions review dashboard](https://claude.ai/artifact/5KBFUaGaYrmcisSiSNUqxc)
- [Text layer vs MinerU](https://claude.ai/artifact/WJFxShnEYkfVkSer1d82cb)
- [N-gram dashboard](https://claude.ai/artifact/WPLSuRnVBwgjoS9uHm5GcP)
- [Clause annotation page](https://claude.ai/artifact/YVG8QfNMXZBKEU6rdZsQna)
- [Shared branch](https://github.com/rain-1/pangram-at-home/tree/woog97/workbench)
