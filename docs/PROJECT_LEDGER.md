# Pangram project ledger

As of Tue Oct 6, 15:56 PDT. Synthesized from all 8 Claude Code sessions on this project (Oct 3–6, 2026). Times are PDT. "Age" is time since a thread was last touched. This file contains no credentials, infrastructure addresses, paper text or per-paper scores.

## Contents
- [What you are trying to solve](#what-you-are-trying-to-solve)
- [What we have learned](#what-we-have-learned)
- [Priorities](#priorities)
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
1. **Splice-trained models catch other writers' edits, less well.** At the deployment threshold the splice-trained 4B catches 63% of Haiku edits, 53% of Luna and 37–41% of Sonnet and Opus; the MoE without splices catches 22–33%. _(Oct 5 23:54)_
2. **Current labels mark some human text as AI.** Clause-level soft labels found that 25% of the AI-labeled text in paragraph rewrites is near-verbatim human; clause labels beat sentence labels by +0.10 macro-F1 on synthetic edits. _(Oct 6 05:37)_
3. **Training data was the main limit on small-edit detection.** Splicing single AI sentences into human passages lifted small-edit recall at 1% FPR from 0.25 to 0.58–0.63 (Qwen3.5-4B, 3 seeds). Recipe changes had stayed at 0.21–0.25. _(Oct 5 22:52)_
4. **The learning-rate schedule caused the collapses, not the recipe.** The 5e-4 constant schedule without warmup collapsed around step 1,000 (4B and 9B); the same peak with warmup and cosine was fine. Recipe A (2e-4 cosine, 6% warmup) became the default. _(Oct 5 06:22)_
5. **Longer training helps paper rewrites, not small edits.** Full-length 4B reached 0.89 on paper_v3 vs 0.77 at 20% length; small-edit recall stayed at 0.21–0.24. _(Oct 5 12:47)_
6. **MoE: short run best, long run unstable.** The 20%-length Qwen3.6-35B-A3B scored all-edit 0.84 and small-edit 0.30; the full-length run at 2e-4 diverged, and the stable 1e-4 rerun was worse than the 20% run. _(Oct 5 22:31)_
7. **No evidence of an extraction-artifact shortcut.** A shortcut here means a feature that predicts the label but is unrelated to authorship. Cleaning human text did not change FPR; adding extraction artifacts to AI text lowered recall only slightly (−0.05 with 10× the artifacts). 0 of 1,074 math sentences were flagged. _(Oct 5 21:45)_
8. **Thresholds set on ordinary human papers do not hold on harder human text.** Cutoffs set on ordinary human papers flag 4.5–7% of human sentences next to edits and 16–29% of human paper_v3 sentences. _(Oct 5 11:13)_
9. **Fast10 4B calibrated at 1% FPR.** Sentence threshold 0.0283 (held-out FPR 0.83%); catches 84% of one-sentence, 86% of two-sentence and 97% of paragraph edits. Rests on 215 pre-2023 papers; document-level threshold rests on 2. _(Oct 5 17:10)_
10. **ICLR flag rates rise sharply by year.** Median share of sentences flagged: ≤2022 0.5%, 2023–2024 ≈0.7%, ICLR 2025 1.5–1.8%, ICLR 2026 14.5%, ICLR 2027 68%. Whether 2027 reflects AI use or a difference in submission format, or in topic and writing style over time, is unresolved. _(Oct 5 18:55)_
11. **Baseline omissions now remove most noise.** On 100 random pages, noise left fell from 175 to 22 per 1k words (references now removed) with prose loss ~3 per 1k; clean pages rose from 16% to 45%. _(Oct 5 21:29)_
12. **Bulk transfers must be bundled.** Per-file uploads stayed at ~8 MB/s regardless of parallelism; one stream reached 47 MB/s and four reached 88 MB/s. Bundles moved 27k files in 126 s. _(Oct 5 20:15)_

## Priorities

"Your time" is active time in conversation: for each message you sent, the gap to your next message in that session, capped at 10 minutes (longer gaps count as away). Messages were assigned to topics by session, time window and keywords, and spot-checked, so hours are approximate (about ±20%). Total: ~27.6 h of active time over 336 chat rounds (a round is one message from you and Claude's reply to it), Oct 3–6.

### Suggested order for today

1. **Train and test the new small-edit data mix.** Wave 2 was lost in the 07:19 UTC restart before it was scored. Build the planned mix from the new data pool, train 4B × 3 seeds against SPG, and run the soft-label A/B at the same time.

   <details><summary>What it takes</summary>

   - **Your time:** ~20 min: approve the mix and GPU use; agree GPUs with River
   - **Compute and wall time:** ~3–4 h wall: 4B at 20% length takes ~75–90 min per run on an A100 (seeds in parallel) plus ~20 min scoring; the soft-label A/B (6 runs) runs on other GPUs at the same time
   - **Depends on:** Free A100s (River's runs); persistence to /data running
   - **Steps:**
     1. Agree with River which A100s are free.
     1. Build the 'table 2' mix (small edits 35%, human paper 25%, paragraph 12%, fully AI 10%, generic human 7%, sections 6%, GRADTEX 5%) from the pool: ~16,000 small edits fit a 510-token window (splices ~7,760, Luna ~4,400, Claude ~4,100 after the strict check). About 1 h of work. Drop the 22 never-train rows.
     1. Train 4B, 20% length, recipe A, 3 seeds; compare with SPG (rerun SPG if its checkpoints are not in the bucket).
     1. Run the soft-label A/B (original vs clause soft labels, 3 seeds each), which the clause-labeling session has prepared.
     1. Score dev/test and the strict multi-writer held-out set (871 Claude rows + 300 Luna); report recall per writer at a held-out 1% FPR with the spread across seeds.
     1. Choose the mix; then one confirmation run at full length or on 9B (~4–5 h).
   - **Risks:**
     - Another restart: write outputs to /data (persist daemon).
     - Splices and some evaluation edits both come from Luna: decide on the Sonnet, Opus and Haiku columns.
     - Single-seed noise is ±0.03–0.06; the held-out test sets are small (±0.07 on 150 sentences).
   - **Done when:** A chosen mix with ≥3 seeds per finalist, per-writer recall at 1% FPR, and the regressions (standalone rewrites, public AUROC) listed.

   </details>

2. **Check whether ICLR 2027's 68% is real.** Read the top-flagged sentences in ~20 papers and score ICLR 2026 submission versions before more site work.

   <details><summary>What it takes</summary>

   - **Your time:** ~45 min: read flagged sentences in ~20 papers (only you can judge these)
   - **Compute and wall time:** ~3–4 h wall, mostly automated
   - **Depends on:** OpenReview quota (140 requests/hour, run on the Space); ~30 GPU-minutes
   - **Steps:**
     1. Break flags down by section from the existing sentence scores (~30 min CPU). References are still in the classification input, so check whether references or template text drive the flags.
     1. Build a reading page: top-flagged sentences in context for 20 papers with high, middle and low flag shares (~1 h).
     1. Version control: download ~1,000 ICLR 2026 rejected submissions (their PDFs are submission versions, the same format as ICLR 2027): ~20 batch requests, extraction and cleanup ~15 min, scoring ~15 min.
     1. Second model: score a 1,000-paper ICLR 2027 subset with the best splice-trained model.
     1. Write the verdict with the evidence on the Calibration page.
   - **Risks:**
     - There is no human ground truth for 2027 papers: this can rule out format and section causes but cannot prove AI use.
   - **Done when:** A written verdict with the section breakdown, the ICLR 2026 submission control and second-model agreement.

   </details>

3. **Add human calibration data.** Extract part of the ~20k 2019–2022 PDFs already in R2; the document-level threshold rests on 2 papers.

   <details><summary>What it takes</summary>

   - **Your time:** ~10 min: approve exporting PDFs from R2
   - **Compute and wall time:** ~half a day wall: ~1 h engineering, ~1 h compute, ~1 h analysis
   - **Depends on:** Read access to the Atlas R2 bucket from the Space (a temporary token-gated export endpoint, like /bundle in reverse)
   - **Steps:**
     1. Select ~2,000 papers stratified by venue and year.
     1. Export them to the Space as tar bundles (~10 GB).
     1. Baseline extraction and positioned-clean-v2 (~10 min on 88 CPU workers); remove papers that overlap training windows.
     1. Score with each model to calibrate (current 4B and the new mix winner): ~15 min per model on 8 GPUs.
     1. Recompute thresholds with paper-level bootstrap intervals (flags cluster within papers; the current intervals treat sentences as independent and are too narrow).
     1. Update the Calibration page and the catalogue metadata.
   - **Risks:**
     - 2019–2022 topics differ from 2027; thresholds may still understate FPR on recent human writing.
   - **Done when:** Thresholds from ≥1,500 papers with paper-clustered intervals; the document-level threshold no longer rests on 2 papers.

   </details>

4. **Decide whether AI-side text goes through the omission step.** This blocks using the cleaned human paper text in training.

   <details><summary>What it takes</summary>

   - **Your time:** ~15 min: choose the policy
   - **Compute and wall time:** ~3–4 h: 1–2 h implementation, ~1 h shortcut test on 2 GPUs, ~2 h CPU to regenerate the clean corpus
   - **Depends on:** Omissions v2 (math policy, references removal) is ready but not applied to the corpus
   - **Steps:**
     1. The problem: human paper text from PDFs contains ⟦… omitted⟧ markers; fully AI training documents (30% of windows) never do, so the model can learn 'marker means human'.
     1. Option A: remove markers from the input at training and inference. Simplest, no leak.
     1. Option B: keep markers, give their tokens an ignore label, and insert synthetic markers into fully AI documents at the human rate.
     1. Option C: render AI documents to PDF and extract them like human papers. Most faithful, most work.
     1. Recommendation: A now; consider C later for full AI manuscripts.
     1. Implement it in data.py and the inference preprocessing so the Atlas uses the same text; validate with the artifact-shortcut harness (marker present vs absent); regenerate the clean text with references removed.
   - **Risks:**
     - Removing markers joins text across omitted regions; check that sentence boundaries stay correct.
   - **Done when:** Policy recorded in AGENTS.md; the same preprocessing in training and inference; markers have no effect in the shortcut test.

   </details>

5. **Rotate the exposed HF and Jupyter tokens.** About 20 minutes, at a time when no running job depends on the old tokens.

   <details><summary>What it takes</summary>

   - **Your time:** ~20 min
   - **Compute and wall time:** None
   - **Depends on:** Changing the Space's Jupyter token restarts the Space and clears /tmp, so do it when no runs are active
   - **Steps:**
     1. HF: huggingface.co → Settings → Access Tokens: create a token with the same scopes, run `hf auth login` on the Mac, then revoke the old token.
     1. Relaunch detached Space jobs that were started with the old HF token in their environment (persist daemon, uploads).
     1. Jupyter: change the Space secret, wait for the restart, update ~/.config/pangram/jupyter_token (mode 600).
   - **Risks:**
     - Revoking the HF token stops any job still using it.
   - **Done when:** Old tokens revoked, new ones in place, dependent jobs relaunched.

   </details>

6. **Rescore ICLR 2027 with the chosen model and update the Atlas.** Uses the existing publish pipeline (bundled uploads).

   <details><summary>What it takes</summary>

   - **Your time:** ~15 min: approve the publish
   - **Compute and wall time:** ~1 day wall, mostly compute: scoring ~3.5 h on 8 A100s (4B), detail files ~1 h, upload ~5 min
   - **Depends on:** Item 1 (model), item 3 (calibration), item 4 (preprocessing); item 2 decides how results are described
   - **Steps:**
     1. Package the chosen checkpoint (merged LoRA, compiled) and benchmark it once.
     1. Score 42,419 papers with the 8-GPU claim queue, writing to /data.
     1. Compute sentence scores, flag rates and thresholds from the new human set.
     1. Build detail files from local copies (the bucket mount is slow) and upload with /bundle.
     1. Add the model to the catalogue next to Qwen3.5-4B · Experimental; extend the Calibration page to several models (~1–2 h).
     1. Rerun the year diagnostic with the new model.
   - **Risks:**
     - A restart during the ~3.5 h scoring run: the claim queue resumes from /data.
   - **Done when:** The new model is live on the Atlas with its own calibration section, and the year diagnostic has been rerun.

   </details>

### Where your active time went

| Topic | Your active time | Chat rounds | Payoff |
|---|---|---|---|
| Space restarts, storage and monitoring | 3.4 h | 50 | overhead |
| Data generation (splices, LLM edits) | 3.2 h | 47 | high |
| MinerU | 3.1 h | 32 | low |
| ICLR scoring, calibration and year diagnostic | 2.5 h | 22 | mixed |
| MoE runs | 2.2 h | 32 | low |
| Omissions (extraction tuning) | 1.9 h | 19 | mixed |
| Training sweeps (recipe, LR) | 1.9 h | 27 | low |
| Evals, metrics and results pages | 1.6 h | 18 | high |
| Atlas publishing and reader UX | 1.5 h | 18 | mixed |
| Clause labeling (soft labels) | 1.5 h | 13 | promising |
| Project ledger, repo and sharing | 1.2 h | 20 | overhead |
| Atlas site redesign (Oct 3) | 0.9 h | 7 | mixed |
| N-gram dashboard | 0.8 h | 9 | low |
| Repo setup and docs | 0.7 h | 9 | overhead |
| Text extraction and cleanup | 0.7 h | 5 | high |
| Claude Code setup | 0.3 h | 6 | overhead |

### Still in progress and probably not worth continuing
- **More hyperparameter or recipe sweeps.** Two independent sweeps (ours and River's) found nothing that beats the reference recipe beyond noise. Data changes moved small-edit recall from 0.25 to ~0.6.
- **Full-length MoE runs and the FSDP A100 path.** Full-length MoE overfit (best epoch 0) and scored below the 20%-length run. The one MoE run worth doing is a short run on the chosen data mix, after item 1.
- **Further omission tuning (table leaks, pseudocode, captions).** Decide item 4 first; until then more tuning may not affect training.
- **More site work on the current 4B scores.** Rescore ICLR 2027 with the chosen model first (item 6).
- **N-gram dashboard follow-ups.** No decision depends on it.

### Most of your time, least payoff

| Work | Your active time and chat rounds | Compute | Outcome | Assessment |
|---|---|---|---|---|
| Space restarts, storage and monitoring | 3.4 h active, 50 chat rounds | 4 restarts in 3 days; checkpoints and wave 2 lost | Persistence to /data, a restart watcher, the storage index and monitoring rules now exist. | Reactive work and the largest share of your time. The new tools should reduce it; the restarts themselves are unexplained. |
| MinerU PDF extraction | 3.1 h active, 32 chat rounds | ~10 GPU hours lost to out-of-memory errors, ~1 h setup | Judged too slow and dropped; a 300-paper subset kept. | Your largest block of time with the least to show for it. |
| ICLR scoring and Atlas publishing of the fast10 4B | 4.0 h active, 40 chat rounds (2.5 + 1.5) | ~8 GPU hours, 15 GB of uploads | ICLR 2027 classified on the site; reader fixes; Calibration page. | The reader and calibration work carries over. The scores come from a model the splice models outperform, and the 68% figure is unvalidated. |
| MoE runs and scaling | 2.2 h active, 32 chat rounds | ~10 H200 hours (two full runs, benchmarks) | Full-length runs overfit (best epoch 0) and scored below the 20%-length run. | Low payoff; the divergence guard built during this work is still in use. |
| Recipe and hyperparameter sweeps | 1.9 h active, 27 chat rounds | ~12 h on 4 A100s + 3 H200s, plus River's sweep | Small-edit recall stayed at 0.21–0.26 for every setting, in both sweeps. | Useful negative result, larger than needed: the data audit had already pointed to data. |
| Omission-step tuning | 1.9 h active, 19 chat rounds | ~8 h of Claude session, ~5M subagent tokens | Noise left fell from 175 to 22 per 1k words; references removal. | Partly useful; later iterations changed little, and the AI-side policy is still undecided. |
| N-gram dashboard | 0.8 h active, 9 chat rounds | minutes of CPU | Exploratory statistics; their use was never decided. | Low payoff. |

## Waiting on you (project owner)

Items that need a decision, approval or action from you as the project owner. Claude or a collaborator can do the work for most of them once you decide; rotating credentials needs your own account access.

- [decision] **Publishing per-paper AI scores, PDF redistribution licensing, unverified-permission passages, open weights.** Raised before the repo existed; per-paper ICLR 2027 scores are now public on the Atlas. _(raised Oct 3 18:29)_
- [decision] **Project name close to Pangram Labs; workbench imitates their dashboard and brand.** Matters before the announcement. _(raised Oct 3 18:41)_
- [decision] **Is the AI-side text run through the same omission step?** If only human text gets markers, the model can learn a formatting shortcut. _(raised Oct 5 20:22)_
- [action] **Credential rotation: the Jupyter token (Oct 5, 00:44 PDT) and HF token (Oct 5, 17:50 PDT) were printed in session output.** Rotation never confirmed. _(raised Oct 5 17:50)_
- [approval] **ICLR 2027 drift checks: read top-flagged sentences; compare ICLR 2026 submission versions.** Needed before treating 68% as AI use. _(raised Oct 5 18:55)_
- [approval] **Soft-label A/B (clause labels).** Approve 2 A100s for 4B original vs soft labels, 3 seeds each.
- [action] **OpenAI-writer data via Codex.** Start a Codex conversation with research/openai-writer-handoff-20261006 (D 1,000 items, E 420).
- [approval] **Commit and push the overnight code.** Branch overnight-20261006 (f479f32) is not pushed; later changes are uncommitted.
- [action] **Manual 'good enough' review of omissions.** Save verdicts on ~10 random pages in the review dashboard.
- [approval] **FSDP MoE training on A100s.** Approve building --fsdp (8-bit optimizer wrapper, sharded save and validation).
- [approval] **Storage cleanup: remaining items.** Bucket cleaned on Oct 6 (1,074 → 652 GB, index at workspace/README.md). Still open: pruning checkpoints of current runs (not approved), 9.3 GB on the H200, 68 GB MoE copy in Space /tmp.

## Unfinished threads

| Thread | Area | State | Last touched | Age | Priority | Next step |
|---|---|---|---|---|---|---|
| Credential rotation | Infra | waiting on you | Oct 5 17:50 | 22 h | high | Rotate the HF token and the Jupyter token. |
| Is ICLR 2027's 68% real AI use? | Evaluation | open | Oct 5 18:55 | 21 h | high | Read top-flagged sentences in 20 papers; score ICLR 2026 submission versions (~20 batch requests). |
| Same omission treatment for AI-side text | Extraction | open | Oct 5 20:22 | 20 h | high | Decide whether AI text goes through the omission step before using omitted human text in training. |
| Soft-label A/B (clause labels) | Data | waiting on you | Oct 6 05:37 | 10 h | high | Approve 2 A100s for 4B original vs soft labels, 3 seeds each. |
| Train and test the new data mix (wave 2 was lost) | Data | open | Oct 6 12:29 | 4 h | high | Build the table-2 mix, train 4B × 3 seeds vs SPG, score per writer. |
| Atlas search latency | Site | stale | Oct 3 17:17 | 71 h | medium | Stream the 80 MB index or move search out of the Worker; live search is 2.5–5 s. |
| Add human calibration data | Evaluation | open | Oct 5 19:40 | 20 h | medium | Extract ~2,000 of the 2019–2022 Atlas PDFs; recalibrate document and 0.1% thresholds. |
| Revisit overnight A vs B conclusions | Models | open | Oct 5 19:55 | 20 h | medium | Re-read arm comparisons excluding diverged 5e-4 seeds. |
| Calibrated colour bands in the reader | Site | open | Oct 5 20:35 | 19 h | medium | Per-model bands and an FPR-labelled threshold slider. |
| Apply references removal and omissions to the corpus | Extraction | open | Oct 5 21:29 | 18 h | medium | Regenerate clean datasets; run the full-corpus omission pass (~2 h CPU). |
| Omission regressions and leaks | Extraction | open | Oct 5 21:29 | 18 h | medium | Fix the small-caps heading regression, pseudocode-as-math losses, table cell leaks, caption cut. |
| Manual 'good enough' review of omissions | Extraction | waiting on you | Oct 5 22:06 | 18 h | medium | Save verdicts on ~10 random pages in the review dashboard. |
| False positives next to AI spans | Evaluation | open | Oct 5 23:05 | 17 h | medium | Measure how often human sentences are flagged at each distance from an AI span. |
| Commit and consolidate code | Infra | open | Oct 5 23:20 | 17 h | medium | Land today's work on woog97/paingram main; one pipeline tool, shared helpers, tests. |
| OpenAI-writer data via Codex | Data | waiting on you | Oct 6 13:07 | 3 h | medium | Start a Codex conversation with research/openai-writer-handoff-20261006 (D 1,000 items, E 420). |
| Commit and push the overnight code | Infra | waiting on you | Oct 6 15:31 | 24 min | medium | Branch overnight-20261006 (f479f32) is not pushed; later changes are uncommitted. |
| Calibration-window overlap with the eval suite | Evaluation | stale | Oct 3 18:31 | 69 h | low | Probably handled by calibration-exclude-papers.json (Oct 5); verify and close. |
| Ettin sentence curves and MoE vs 9B bootstrap | Evaluation | stale | Oct 4 21:17 | 43 h | low | Low value now; close unless needed for a write-up. |
| 9B scoring of ICLR 2027 | Classification | deferred | Oct 5 03:20 | 37 h | low | You skipped it; revisit with the splice-trained model instead of fast10. |
| Small-caps stray space and NeurIPS checklist spot-check | Extraction | stale | Oct 5 12:24 | 28 h | low | Fix 'LOW -RANK' joins; spot-check ~18.7k removed checklist words. |
| Exact 10%-length runs | Models | stale | Oct 5 13:01 | 27 h | low | Run if length vs recipe still matters; otherwise close. |
| Code-block and algorithm policy | Extraction | open | Oct 5 19:31 | 20 h | low | Decide whether code listings get a ⟦code omitted⟧ marker. |
| FSDP MoE training on A100s | Infra | waiting on you | Oct 5 22:32 | 17 h | low | Approve building --fsdp (8-bit optimizer wrapper, sharded save and validation). |
| Storage cleanup: remaining items | Infra | waiting on you | Oct 6 00:00 | 16 h | low | Bucket cleaned on Oct 6 (1,074 → 652 GB, index at workspace/README.md). Still open: pruning checkpoints of current runs (not approved), 9.3 GB on the H200, 68 GB MoE copy in Space /tmp. |

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
| Oct 5 23:38 | Evaluation | Cross-model eval (heterogeneous-ai-spans) | both MoE runs | Sentence AUROC 0.87–0.97 across six writers; corrected numbers on the comparison page. | done |
| Oct 5 23:54 | Evaluation | Multi-writer held-out eval | 1,185 Claude + 300 Luna edits on never-train papers | Recall at the deployment threshold: 4B with splices Haiku 63%, Luna 53%, Sonnet/Opus 37–41%; MoE without splices 22–33%. | done |
| Oct 6 00:00 | Infra | Training-storage cleanup and index | owner-approved Tier 1 + Tier 3 | Bucket 1,074 → 652 GB: removed orphaned partial uploads (72 GB), duplicated ICLR round-2/3 PDFs (77 GB) and Oct 1–3 experiment weights (273 GB); every folder indexed in workspace/README.md. | done |
| Oct 6 00:19 | Data | Wave 2 (LLE, MIX, Arep) | 4B, 20% length, 1 seed each | Lost in the 07:19 UTC Space restart before it was scored. | failed |
| Oct 6 05:37 | Data | Clause-labeling test (soft labels) | spaCy vs Luna splitters; synthetic v3 edits; relabel of 25,283 pairs | Clauses beat sentences by +0.10 macro-F1 (CI +0.05 to +0.16); 25% of AI-labeled text in paragraph rewrites is near-verbatim human. 4B A/B pending approval. | done |
| Oct 6 13:17 | Data | Overnight data generation | Claude subagents, ~120M tokens | 4,905 Claude sentence edits (strict check), 2,078 sections, 254 full papers, 1,228 paragraph edits; OpenAI-writer batch prepared for Codex. | done |
| Oct 6 15:31 | Models | River's one-factor sweep | 4B, 1 seed per setting | No setting beat the reference (small-edit 0.26) beyond noise; River's MoE runs 0.22–0.30. | done |

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
| low | Storage cleanup of current runs: prune non-selected checkpoints (~70–80 GB), 9.3 GB on the H200, 68 GB MoE copy in Space /tmp | Oct 6 cleanup handled the archive; these were not approved. | Oct 6 00:00 |

## Errors, time lost and fixes

| Category | Problem | Times | Cost | Fix |
|---|---|---|---|---|
| Storage | Space restarts wiped /tmp | 4 | Sweep checkpoints, Space tools and wave 2 lost; ~75 min of baseline reruns; classification runtime rebuilt | Persist daemon mirrors /tmp to /data every 5 min; restart watcher; vendored packages on /data |
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
- No direct OpenAI API generation; OpenAI-model data goes through a written handoff to a new Codex conversation.

**Collaboration**

- River (collaborator) runs jobs on the same hardware: anything not started by you or your sessions is River's; don't stop, change or delete it without asking.

**Operations**

- Space outputs go to /data (persist daemon); don't change Space hardware or settings during live runs without asking.

**Communication**

- Answer the actual question plainly; don't be pedantic. Write literally.

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
- [Project ledger page](https://claude.ai/artifact/VDHHWuQhGUPNcxFvwJRSRn)
