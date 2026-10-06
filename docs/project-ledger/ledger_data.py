"""Structured project ledger for Pangram, synthesized from all Claude Code sessions (Oct 3–6, 2026).

Times are stored in UTC ("YYYY-MM-DD HH:MM") and rendered in PDT (UTC-7) per the project rule.
"""
NOW = "2026-10-06 06:32"

SESSIONS = [
    {"id": "f0fb", "start": "2026-10-03 22:42", "end": "2026-10-04 00:17", "focus": "Paper Atlas redesign, mobile layout, search performance, Remote Control setup"},
    {"id": "2f2a", "start": "2026-10-04 00:21", "end": "2026-10-04 01:42", "focus": "State survey of Space/R2/H200; git repo creation and publication rules"},
    {"id": "9068", "start": "2026-10-03 22:39", "end": "2026-10-06 06:26", "focus": "Backbone evals, results page, overnight LoRA sweep, MoE runs, divergence guard, splice/LLM edit data, cross-model and held-out evals"},
    {"id": "d22e", "start": "2026-10-05 03:44", "end": "2026-10-05 03:58", "focus": "ROC/FPR explainer; Pangram report's AUROC/FPR inconsistency"},
    {"id": "67fb", "start": "2026-10-05 04:17", "end": "2026-10-05 04:38", "focus": "H200 storage, Space helper recovery out of /tmp"},
    {"id": "25a9", "start": "2026-10-05 04:33", "end": "2026-10-06 05:10", "focus": "Text extraction artifacts, positioned-clean-v2, MinerU, baseline omissions and LLM-judge hill-climb"},
    {"id": "3b26", "start": "2026-10-05 08:06", "end": "2026-10-06 06:32", "focus": "ICLR 2027 scoring at scale, 1%-FPR calibration, year diagnostic, OpenReview sample, Atlas publication and reader UX, sharing"},
    {"id": "33a3", "start": "2026-10-06 02:01", "end": "2026-10-06 06:27", "focus": "Soft n-gram / clause labeling test (Luna splitter, spaCy, synthetic edits)"},
]

GOALS = [
    ("Core research", "Detect AI-written text in research papers at document, sentence and token level while keeping false positives on human writing very low. The hardest case is catching one- or two-sentence AI edits inside otherwise human papers."),
    ("Training data", "Close the data gap that limits small-edit detection (only 2 of 24,000 stage-2 windows had an AI span under 200 characters), and grow trustworthy human paper text without teaching a formatting shortcut."),
    ("Text extraction", "Produce clean, positioned paper text from PDFs (the 'baseline': text-layer extraction plus cleanup), with figures, tables, garbled math and references removed and every omission marked."),
    ("Evaluation", "Report recall at a fixed 1% human false-positive rate, with ROC curves, held-out sets and enough seeds to beat noise, and test against writers other than the training generator."),
    ("Model and recipe", "Pick a backbone and LoRA recipe that is stable and cheap to iterate on; test whether larger models (MoE) or longer training help."),
    ("Paper Atlas", "Publish classifications on the Paper Atlas website with PDF overlays, calibration context and a reader that is pleasant to use, as a public version for collecting feedback before the announcement."),
    ("Infrastructure", "Run reliably across the HF Space (8×A100), the shared H200 node and Cloudflare R2 without losing work to restarts, quotas or rate limits."),
    ("Openness", "Keep the repo public-ready (no credentials, addresses, paper text or per-paper scores) and share work with a collaborator."),
]

FINDINGS = [
    ("Training data was the main limit on small-edit detection", "Splicing single AI sentences into human passages lifted small-edit recall at 1% FPR from 0.25 to 0.58–0.63 (Qwen3.5-4B, 3 seeds). Recipe changes had stayed at 0.21–0.25.", "2026-10-06 05:52"),
    ("The learning-rate schedule caused the collapses, not the recipe", "The 5e-4 constant schedule without warmup collapsed around step 1,000 (4B and 9B); the same peak with warmup and cosine was fine. Recipe A (2e-4 cosine, 6% warmup) became the default.", "2026-10-05 13:22"),
    ("Longer training helps paper rewrites, not small edits", "Full-length 4B reached 0.89 on paper_v3 vs 0.77 at 20% length; small-edit recall stayed at 0.21–0.24.", "2026-10-05 19:47"),
    ("MoE: short run best, long run unstable", "The 20%-length Qwen3.6-35B-A3B scored all-edit 0.84 and small-edit 0.30; the full-length run at 2e-4 diverged, and the stable 1e-4 rerun was worse than the 20% run.", "2026-10-06 05:31"),
    ("No evidence of an extraction-artifact shortcut", "A shortcut here means a feature that predicts the label but is unrelated to authorship. Cleaning human text did not change FPR; adding extraction artifacts to AI text lowered recall only slightly (−0.05 with 10× the artifacts). 0 of 1,074 math sentences were flagged.", "2026-10-06 04:45"),
    ("Thresholds set on ordinary human papers do not hold on harder human text", "Cutoffs set on ordinary human papers flag 4.5–7% of human sentences next to edits and 16–29% of human paper_v3 sentences.", "2026-10-05 18:13"),
    ("Fast10 4B calibrated at 1% FPR", "Sentence threshold 0.0283 (held-out FPR 0.83%); catches 84% of one-sentence, 86% of two-sentence and 97% of paragraph edits. Rests on 215 pre-2023 papers; document-level threshold rests on 2.", "2026-10-06 00:10"),
    ("ICLR flag rates rise sharply by year", "Median share of sentences flagged: ≤2022 0.5%, 2023–2024 ≈0.7%, ICLR 2025 1.5–1.8%, ICLR 2026 14.5%, ICLR 2027 68%. Whether 2027 reflects AI use or a difference in submission format, or in topic and writing style over time, is unresolved.", "2026-10-06 01:55"),
    ("Baseline omissions now remove most noise", "On 100 random pages, noise left fell from 175 to 22 per 1k words (references now removed) with prose loss ~3 per 1k; clean pages rose from 16% to 45%.", "2026-10-06 04:29"),
    ("Bulk transfers must be bundled", "Per-file uploads stayed at ~8 MB/s regardless of parallelism; one stream reached 47 MB/s and four reached 88 MB/s. Bundles moved 27k files in 126 s.", "2026-10-06 03:15"),
]

# area, when (UTC), name, details, result, status
EXPERIMENTS = [
    ("Models", "2026-10-04 02:46", "Six-backbone comparison (fast10)", "ModernBERT, Ettin-1B, Qwen3.5-4B/9B, Gemma-4-12B, Qwen3.6-35B-A3B; five at 10% length", "Decoders beat encoders on public text by ~0.16 AUROC; paper AUROC tied (MoE 0.970). Sentence recall@1% FPR (1-sent/2-sent/para): MoE 32/46/95%, 9B 25/37/91%, 4B 18/18/44%.", "done"),
    ("Data", "2026-10-05 05:05", "Training-window audit", "24,000 stage-2 windows", "Only 2 windows contain an AI span under 200 characters.", "done"),
    ("Models", "2026-10-05 19:47", "Overnight LoRA sweep (arms A–L)", "Qwen3.5-4B/9B, 20% length, up to 3 seeds; A100 + H200", "A (2e-4 cosine) all-edit 0.770, small 0.249. 5e-4 constant collapses; standard AdamW (G) 0.515; short-span oversampling (L) no help. Small-edit recall stays at 0.21–0.25.", "done"),
    ("Evaluation", "2026-10-05 18:13", "Calibration transfer test", "16 checkpoints on calibration windows", "Cutoffs flag 0.2–1.6% of ordinary human papers but 4.5–7% of human sentences beside edits.", "done"),
    ("Evaluation", "2026-10-05 20:01", "Held-out 1% FPR, matched length", "cutoff fit on dev, applied to test", "Best now vs Oct 3: all-edit 0.78→0.80, small 0.22→0.26, paper_v3 0.59→0.87.", "done"),
    ("Models", "2026-10-05 21:39", "MoE 20% length (moe-A-s1)", "1 H200, 62 min", "All-edit 0.84, paragraph 0.98, small 0.30 (1 seed).", "done"),
    ("Models", "2026-10-06 02:25", "MoE full length at 2e-4", "2 H200s, data parallel", "Diverged near step 950 (loss peak 1.4); small-edit 0.14. ~3 h of H200 time lost.", "failed"),
    ("Infra", "2026-10-06 02:55", "Divergence guard replay", "all past H200 loss logs", "Flagged all 5 blown-up runs within 20–60 steps; 0 false alarms on 30 healthy runs.", "done"),
    ("Models", "2026-10-06 05:31", "MoE full length at 1e-4", "2 H200s, guard on", "Stable, but small-edit 0.16 dev / 0.21–0.23 test, worse than the 20% run.", "done"),
    ("Infra", "2026-10-06 05:31", "A100 MoE benchmark (FSDP2)", "2 A100s", "7.4 s/step at micro-batch 16; full run ≈5.1 h on 2 A100s vs 2.6 h on 2 H200s.", "done"),
    ("Evaluation", "2026-10-06 04:45", "Artifact-shortcut test", "4B and 9B; human text cleaned, AI text given extraction artifacts", "No FPR change from cleaning human text; adding artifacts to AI text lowers small-edit recall by up to 0.05.", "done"),
    ("Data", "2026-10-06 05:52", "Splice wave 1 (SPH, SPG)", "4B, 20% length, 3 seeds each", "Small-edit recall 0.249 → 0.582 (SPH) / 0.629 (SPG); costs: standalone rewrites 0.51 → 0.36 / 0.26, public AUROC (SPG) 0.911.", "done"),
    ("Data", "2026-10-06 05:21", "Luna LLM sentence edits v1", "5,370 calls, Flex tier", "5,000 accepted for $0.46; median span 131 characters.", "done"),
    ("Evaluation", "2026-10-06 06:05", "Cross-model eval (heterogeneous-ai-spans)", "both MoE runs", "Sentence AUROC 0.87–0.97 across six writers; recall@1% understated because 8% of human sentences inside mixed documents are flagged, mostly next to AI spans.", "rerunning"),
    ("Data", "2026-10-06 06:10", "Wave 2 (LLE, MIX, Arep)", "4B, 20% length, 1 seed each", "Results pending.", "running"),
    ("Evaluation", "2026-10-06 06:26", "Multi-writer held-out edit set", "Claude subagents + Luna", "1,185 Claude edits (Opus 349, Sonnet 335, Haiku 201) + 300 Luna; scoring of all models queued.", "running"),
    ("Data", "2026-10-06 06:20", "Clause-labeling test (soft n-gram labels)", "Luna vs spaCy splitters, synthetic edits", "Synthetic v2 macro-F1: Luna clauses 0.953, sentences 0.971 (v2 biased to sentences); v3 with clause edits running.", "running"),
    ("Data", "2026-10-05 23:40", "N-gram dashboard", "all human vs AI pools, hashing counter", "11 groups in under 2 min; soft-label explainer added.", "done"),
    ("Extraction", "2026-10-05 05:02", "Extraction artifact scan", "31,211 papers", "11 artifact classes (e.g. ICLR review line numbers in 100% of ICLR 2027).", "done"),
    ("Extraction", "2026-10-05 07:03", "positioned-clean-v2 + backfill", "41,004 + 1,414 ICLR 2027 papers", "Cleaned in ~28 min; 59.5M line-number words removed; 4 papers failed extraction.", "done"),
    ("Extraction", "2026-10-05 20:22", "MinerU2.5-Pro", "A100, vLLM", "First full run lost ~10 h to OOM; coverage check 98.3% median; judged too slow; kept a 300-paper, 9-page subset.", "stopped"),
    ("Extraction", "2026-10-06 04:29", "Baseline omissions hill-climb (v4→v6→math policy→references)", "Sonnet judges, 176 pages", "Noise left 175 → 22 per 1k; prose loss ~3 per 1k; clean pages 16% → 45%.", "done"),
    ("Classification", "2026-10-05 10:15", "Inference throughput benchmarks", "A100, 256 windows", "9B 9.7 → 11.3 win/s (merged LoRA); 4B 14.4 → 23.0 win/s (merged + torch.compile). Batch size had no effect.", "done"),
    ("Classification", "2026-10-05 23:30", "ICLR 2027 scoring (fast10 4B)", "42,419 papers; 1 GPU, then 8", "Done after a Space restart; 8-GPU claim queue finished the last 24k papers in ~2 h.", "done"),
    ("Evaluation", "2026-10-06 00:10", "1%-FPR calibration and ROC (fast10 4B)", "215 human papers, eval suite", "Threshold 0.0283, held-out FPR 0.83%; 0.1% threshold unstable; document level rests on 2 papers.", "done"),
    ("Evaluation", "2026-10-06 01:55", "Year diagnostic", "7,102 papers, 2023–2026 + downloads", "Flag rates flat through 2024, rising from 2025; ICLR 2026 14.5%, ICLR 2027 68%.", "done"),
    ("Extraction", "2026-10-06 01:12", "OpenReview ICLR 2024/2025 sample", "71 batch requests, Space extraction", "3,477 new papers extracted and cleaned with positioned-clean-v2.", "done"),
    ("Site", "2026-10-06 03:30", "Atlas publication", "42,418 detail files, catalogue update", "ICLR 2027 classified as 'Qwen3.5-4B · Experimental'; Calibration page live.", "done"),
    ("Site", "2026-10-04 00:17", "Atlas redesign and performance", "Cloudflare Workers", "Redesign and mobile fixes shipped; live search still 2.5–5 s.", "partial"),
]

# label, when (UTC), session, detail
ACCEPTED = [
    ("Git repo, public-ready but private until announcement; scrub from first commit", "2026-10-04 01:36", "2f2a"),
    ("Sentence-level ROC curves on the results page", "2026-10-05 04:08", "9068"),
    ("Monitor runs every 15 min; one-minute sanity check after launch", "2026-10-05 18:55", "25a9"),
    ("Pause runs immediately to free A100 GPUs 2 and 3", "2026-10-05 07:39", "9068"),
    ("Stride 510 and merged LoRA for ICLR scoring", "2026-10-05 10:05", "3b26"),
    ("Parallelize ICLR scoring across all 8 GPUs with an interruptible queue", "2026-10-05 21:20", "3b26"),
    ("Flag at 1% human FPR with ROC curves as standard practice", "2026-10-06 00:05", "3b26"),
    ("Raw probabilities plus calibration metadata instead of rescaled scores", "2026-10-06 02:55", "3b26"),
    ("RAM disk (/dev/shm) as H200 storage for re-downloadable files", "2026-10-06 00:19", "9068"),
    ("Rerun MoE at LR 1e-4 with divergence guard; auto-rollback option", "2026-10-06 02:43", "9068"),
    ("Data priorities: splice edits, artifact-shortcut test, LLM sentence edits", "2026-10-06 04:07", "9068"),
    ("Math policy for omissions; remove references entirely; span-anchored judge metrics", "2026-10-06 03:22", "25a9"),
    ("Bundle bulk transfers (your idea); downloads run on the Space", "2026-10-06 03:20", "3b26"),
    ("Checkpoint policy: one resumable checkpoint during a run, final weights after scoring", "2026-10-05 20:58", "9068"),
]

OVERRIDDEN = [
    ("Claude recommended keeping ICLR 2027 flags private until drift was understood", "You published them publicly as 'Experimental' to iterate on the UX", "2026-10-06 02:35", "3b26"),
    ("Claude suggested rescaling scores to the human-FPR scale", "You questioned it as possibly post-hoc; raw scores were kept", "2026-10-06 02:55", "3b26"),
    ("Claude flagged that pushing to a public repo publishes before the announcement", "You pushed the branch woog97/workbench to rain-1/pangram-at-home", "2026-10-06 06:20", "3b26"),
    ("Claude argued against a VLM parser (MinerU) as a text source", "You ran MinerU anyway, then judged it too slow and cut it to 300 papers", "2026-10-05 06:53", "25a9"),
    ("Claude proposed re-adding the 9B curve on the baseline recipe", "\"i dont trust u, i trust john schulman\": stayed on LoRA Without Regret (which later collapsed on its schedule)", "2026-10-05 07:23", "9068"),
    ("Claude wanted 3 seeds for wave 2", "You cut to 1 seed to get the comparison artifact sooner", "2026-10-06 06:09", "9068"),
    ("Claude proposed an API budget for the multi-writer eval set", "You chose Claude subagents on your account", "2026-10-06 05:53", "9068"),
    ("Claude proposed exact 10%-length runs to separate length from recipe", "You chose a 20% MoE run instead", "2026-10-05 20:18", "9068"),
    ("Claude recommended pausing or shrinking the Space to save cost", "GPUs are a community grant; only storage costs money", "2026-10-05 20:48", "9068"),
    ("Claude suggested more GPUs for MinerU", "You cut scope to 9 pages × 300 papers", "2026-10-05 20:19", "25a9"),
    ("Claude deleted MoE base weights after a run", "You objected; weights restored; delete base weights only on explicit request", "2026-10-05 21:56", "9068"),
    ("Claude launched a full classification run before benchmarking", "You asked to benchmark small batches first", "2026-10-05 09:00", "3b26"),
]

# Flags Claude raised that got no answer and are still relevant. severity: high / medium / low
IGNORED = [
    ("Publishing per-paper AI scores, PDF redistribution licensing, unverified-permission passages, open weights", "Raised before the repo existed; per-paper ICLR 2027 scores are now public on the Atlas.", "2026-10-04 01:29", "2f2a", "high"),
    ("Project name close to Pangram Labs; workbench imitates their dashboard and brand", "Matters before the announcement.", "2026-10-04 01:41", "2f2a", "high"),
    ("Is the AI-side text run through the same omission step?", "If only human text gets markers, the model can learn a formatting shortcut.", "2026-10-06 03:22", "25a9", "high"),
    ("Credential rotation: the Jupyter token (Oct 5, 00:44 PDT) and HF token (Oct 5, 17:50 PDT) were printed in session output", "Rotation never confirmed.", "2026-10-06 00:50", "3b26", "high"),
    ("ICLR 2027 drift checks: read top-flagged sentences; compare ICLR 2026 submission versions", "Needed before treating 68% as AI use.", "2026-10-06 01:55", "3b26", "high"),
    ("Add to the human calibration set from ~20k 2019–2022 Atlas PDFs", "Document-level threshold rests on 2 papers; 0.1% threshold unstable.", "2026-10-06 02:40", "3b26", "medium"),
    ("PyMuPDF AGPL licence risk for the omissions code", "Raised twice.", "2026-10-06 00:03", "25a9", "medium"),
    ("Pre-2022 arXiv LaTeX / PMC as the top human data source", "Proposed a 10k-paper first batch.", "2026-10-06 05:11", "9068", "medium"),
    ("Build the --fsdp A100 MoE training path", "Benchmark done; ~2 h of work.", "2026-10-06 05:32", "9068", "medium"),
    ("fp32 master weights if instability persists", "Trainable weights are BF16 without an fp32 copy.", "2026-10-05 07:24", "9068", "low"),
    ("Manual review of omission pages in the review dashboard", "No verdicts saved yet; 'good enough' decision pending.", "2026-10-06 04:08", "25a9", "medium"),
    ("Do paired passages label the whole rewritten span as AI? Compare to soft labels", "data.py not checked.", "2026-10-06 02:03", "33a3", "medium"),
    ("What is the n-gram work for: exploration or detector features?", "", "2026-10-05 23:45", "9068", "low"),
    ("Storage cleanup approvals: ~79 GB old sweep checkpoints in the bucket, 9.3 GB on the H200, 67 GB MoE copy in Space /tmp", "", "2026-10-06 05:31", "9068", "low"),
    ("Run one agent at a time on shared files and GPUs", "Parallel sessions interfered with each other several times (duplicate GPU runs, outdated CLAUDE.md, uncommitted files from several sessions in one working tree).", "2026-10-03 23:54", "f0fb", "medium"),
    ("Calibrated colour bands per model in the reader and list", "Raw-score bands colour most flagged sentences red and some green.", "2026-10-06 03:35", "3b26", "low"),
    ("Retry the ICLR round-3 loader (stale lock on the Space)", "Likely superseded by the complete ICLR 2027 dataset; confirm and close.", "2026-10-04 00:34", "9068", "low"),
]

# title, area, last touched (UTC), state, next step, priority
THREADS = [
    ("Is ICLR 2027's 68% real AI use?", "Evaluation", "2026-10-06 01:55", "open", "Read top-flagged sentences in 20 papers; score ICLR 2026 submission versions (~20 batch requests).", "high"),
    ("Multi-writer held-out scoring", "Evaluation", "2026-10-06 06:26", "running", "Score SPG, SPH, wave 2 and both MoE runs; report per writer.", "high"),
    ("Wave 2 results (LLE, MIX, Arep)", "Data", "2026-10-06 06:10", "running", "Compare against splice wave 1 at 1% FPR; decide the next data mix.", "high"),
    ("Republish the comparison artifact", "Evaluation", "2026-10-06 06:15", "open", "Add corrected cross-model numbers, wave 2 and per-writer held-out results.", "high"),
    ("Same omission treatment for AI-side text", "Extraction", "2026-10-06 03:22", "open", "Decide whether AI text goes through the omission step before using omitted human text in training.", "high"),
    ("Apply references removal and omissions to the corpus", "Extraction", "2026-10-06 04:29", "open", "Regenerate clean datasets; run the full-corpus omission pass (~2 h CPU).", "medium"),
    ("Omission regressions and leaks", "Extraction", "2026-10-06 04:29", "open", "Fix the small-caps heading regression, pseudocode-as-math losses, table cell leaks, caption cut.", "medium"),
    ("Code-block and algorithm policy", "Extraction", "2026-10-06 02:31", "open", "Decide whether code listings get a ⟦code omitted⟧ marker.", "low"),
    ("Manual 'good enough' review of omissions", "Extraction", "2026-10-06 05:06", "waiting on you", "Save verdicts on ~10 random pages in the review dashboard.", "medium"),
    ("Revisit overnight A vs B conclusions", "Models", "2026-10-06 02:55", "open", "Re-read arm comparisons excluding diverged 5e-4 seeds.", "medium"),
    ("Add human calibration data", "Evaluation", "2026-10-06 02:40", "open", "Extract ~2,000 of the 2019–2022 Atlas PDFs; recalibrate document and 0.1% thresholds.", "medium"),
    ("False positives next to AI spans", "Evaluation", "2026-10-06 06:05", "open", "Measure how often human sentences are flagged at each distance from an AI span.", "medium"),
    ("Clause-labeling v3", "Data", "2026-10-06 06:24", "running", "Score v3 with 0.6B embeddings; rerun spaCy units; 4B embeddings when a GPU frees.", "low"),
    ("FSDP MoE training on A100s", "Infra", "2026-10-06 05:32", "waiting on you", "Approve building --fsdp (8-bit optimizer wrapper, sharded save and validation).", "low"),
    ("Exact 10%-length runs", "Models", "2026-10-05 20:01", "stale", "Run if length vs recipe still matters; otherwise close.", "low"),
    ("Ettin sentence curves and MoE vs 9B bootstrap", "Evaluation", "2026-10-05 04:17", "stale", "Low value now; close unless needed for a write-up.", "low"),
    ("9B scoring of ICLR 2027", "Classification", "2026-10-05 10:20", "deferred", "You skipped it; revisit with the splice-trained model instead of fast10.", "low"),
    ("Calibrated colour bands in the reader", "Site", "2026-10-06 03:35", "open", "Per-model bands and an FPR-labelled threshold slider.", "medium"),
    ("Atlas search latency", "Site", "2026-10-04 00:17", "stale", "Stream the 80 MB index or move search out of the Worker; live search is 2.5–5 s.", "medium"),
    ("Commit and consolidate code", "Infra", "2026-10-06 06:20", "open", "Land today's work on woog97/paingram main; one pipeline tool, shared helpers, tests.", "medium"),
    ("Credential rotation", "Infra", "2026-10-06 00:50", "waiting on you", "Rotate the HF token and the Jupyter token.", "high"),
    ("Storage cleanup approvals", "Infra", "2026-10-06 05:31", "waiting on you", "Approve or decline the bucket, H200 and Space /tmp deletions.", "low"),
    ("Calibration-window overlap with the eval suite", "Evaluation", "2026-10-04 01:31", "stale", "Probably handled by calibration-exclude-papers.json (Oct 5); verify and close.", "low"),
    ("Small-caps stray space and NeurIPS checklist spot-check", "Extraction", "2026-10-05 19:24", "stale", "Fix 'LOW -RANK' joins; spot-check ~18.7k removed checklist words.", "low"),
]

# category, title, when (UTC), cost, fix, recurrences
ISSUES = [
    ("Storage", "Space restarts wiped /tmp", "2026-10-05 20:48", "All A100 sweep checkpoints and Space tools lost; ~75 min of baseline reruns; classification runtime rebuilt", "Persistent helpers in ~/.config, vendored packages on /data, checkpoints to the bucket as each run finishes", 3),
    ("Storage", "Shared H200 quota exhausted", "2026-10-05 23:39", "curve-9b-B killed at 36%; moe-A-full crashed mid-run (~45 min)", "RAM disk for re-downloadable files; measure real free space before large writes", 2),
    ("Training", "Loss divergence detected late", "2026-10-06 02:43", "~3 h of H200 time on a damaged MoE run; detection ~1.5 h late; pre-spike weights lost", "Divergence guard, rolling snapshots, loss-aware health checks, auto-rollback", 5),
    ("Transfer", "Per-file transfers and polling hit rate limits", "2026-10-05 07:15", "~30 min Space 429 lockout; ~25 min slow R2 uploads; 18 GB downloaded to the Mac and 19 GB uploaded again instead of transferring directly", "500 requests / 5 min budget; bundle everything; run downloads on the Space", 3),
    ("Extraction", "MinerU settings changed after calibration", "2026-10-05 18:34", "~10 h of failed GPU work (21,001 OOM errors)", "Never change settings between calibration and the full run; failure-rate breaker", 1),
    ("Environment", "Missing tools after restarts", "2026-10-06 01:00", "Repeated stalls: poppler, tesseract, pypdf, Pillow, nvcc, OpenCV, transformers versions, W&B key, HF token", "Pinned micromamba envs, vendored packages, keys passed via env at launch", 8),
    ("Security", "Credentials printed in session output", "2026-10-06 00:50", "Jupyter token and HF token exposed in transcripts", "Redaction and compile-check-before-send rules; rotation still pending", 2),
    ("Analysis", "Wrong claims later corrected", "2026-10-06 04:32", "Misleading numbers on the results page and in CLAUDE.md for hours", "Corrected: edit-catch rate, MinerU text loss, coverage counts, '8–12 pt seed noise', cause of slow uploads", 6),
    ("Coordination", "Parallel sessions interfered with each other", "2026-10-05 21:10", "Duplicate GPU runs, misread GPU assignments, outdated CLAUDE.md reads, uncommitted files from several sessions in one working tree", "Session handoff messages; per-thread scope; separate git index for commits", 4),
    ("Tooling", "Blocked tool calls and permission prompts", "2026-10-06 06:20", "Blocked publishes, deletes, pushes and launches needed manual approval; sleep blocked; browser extension unavailable", "Manual mode for approvals; background watchers; headless browser via puppeteer", 9),
    ("Process", "Jobs launched before checks were in place", "2026-10-05 23:45", "Unbenchmarked launch; prototype without a progress counter; watchers that only checked completion", "Benchmark first; 1/2/4/…/60-min backoff checks; loss-aware monitoring", 3),
    ("Process", "Annotation effort abandoned", "2026-10-06 05:03", "300-sentence gold set stopped after ~15", "Synthetic known-truth edits instead", 1),
    ("Site", "Reader layout bugs", "2026-10-06 03:40", "Page jumps broken, highlights hidden on Discover, page-strip marks collapsed", "Bounded reader height, full reader on Discover, gradient strips, follow-reading panel", 3),
]

RULES = [
    ("Communication", "Keep answers short and direct; when asked for status, give status, not results."),
    ("Communication", "Report times in PDT."),
    ("Evaluation", "Flag at 1% human FPR with ROC curves; never present 0.5-cutoff labels as classifications."),
    ("Evaluation", "Papers from 2022 or earlier are human-written; don't argue it."),
    ("Evaluation", "Distrust aggregates; validate visually and with frozen, random held-out sets."),
    ("Training", "Follow LoRA Without Regret where possible; present evidence when data disagrees."),
    ("Training", "BF16 for inference, trainable weights and checkpoints; no fp32 copies."),
    ("Training", "During a run keep one resumable checkpoint; after scoring keep final weights only; delete base weights only on explicit request; never delete the MoE base weights."),
    ("Operations", "Benchmark throughput on small batches before launching large runs."),
    ("Operations", "Check every job at 1, 2, 4, 8, 16, 32 minutes, then hourly; watch loss, not just liveness; fix and reset on stalls."),
    ("Operations", "Bundle bulk transfers; stay under 500 HF requests per 5 minutes; run downloads on the Space."),
    ("Operations", "Download new model weights only on the Space (exceptions recorded in AGENTS.md); use /dev/shm on the H200 for re-downloadable files."),
    ("Operations", "Ask before deleting anything on the Space or launching on shared hardware beyond what was agreed."),
    ("Security", "Never print or commit credentials; compile-check credential-bearing remote code; keep helpers and keys in ~/.config/pangram."),
    ("Openness", "Commits must be public-ready: no credentials, infrastructure addresses, third-party paper text, per-paper scores or brand assets."),
    ("Data", "OCR only as a fallback for garbled pages; omit figure text and keep captions; mark every omission; remove references from classification input."),
    ("Data", "LLM labelling on Luna (Flex tier) with spend caps and a pilot first; eval edits via Claude subagents."),
]

LINKS = [
    ("Paper Atlas (live)", "https://pangram-paper-atlas.woog09.workers.dev"),
    ("Calibration page", "https://pangram-paper-atlas.woog09.workers.dev/?view=calibration"),
    ("Backbone and sweep results", "https://claude.ai/artifact/QJJUjcHwHNmXuqFNKX28QQ"),
    ("Model comparison (latest)", "https://claude.ai/artifact/1Wey5VmYi8yN2gYgZJ7Zjy"),
    ("Fast10 4B calibration", "https://claude.ai/artifact/4FNSVgD1vH8i87w4XYpq7D"),
    ("Omissions review dashboard", "https://claude.ai/artifact/5KBFUaGaYrmcisSiSNUqxc"),
    ("Text layer vs MinerU", "https://claude.ai/artifact/WJFxShnEYkfVkSer1d82cb"),
    ("N-gram dashboard", "https://claude.ai/artifact/WPLSuRnVBwgjoS9uHm5GcP"),
    ("Clause annotation page", "https://claude.ai/artifact/YVG8QfNMXZBKEU6rdZsQna"),
    ("Shared branch", "https://github.com/rain-1/pangram-at-home/tree/woog97/workbench"),
]

# What each owner item needs: a decision (policy choice), an approval (Claude can do it once approved), or an action (only you can do it)
OWNER_TYPE = {
    "Publishing per-paper AI scores": "decision",
    "Project name close to Pangram Labs": "decision",
    "Is the AI-side text run through the same omission step?": "decision",
    "Credential rotation": "action",
    "ICLR 2027 drift checks": "approval",
    "Manual 'good enough' review of omissions": "action",
    "FSDP MoE training on A100s": "approval",
    "Storage cleanup approvals": "approval",
}
def owner_type(text):
    return next((v for k, v in OWNER_TYPE.items() if text.startswith(k)), "decision")
