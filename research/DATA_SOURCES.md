# Data sources: what to train on, what to evaluate on, and why

This is the single place for data decisions in the merged workbench. Each verdict links to the evidence behind it. Update this file whenever a verdict changes, with the date and the evidence.

Verdicts:
- **train**: use in detector training.
- **train (capped)**: train, at the stated dose.
- **eval**: held-out evaluation only.
- **stress**: report separately; never use to set thresholds.
- **drop**: do not use.

Evidence keys:
- `woog` = `research/README.md`, `research/human-source-pool/`, `research/TRAINING_DATA_INVENTORY.md`, `benchmarks/pangram4/training/overnight-sweep-20261004/NOTES.md`.
- `r1` = rain1's reports in `research/span-detection-20260928/reports/`.
- `mix` = `research/data-mix-20261006/README.md` (the 2026-10-06 analysis, with exact token statistics and leakage checks).
- `v3` = rain1's evaluation repo `pangram-at-home-eval`, `artifacts/benchmark-v3` (frozen 2026-10-06).

## Current training mixes

In woog's trainer, `stage2-epochN` files are **shards** (fresh draws), not repeated passes. See `benchmarks/pangram4/training/hparam-sweep-20261006/README.md`.

| Mix | Contents | Where |
|---|---|---|
| woog `prepared-v2` (base) | Per stage-2 epoch: mirrors 6,000, papers 6,000, GRADTEX 4,800 (document labels only), human 6,000, full papers 1,200 | Training Space, `backbone-launch-20261003/runs/<model>/prepared-v2` |
| `qwen36-35b-a3b-mix` | Base plus ~2,740 windows per epoch (+11.4%): hetero mixed 1,450, hetero controls 1,020, v14 GRADTEX spans 85, NASA/NOAA human 184 | Built by `benchmarks/pangram4/training/hparam-sweep-20261006/setup_mix.py` from `research/data-mix-20261006/mix-spec.json` |

## Sources

| Source | Verdict | Licence | Notes and evidence |
|---|---|---|---|
| GPT-6 Luna synthetic mirrors (woog) | train | project-generated | The base's main AI source. **Nearly every AI token in the base comes from this one generator**, which is a known weakness (woog NOTES). |
| Paper paired edits, `paper_v3` (woog) | train | per paper | Paragraph-scale replacements: median AI span about 550 chars, almost none under 200. Hence the small-edit recall plateau of 0.21–0.27 (woog NOTES; mix §2). |
| Curated human pool `human-source-mix-v1` (woog) | train | commercial-use screened | 34–48 sources across 9 categories. It already contains ASAP2, PERSUADE, Stack Exchange and Gutenberg, so do not add those again from v14 (mix §3). |
| GRADTEX, document labels (woog) | train | per GRADTEX | Built on MAGE seeds; boundaries are inferred, so it uses document labels only. |
| **heterogeneous-ai-spans** v1.3.0 (rain1), JMLR / Gutenberg / Standard Ebooks / Hansard | **train (core addition)** | mixed-source terms (`LICENSE.md`); JMLR licence unconfirmed | Coherent in-place replacements by 6 frontier writers (Haiku 4.5, Sonnet 5.5, Opus 5.5, Opus 3, GPT-6.1 Sol, GPT-6 Luna), with matched human controls. Use only the mixed/AI windows of mixed documents, because their human windows duplicate the controls. Spans are paragraph-sized (median ~1,300–1,700 chars), so this source **does not fix small edits**. Opus 5.5 appears only in fiction. Its val/test splits are held out (mix §2, §5). |
| heterogeneous-ai-spans, Beige Book cohort | drop (train); stress (eval) | public domain | rain1 considers it confounded: formulaic Fed prose repeated across 12 districts, and it borders on 2022. 73/173 pairs share boilerplate with hetero val/test and v3. |
| heterogeneous-ai-spans, WikiText cohort | drop | CC BY-SA (version discrepancy) | Overlaps v3's `wikitext_detok` panel (16 documents share text; same source) (mix §3). |
| v14 GRADTEX span rows (rain1) | train | per GRADTEX | Preserved-context completions from gemma-4-31b and mistral-small-3.2. The only supply of sentence-scale and mid-sentence spans after the drops (mix §4). |
| v14 NASA/NOAA science articles (rain1), human side | train | public domain | Hard negatives behind rain1's largest false-alarm drop (external articles 77→39 of 150; `r1 science_paired_v9_comparison.md`). The AI side was dropped: 1.7B/3B generators, partly noncommercial. |
| v14 LLMTrace | drop while v3 evaluates it | per LLMTrace | The best short-span and multi-generator source, but 46 rows (21 exact) and 42 topic groups overlap v3's LLMTrace panels. rain1 capped it at 3% (`r1 span_external_sources_v5.md`). |
| v14 MAGE (9 domains) | drop | Apache-2.0 plus upstream | Label noise: 3,843 of 4,801 AI rows are continuations whose human opening sentence is labelled AI, and 176 of those openings match rows labelled human elsewhere. 1,876 rows come from models under 3B. MAGE is a v3 reserve set (mix §4). |
| v14 ACL / PMC joins | drop | CC BY | Synthetic joins of unrelated excerpts, 27% cut mid-sentence, written by 0.5B–1.7B models. Redundant with the base papers (`r1 span-training-data-v4.md`; mix §4). |
| v14 Common Pile news | drop | CC BY | Made false alarms worse in v8: 64→77 of 150 external human articles (`r1 publication_hardneg_v8.md`). |
| v14 ASAP2 / PERSUADE | drop (redundant) | ASAP CC BY; PERSUADE CC BY-NC-SA | Already in woog's pool. Overlaps v3's `persuade` panel (79 / 77 rows). |
| Dolly 15k | drop | CC BY-SA | Its "no generative AI" instruction cannot be verified. It was a frequent false-alarm source (9–15 of 150), in the same class as No Robots (mix §4). |
| Writers Stack Exchange (pre-2023) | eval / via woog pool | CC BY-SA | Redundant with woog's Stack Exchange share. |
| **Beemo human texts** | drop | — | rain1's evaluations (2026-10-06): human texts unreliable. Not in any training mix. |
| **No Robots** | drop | — | rain1's evaluations (2026-10-06): human texts unreliable. Not in any training mix. |
| Federal Reserve Beige Book (v1 human audit) | stress | public domain | Formerly 320 of 1,120 passages in rain1's threshold-calibration set. **Keep it out of calibration**: it is confounded, and at that weight it skews the operating point. |
| AITDNA | eval | CC BY-SA / GPL-2.0 (card) | Real human–AI collaboration, 362 documents, so its many "views" are not independent samples. Synthetic joins overstate performance: human-token FPR inside mixed documents was 15.7–19.1% for v13/v14 (`r1 real_collaboration_v4.md`). |
| CoAuthor | eval | licence not stated | Short GPT-3 suggestions. Stress test for small edits; also a v3 reserve. |
| Human Detectors external articles | eval (development-exposed) | publishers' copyright | Used repeatedly while tuning v5–v14, so it is not held out. Never train on it or redistribute it. |
| Smithsonian archive | drop | copyright | Research use only (`r1 smithsonian_archive_v10_data_card.md`). |
| DAMASHA | drop | — | 735 rows touch a locked human test set and there are no upstream ids (`r1 span_external_sources_v5.md`). |
| EditLens ICLR | train only for noncommercial models | CC BY-NC-SA | Gated; v14 references its rows by id only. |

## Evaluation sets and what they are exposed to

- **woog frozen suite** (`evaluation-suite-v1-20261003`, HF `woog/ai-paper-workflow-eval`):
  - Profiles: workflow, comparison, assistance and full.
  - The comparison profile is development-exposed.
  - `luna-flex-hillclimb` tuned prompts on its test papers.
- **Sweep evaluation rows** (`sweep-eval-rows.jsonl.gz`, 3,511 rows): split into dev and test halves by paper group. Select on dev and report on test. Note that `sweep_eval.py`'s recall@1% fits its cutoff on the same half it reports, which makes it optimistic.
- **rain1 benchmark-v3** (14,322 examples):
  - A development/regression benchmark that has been inspected repeatedly.
  - It contains LLMTrace, CNN/DailyMail, scientific papers, WikiText, PG-19, IMDb, PERSUADE, AITDNA and Beemo/No Robots panels, plus the heterogeneous-ai-spans **test** split.
  - For any model trained on heterogeneous-ai-spans, its `generate_heterogeneous` panel and the hetero val/test splits are **in-distribution**. Report them separately and do not treat them as evidence of generalisation.
- **heterogeneous-ai-spans val/test**: held out, and shares no text with the base training data.

## Lessons that hold across both projects

1. **Calibrate at the document level, on hard human text.**
   - A false alarm is any highlight on a human document.
   - Sentence cutoffs fitted on easy human windows realised 2–23% FPR on held-out test (woog NOTES). At 0.8% token FPR, 9 of 100 PMC papers still got a highlight (`r1 two_day_progress`).
   - Use one rule for every model, and report a sweep of target rates.
2. **A hard negative helps only when it is close in style to the actual false positives.**
   - Generic news hurt (v8); paired science and essays helped (v9, v10).
   - Every mix-in moved false alarms somewhere else (v12, v14). Check the broad human audit after each change.
3. **Prefer coherent in-place replacements to synthetic joins of unrelated text.** Join boundaries become a shortcut, and joins overstate real-collaboration performance.
4. **Diversity in name is not diversity in signal.** MAGE's 27 generators are mostly 2020–23 continuation models with prefix label noise.
5. **Small edits need data built for them.** Neither woog's data nor heterogeneous-ai-spans has in-context AI spans under 200 chars. Candidate fix: 1–3-sentence replacements written into the hetero control passages by the same writers.
6. **Evaluation sets get burned.** Track which sets have steered decisions, and keep a confirmation set untouched.
7. **Expert LoRA checkpoints depend on the PEFT version.**
   - woog's H200 runs and the vendored PEFT 0.18.1 on the Space disagree on input versus output axes for fused MoE expert parameters. Same keys, transposed factors.
   - A checkpoint trained under one version does not load under the other. Worse, if the shapes ever matched it would load silently with the wrong update.
   - Record the PEFT version with every MoE checkpoint. To convert, use new A = old Bᵀ and new B = old Aᵀ, and verify numerically.

8. **Formatting is a label leak unless both classes share one convention.**
   - heterogeneous-ai-spans v1.3.0 had CRLF line endings, hard wraps and curly quotes on the human side only (Gutenberg/Standard Ebooks). The v1 mix MoE then flagged 110/120 straight-quoted, unwrapped HAP-E-2 human fiction on benchmark-v3, and 40% of PG-19 calibration humans. v1.4.0 normalises both sides.
   - woog's `prepared-v2` has the reverse cue: Luna mirrors (AI) use curly quotes (31 per 10k chars, straight 0.3), while the human pool uses straight quotes (27) and line breaks (44). This is not yet normalised.
   - Audit every new source with `research/data-mix-20261006/scripts/format_audit_base.py` before training.
