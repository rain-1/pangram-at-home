# Span detection v1–v14 (rain1, September 24–28, 2026)

This is a curated import of rain1's separate span-detection project into the workbench. The full history, about 200 one-off scripts and the PDF renderings of reports remain on the `main` branch of `rain-1/pangram-at-home`. Only the parts the workbench lacks were brought here. The datasets, adapters and score caches stayed outside git under `/mnt/f/pangram-at-home`. The scripts are reference recipes: their data paths point there and are not rewired to the Space.

The two projects converged independently on the same model: a causal LoRA token classifier that reads each window twice (Repeat2), with character-region labels masked to -100 at boundaries. This project used Qwen3-1.7B on Vast GPUs, evaluated against the Pangram EditLens RoBERTa and Llama checkpoints under one threshold rule. The workbench has more scale and backbones. This folder contributes calibration methodology, human hard negatives, evaluation sources and a record of which data mixtures failed.

## 1. Document-level "any highlight" calibration

The workbench overnight sweep found that sentence cutoffs fit on calibration windows realize 2–23% human FPR on the test half (`benchmarks/pangram4/training/overnight-sweep-20261004/NOTES.md`). This project reached the same conclusion earlier and changed the operating rule:

- **Measure what a reader sees.** A human document is a false alarm if *any* part of it is highlighted. Here, a 0.8% token FPR still put a false highlight on 9/100 PMC papers (`reports/two_day_progress_2026-09-24_to_26.md`).
- **Calibrate on hard, separate pure-human documents.** Report token or sentence FPR and document-any FPR separately.
- **Use one rule for every model**, baselines included.
- **Report a threshold sweep.** The apparent v10→v12 change was largely a move along the recall/FPR curve (`reports/span_new_sources_thresholds_v12.md`).

The rule is now integrated: `calibrated_report.py` in the overnight sweep reports `test_human_docs_any_highlight` at every cutoff. It also adds a `document_any` operating point fitted on dev-half pure-human rows and applied to the group-disjoint test half (`test_calibrated_report.py`).

## 2. Human hard negatives

A hard negative helped only when it was stylistically close to the actual false positives, and every added mix-in moved false positives somewhere else. Locked external articles had 150 human documents. The broad human audit had 3,579.

| Version | Data added | External human false alarms | Broad human false alarms | Report |
|---|---|---:|---:|---|
| v6 | Source-balanced mix | 64/150 | | `reports/publication_hardneg_v8.md` |
| v8 | Common Pile news | 77/150 (**failed**) | | `reports/publication_hardneg_v8.md` |
| v9 | Paired NASA, NOAA, Climate.gov and EPA science articles | 39/150 | | `reports/science_paired_v9_comparison.md` |
| v10 | ASAP 2.0 student essays (conservative default) | 16/150 | 9/3,579 | `reports/essay_paired_v10_comparison.md` |
| v13 | Half-dose GRADTEX | 13/150 | 19/3,579 | `reports/span_new_sources_v13.md` |
| v14 | PERSUADE, Writers SE, MAGE CMV/ELI5 | 11/150 | 25/3,579 | `reports/span_hardneg_v14_results.md` |

v14 also raised mixed-span recall: LLMTrace 48.7% → 58.9%. But CNN human false alarms went from 8 to 25 of 500, and LLMTrace human from 5 to 28 of 720.

The workbench human pool already plans ASAP, PERSUADE, Stack Exchange, Standard Ebooks, PMC and ACL. The **paired science articles** are new to it. They are dated, bylined, public-domain government pages with topic-matched AI counterparts, collected by `scripts/collect_science_articles_v9.py` through `build_science_articles_v9_splits.py` (data card `reports/science_articles_v9_data_card.md`).

The Smithsonian magazine set (`reports/smithsonian_archive_v10_data_card.md`) is copyrighted and was for local research only. It does not meet the workbench's commercial-reuse standard, so its collector was not imported.

## 3. Evaluation sources the workbench lacks

- **AITDNA (real human–AI collaboration):** synthetic joins overstate performance. Human-token FPR inside real mixed documents was 15.7% (v13), 19.1% (v14), and 20.5–57.1% for the Pangram checkpoints. AITDNA's many views are the same 362 documents (`reports/real_collaboration_v4.md`).
- **CoAuthor (short GPT-3 suggestions):** directly relevant to the workbench's small-edit recall plateau.
- **LLMTrace mixed (multi-generator):** almost all workbench AI text is from one generator.
- **Broad pure-human audit:** CNN/DailyMail, Beige Book, Standard Ebooks and PMC bodies (`scripts/build_span_human_eval_v2.py`, `reports/span-human-eval-v2.md`).
- **Human Detectors external articles:** 150 human and 150 AI. Now development-exposed; do not report it as held out.

Source normalisation and overlap screening: `scripts/prepare_external_span_sources_v5.py`, `audit_external_span_sources_v5.py` and `audit_span_candidate_intake.py`, using the 24-word phrase fingerprints in `scripts/overlap.py`. These audits found 735 DAMASHA rows touching a locked human test set; do not merge DAMASHA wholesale (`reports/span_external_sources_v5.md`).

Against Pangram, the gap on pure humans is large: 1/3,579 broad-human false alarms against 19–25 here. The Qwen span models localise mixed spans much better: LLMTrace recall 49–59% against 12–22%. The Pangram span scores are window broadcasts.

## 4. Other findings

- **Attribution heads:** a linear head on a frozen detector backbone beats the raw backbone. Arena 50-model top-1 (on the workbench's `woog/arena-prose-100-49-models`) was 48.7% vs 42.7%; four named writers 12/12 vs 11/12. A frozen backbone matched full fine-tuning (`reports/attribution_comparison_v1.md`, `scripts/*attribution*`).
- **Mixture ablation:** removing papers or creative writing cost about 4.5–4.9 points of partial AUC (`reports/ablation_results_v3.md`).
- **Data size:** 5k documents × 4 epochs roughly matched 20k × 1 (`reports/span_size_curve_v5.md`).
- **Stage-1 HPO** on Qwen3-1.7B (24 Ray/ASHA trials): LR 7.6e-5, r32/α64, effective batch 8 (`reports/hpo_results_v3.md`).
- **Fiction candidates:** PAN fanfiction, Beneath Ceaseless Skies, StoriesInTheWild and historical fiction (`reports/fiction-*.md`, `scripts/fetch_fiction_*.py`). Their rights notes are in each report.
- **Licensing:** this project became noncommercial-only on 2026-09-27 because of EditLens and PERSUADE (CC BY-NC-SA). See `notes/datasets.md`. Check each source against the workbench publication plan before training a releasable model.

## Layout

- `reports/` — markdown reports, data cards and their JSON/CSV. Start with `two_day_progress_2026-09-24_to_26.md` and `span_hardneg_v14_results.md`.
- `scripts/` — science-article, fiction, external-source, human-audit and attribution scripts.
- `manifests/` — frozen v1 data pyramids.
- `notes/datasets.md` — source and rights decisions.
- `tests/` — run with `uv run --with numpy --with pytest --with beautifulsoup4 --with lxml --with requests python -m pytest tests`.
