# Length-aware aggregation for shared_windows_v1

Run: `artifacts/runs/ai-detector-panels-v3__577af7a44f`, benchmark `artifacts/benchmark-v3`, protocol `shared_windows_v1`.
Models: moe_base_s1_mean, moe_qwen_t21a2_mean, moe_nemo_t21a2_mean, open_pangram_llama.

Every rule is re-calibrated on the same 3000 human calibration documents the benchmark uses (role=calibration,
authorship=human), with the benchmark's order-statistic rule (threshold = sorted_desc[floor(alpha*n)], flag iff score > threshold).
Metrics are plain proportions over scored evaluation documents in each panel (same denominators as results.summary).

Rules:

- `max`: baseline, document score = max window score, one global threshold.
- `binned_max`: max score with a separate 1% threshold per window-count bin (1, 2-3, 4-7, 8+), merging any bin with fewer than 200
  calibration humans into its neighbour. Only 95 calibration humans have 8+ windows, so 8+ is merged into 4+ (bins 1 / 2-3 / 4+, with 566 / 1863 / 571 docs).
- `binned_max_nomerge`: the same, with 8+ kept as its own bin (95 docs, so k=0 and the threshold is the bin's maximum).
- `per_window_n`: flag iff max > t(n), where t(n) is the per-window threshold with exceedance p_n = 1-(0.99)^(1/n) in the pooled
  window scores of the calibration humans (8,687 windows). n = number of scored windows.
- `top2_mean`: mean of the two highest window scores (max if there is 1 window).
- `mean`: mean over windows.

Files: `lencal.py` (all rules, writes `panel_metrics.csv`, `calibration_thresholds.csv`, `window_counts.csv`, `doc_flags_1pct__*.csv.gz`),
`make_tables.py` (baseline check and tables, writes `tables.md` incl. the 5% operating point), `diag.py` and `diag_subdomain.py` (diagnostics).
`baseline_cells.csv` is the extracted reference slice of `results.summary.cells.csv`.

## Findings

- **Baseline reproduced exactly.** For all 4 models at both 1% and 5%, the recomputed `max` threshold is bit-identical to `calibration.json`.
  All 18 panel-level cells (view=all) match `results.summary.cells.csv` exactly in numerator and denominator.
- **Length alone does not inflate human FPR on the calibration or audited sets.** With the frozen `max` threshold, calibration humans with
  4-7 windows are flagged at 0.0-1.7% and those with 8+ at 0.0-2.1%. audited_human_core 8+ is 0.5-1.6%. Per-window exceedance falls
  with length (e.g. moe_qwen calibration: 0.53% at 1 window, 0.00% at 4-7). So the per-bin thresholds are close to the global one
  (moe_base: 0.00142 / 0.00184 / 0.00183 vs global 0.00159), and `binned_max` barely changes heterogeneous_matched_human_controls:
  47.0 to 43.7 (base), 23.3 to 26.3 (qwen), 21.3 to 19.0 (nemo), 19.7 to 13.0 (llama).
- **heterogeneous_matched_human_controls is mainly a per-window score shift plus a length effect in one subdomain.** Its windows exceed the
  global threshold at 4.1-13.5%, against 0.35-0.38% for calibration windows. By subdomain under `max`: economic_report (49 docs, median 9
  windows) is flagged 65-90% with 11-27% of windows over threshold. ml_paper_body (145 docs, median 3 windows) is flagged 60% for moe_base and 0-12% for the others.
  historical_fiction (106 docs, median 9 windows) is flagged 9-20% with only 1-3% of windows over threshold, which is the pattern a length correction can fix.
- **`per_window_n` is the only rule that cuts long-document human FPR substantially while keeping recall, for qwen and nemo.** On the
  controls panel it gives 23.3 to 7.7 (qwen) and 21.3 to 10.0 (nemo), with historical_fiction 19.8 to 1.9 and 14.2 to 1.9 and
  economic_report 69 to 20 and 65 to 20. Mixed recall on heterogeneous_block_replacement stays at 99.7-100%. Short-document recall rises:
  fully_ai_core 79.2 to 81.3 (qwen) and 79.4 to 82.0 (nemo), controlled_replacement 29.1 to 33.1 and 31.4 to 39.5. Short human panels move
  by 0.2 points or less (e.g. audited_human_core 1.4 to 1.6 for qwen, 1.2 to 1.4 for nemo), except qwen benchmark refs (2.1 to 2.6).
- **The price of `per_window_n` is that it moves the calibration FP budget onto short documents.** Windows within a document are positively
  correlated, so the independence formula is conservative for long documents. For the MoE models, calibration-set FPR is 0.7-1.6% for 1-3 windows and
  0.0-0.4% for 4+ windows. For llama it is 0.5-1.05% in every bin. Overall it is 0.9-1.1%. For moe_base this costs: benchmark_source_matched_human_references 3.8 to 6.4,
  heterogeneous_block_replacement recall 96.7 to 94.3, and the controls panel only drops to 37.0, because ml_paper_body (short) and
  economic_report stay at 54% and 67%. For open_pangram_llama, controls fall 19.7 to 11.3, but block-replacement recall falls 64.0 to 54.0.
  t(n) for n > 14 extrapolates beyond any calibration document (the controls' p90 is 17 windows).
- **`mean` and `top2_mean` are not fixes.** `top2_mean` raises controls FPR for every model (51.3 / 30.3 / 31.7 / 27.3). `mean` reduces it
  (41.7 / 12.3 / 16.7 / 9.7) and gives the largest short-panel recall gains, but it drops llama block-replacement recall 64.0 to 47.0 and raises
  moe_base benchmark-refs FPR 3.8 to 6.9. No rule brings the controls panel near 1%. The economic_report and (for moe_base) ml_paper_body
  controls are scored as AI window by window, which is a source/domain problem rather than an aggregation problem.

## Baseline reproduction (rule `max` vs results.summary.cells.csv)

| model | op | cells | exact_num_den_match | max_abs_diff | threshold_equal |
|---|---|---|---|---|---|
| moe_base_s1_mean | human_fpr_1pct | 18 | 18 | 0.0 | True |
| moe_base_s1_mean | human_fpr_5pct | 18 | 18 | 0.0 | True |
| moe_nemo_t21a2_mean | human_fpr_1pct | 18 | 18 | 0.0 | True |
| moe_nemo_t21a2_mean | human_fpr_5pct | 18 | 18 | 0.0 | True |
| moe_qwen_t21a2_mean | human_fpr_1pct | 18 | 18 | 0.0 | True |
| moe_qwen_t21a2_mean | human_fpr_5pct | 18 | 18 | 0.0 | True |
| open_pangram_llama | human_fpr_1pct | 18 | 18 | 0.0 | True |
| open_pangram_llama | human_fpr_5pct | 18 | 18 | 0.0 | True |

## Window counts per panel (all evaluation docs in panel; calibration_human = calibration set)

| panel_id | n_docs | median_windows | p90_windows | max_windows | frac_ge4 | frac_ge8 |
|---|---|---|---|---|---|---|
| beemo_matched_editing | 400 | 1.00 | 2.00 | 6 | 0.04 | 0.00 |
| lamp_human_edited_ai | 200 | 1.00 | 2.00 | 3 | 0.00 | 0.00 |
| controlled_replacement_noncore | 200 | 2.00 | 3.00 | 15 | 0.09 | 0.01 |
| benchmark_source_matched_human_references | 720 | 2.00 | 4.00 | 21 | 0.27 | 0.02 |
| editlens_in_family_reference | 240 | 2.00 | 4.00 | 6 | 0.28 | 0.00 |
| fully_ai_core | 1500 | 2.00 | 4.00 | 12 | 0.14 | 0.00 |
| fully_ai_noncore | 300 | 2.00 | 3.00 | 7 | 0.08 | 0.00 |
| controlled_replacement | 800 | 2.00 | 3.00 | 7 | 0.04 | 0.00 |
| audited_human_core | 5000 | 3.00 | 5.00 | 15 | 0.17 | 0.04 |
| aitdna_real_cowriting | 362 | 3.00 | 5.00 | 11 | 0.31 | 0.03 |
| calibration_human | 3000 | 3.00 | 5.00 | 14 | 0.19 | 0.03 |
| student_human | 1000 | 3.00 | 5.00 | 9 | 0.33 | 0.02 |
| heterogeneous_matched_human_controls | 300 | 6.00 | 17.00 | 18 | 0.65 | 0.43 |
| heterogeneous_block_replacement | 300 | 6.00 | 15.10 | 23 | 0.70 | 0.43 |

## Per-panel results, human_fpr_1pct (values in %; recalibrated to target on the 3000 calibration humans)

### moe_base_s1_mean

| row | max | binned_max | binned_max_nomerge | per_window_n | top2_mean | mean |
|---|---|---|---|---|---|---|
| aitdna_real_cowriting / fpr (n=95) | 3.2 | 2.1 | 2.1 | 3.2 | 4.2 | 5.3 |
| aitdna_real_cowriting / mix_rec (n=267) | 92.9 | 92.9 | 92.9 | 93.6 | 93.6 | 93.6 |
| audited_human_core / fpr (n=5000) | 1.2 | 1.1 | 1.0 | 1.1 | 1.2 | 1.2 |
| beemo_matched_editing / ai_rec (n=200) | 89.0 | 90.5 | 90.5 | 92.5 | 92.0 | 93.0 |
| beemo_matched_editing / mix_rec (n=200) | 60.0 | 62.5 | 62.5 | 72.0 | 68.5 | 73.5 |
| benchmark_source_matched_human_references / fpr (n=720) | 3.8 | 4.0 | 4.0 | 6.4 | 5.8 | 6.9 |
| controlled_replacement / mix_rec (n=800) | 60.0 | 60.6 | 60.6 | 64.5 | 62.7 | 65.2 |
| controlled_replacement_noncore / mix_rec (n=200) | 60.0 | 61.0 | 61.0 | 63.5 | 62.0 | 64.5 |
| editlens_in_family_reference / fpr (n=80) | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 |
| editlens_in_family_reference / ai_rec (n=80) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| editlens_in_family_reference / mix_rec (n=80) | 82.5 | 82.5 | 82.5 | 83.8 | 83.8 | 83.8 |
| fully_ai_core / ai_rec (n=1500) | 87.1 | 87.2 | 87.2 | 88.2 | 88.0 | 88.8 |
| fully_ai_noncore / ai_rec (n=300) | 87.3 | 87.0 | 87.0 | 88.0 | 87.7 | 88.3 |
| heterogeneous_block_replacement / mix_rec (n=300) | 96.7 | 96.0 | 96.7 | 94.3 | 96.7 | 94.7 |
| heterogeneous_matched_human_controls / fpr (n=300) | 47.0 | 43.7 | 45.7 | 37.0 | 51.3 | 41.7 |
| lamp_human_edited_ai / ai_rec (n=100) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lamp_human_edited_ai / mix_rec (n=100) | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 |
| student_human / fpr (n=1000) | 2.3 | 1.8 | 1.9 | 1.5 | 2.4 | 1.7 |
| (calibration humans, realised FPR) | 1.0 | 0.9 | 0.9 | 0.9 | 1.0 | 1.0 |

### moe_qwen_t21a2_mean

| row | max | binned_max | binned_max_nomerge | per_window_n | top2_mean | mean |
|---|---|---|---|---|---|---|
| aitdna_real_cowriting / fpr (n=95) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aitdna_real_cowriting / mix_rec (n=267) | 89.5 | 89.5 | 89.5 | 89.1 | 89.9 | 90.3 |
| audited_human_core / fpr (n=5000) | 1.4 | 1.8 | 1.9 | 1.6 | 1.6 | 1.5 |
| beemo_matched_editing / ai_rec (n=200) | 48.0 | 61.0 | 61.0 | 57.0 | 56.5 | 58.5 |
| beemo_matched_editing / mix_rec (n=200) | 21.5 | 33.0 | 33.0 | 25.5 | 26.5 | 30.0 |
| benchmark_source_matched_human_references / fpr (n=720) | 2.1 | 2.9 | 3.2 | 2.6 | 2.8 | 2.4 |
| controlled_replacement / mix_rec (n=800) | 29.1 | 35.1 | 35.2 | 33.1 | 32.8 | 35.8 |
| controlled_replacement_noncore / mix_rec (n=200) | 28.5 | 31.5 | 32.0 | 32.5 | 35.5 | 35.5 |
| editlens_in_family_reference / fpr (n=80) | 1.2 | 2.5 | 2.5 | 1.2 | 1.2 | 1.2 |
| editlens_in_family_reference / ai_rec (n=80) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| editlens_in_family_reference / mix_rec (n=80) | 77.5 | 82.5 | 82.5 | 80.0 | 80.0 | 81.2 |
| fully_ai_core / ai_rec (n=1500) | 79.2 | 82.1 | 82.2 | 81.3 | 81.4 | 82.1 |
| fully_ai_noncore / ai_rec (n=300) | 78.7 | 81.3 | 81.3 | 81.0 | 80.7 | 81.3 |
| heterogeneous_block_replacement / mix_rec (n=300) | 100.0 | 100.0 | 100.0 | 99.7 | 100.0 | 100.0 |
| heterogeneous_matched_human_controls / fpr (n=300) | 23.3 | 26.3 | 22.0 | 7.7 | 30.3 | 12.3 |
| lamp_human_edited_ai / ai_rec (n=100) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lamp_human_edited_ai / mix_rec (n=100) | 92.0 | 95.0 | 95.0 | 95.0 | 95.0 | 95.0 |
| student_human / fpr (n=1000) | 0.2 | 0.2 | 0.3 | 0.1 | 0.1 | 0.0 |
| (calibration humans, realised FPR) | 1.0 | 0.9 | 0.9 | 0.9 | 1.0 | 1.0 |

### moe_nemo_t21a2_mean

| row | max | binned_max | binned_max_nomerge | per_window_n | top2_mean | mean |
|---|---|---|---|---|---|---|
| aitdna_real_cowriting / fpr (n=95) | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 |
| aitdna_real_cowriting / mix_rec (n=267) | 90.6 | 90.3 | 90.3 | 91.8 | 91.8 | 92.1 |
| audited_human_core / fpr (n=5000) | 1.2 | 1.3 | 1.3 | 1.4 | 1.6 | 1.6 |
| beemo_matched_editing / ai_rec (n=200) | 62.5 | 70.0 | 70.0 | 73.5 | 73.0 | 80.0 |
| beemo_matched_editing / mix_rec (n=200) | 25.5 | 29.0 | 29.0 | 31.0 | 31.5 | 35.0 |
| benchmark_source_matched_human_references / fpr (n=720) | 2.1 | 2.1 | 2.2 | 2.1 | 3.1 | 3.2 |
| controlled_replacement / mix_rec (n=800) | 31.4 | 35.4 | 35.5 | 39.5 | 38.9 | 45.5 |
| controlled_replacement_noncore / mix_rec (n=200) | 30.5 | 35.0 | 35.0 | 38.0 | 38.0 | 41.5 |
| editlens_in_family_reference / fpr (n=80) | 3.8 | 3.8 | 3.8 | 3.8 | 3.8 | 2.5 |
| editlens_in_family_reference / ai_rec (n=80) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| editlens_in_family_reference / mix_rec (n=80) | 70.0 | 71.2 | 71.2 | 75.0 | 73.8 | 76.2 |
| fully_ai_core / ai_rec (n=1500) | 79.4 | 80.7 | 80.8 | 82.0 | 82.1 | 83.5 |
| fully_ai_noncore / ai_rec (n=300) | 79.3 | 80.3 | 80.3 | 81.0 | 81.3 | 83.0 |
| heterogeneous_block_replacement / mix_rec (n=300) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| heterogeneous_matched_human_controls / fpr (n=300) | 21.3 | 19.0 | 17.7 | 10.0 | 31.7 | 16.7 |
| lamp_human_edited_ai / ai_rec (n=100) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lamp_human_edited_ai / mix_rec (n=100) | 88.0 | 89.0 | 89.0 | 90.0 | 90.0 | 94.0 |
| student_human / fpr (n=1000) | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 |
| (calibration humans, realised FPR) | 1.0 | 0.9 | 0.9 | 1.1 | 1.0 | 1.0 |

### open_pangram_llama

| row | max | binned_max | binned_max_nomerge | per_window_n | top2_mean | mean |
|---|---|---|---|---|---|---|
| aitdna_real_cowriting / fpr (n=95) | 5.3 | 4.2 | 4.2 | 4.2 | 3.2 | 5.3 |
| aitdna_real_cowriting / mix_rec (n=267) | 91.0 | 91.0 | 91.0 | 91.4 | 92.1 | 92.1 |
| audited_human_core / fpr (n=5000) | 0.6 | 0.7 | 0.7 | 0.6 | 0.6 | 1.0 |
| beemo_matched_editing / ai_rec (n=200) | 71.0 | 79.0 | 79.0 | 77.5 | 77.0 | 80.5 |
| beemo_matched_editing / mix_rec (n=200) | 50.0 | 59.0 | 59.0 | 55.0 | 55.0 | 61.0 |
| benchmark_source_matched_human_references / fpr (n=720) | 2.1 | 1.7 | 1.8 | 1.9 | 1.8 | 2.6 |
| controlled_replacement / mix_rec (n=800) | 30.8 | 35.1 | 35.1 | 34.0 | 32.4 | 37.2 |
| controlled_replacement_noncore / mix_rec (n=200) | 28.5 | 34.5 | 35.0 | 32.5 | 33.5 | 39.5 |
| editlens_in_family_reference / fpr (n=80) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| editlens_in_family_reference / ai_rec (n=80) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| editlens_in_family_reference / mix_rec (n=80) | 61.3 | 63.7 | 67.5 | 67.5 | 67.5 | 68.8 |
| fully_ai_core / ai_rec (n=1500) | 82.2 | 83.1 | 83.6 | 83.4 | 83.3 | 84.7 |
| fully_ai_noncore / ai_rec (n=300) | 80.3 | 84.7 | 84.7 | 84.3 | 84.0 | 88.0 |
| heterogeneous_block_replacement / mix_rec (n=300) | 64.0 | 54.3 | 55.7 | 54.0 | 65.7 | 47.0 |
| heterogeneous_matched_human_controls / fpr (n=300) | 19.7 | 13.0 | 12.3 | 11.3 | 27.3 | 9.7 |
| lamp_human_edited_ai / ai_rec (n=100) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lamp_human_edited_ai / mix_rec (n=100) | 93.0 | 95.0 | 95.0 | 94.0 | 92.0 | 95.0 |
| student_human / fpr (n=1000) | 2.0 | 1.2 | 1.5 | 1.7 | 1.1 | 0.8 |
| (calibration humans, realised FPR) | 1.0 | 0.9 | 0.9 | 0.9 | 1.0 | 1.0 |

### Calibration documents and thresholds per bin (binned rules, 1%)

| model_id | rule | bin | n_cal | threshold | cal_fpr |
|---|---|---|---|---|---|
| moe_base_s1_mean | binned_max | 1-1 | 566 | 0.001417 | 0.008834 |
| moe_base_s1_mean | binned_max | 2-3 | 1863 | 0.001835 | 0.009662 |
| moe_base_s1_mean | binned_max | 4-inf | 571 | 0.00183 | 0.008757 |
| moe_base_s1_mean | binned_max_nomerge | 1-1 | 566 | 0.001417 | 0.008834 |
| moe_base_s1_mean | binned_max_nomerge | 2-3 | 1863 | 0.001835 | 0.009662 |
| moe_base_s1_mean | binned_max_nomerge | 4-7 | 476 | 0.002061 | 0.008403 |
| moe_base_s1_mean | binned_max_nomerge | 8-inf | 95 | 0.001588 | 0 |
| moe_qwen_t21a2_mean | binned_max | 1-1 | 566 | 0.08187 | 0.008834 |
| moe_qwen_t21a2_mean | binned_max | 2-3 | 1863 | 0.1963 | 0.009662 |
| moe_qwen_t21a2_mean | binned_max | 4-inf | 571 | 0.1534 | 0.008757 |
| moe_qwen_t21a2_mean | binned_max_nomerge | 1-1 | 566 | 0.08187 | 0.008834 |
| moe_qwen_t21a2_mean | binned_max_nomerge | 2-3 | 1863 | 0.1963 | 0.009662 |
| moe_qwen_t21a2_mean | binned_max_nomerge | 4-7 | 476 | 0.1263 | 0.008403 |
| moe_qwen_t21a2_mean | binned_max_nomerge | 8-inf | 95 | 0.2176 | 0 |
| moe_nemo_t21a2_mean | binned_max | 1-1 | 566 | 0.09425 | 0.008834 |
| moe_nemo_t21a2_mean | binned_max | 2-3 | 1863 | 0.1483 | 0.009662 |
| moe_nemo_t21a2_mean | binned_max | 4-inf | 571 | 0.1595 | 0.008757 |
| moe_nemo_t21a2_mean | binned_max_nomerge | 1-1 | 566 | 0.09425 | 0.008834 |
| moe_nemo_t21a2_mean | binned_max_nomerge | 2-3 | 1863 | 0.1483 | 0.009662 |
| moe_nemo_t21a2_mean | binned_max_nomerge | 4-7 | 476 | 0.1421 | 0.008403 |
| moe_nemo_t21a2_mean | binned_max_nomerge | 8-inf | 95 | 0.1803 | 0 |
| open_pangram_llama | binned_max | 1-1 | 566 | 0.2696 | 0.008834 |
| open_pangram_llama | binned_max | 2-3 | 1863 | 0.3793 | 0.009662 |
| open_pangram_llama | binned_max | 4-inf | 571 | 0.4658 | 0.008757 |
| open_pangram_llama | binned_max_nomerge | 1-1 | 566 | 0.2696 | 0.008834 |
| open_pangram_llama | binned_max_nomerge | 2-3 | 1863 | 0.3793 | 0.009662 |
| open_pangram_llama | binned_max_nomerge | 4-7 | 476 | 0.4259 | 0.008403 |
| open_pangram_llama | binned_max_nomerge | 8-inf | 95 | 0.4776 | 0 |

### per_window_n thresholds t(n) by window count (1%)

| bin | n_cal_docs | moe_base_s1_mean | moe_nemo_t21a2_mean | moe_qwen_t21a2_mean | open_pangram_llama |
|---|---|---|---|---|---|
| 1 | 566 | 0.0007689 | 0.07955 | 0.109 | 0.311 |
| 2 | 739 | 0.001364 | 0.1182 | 0.1609 | 0.3588 |
| 3 | 1124 | 0.001853 | 0.1476 | 0.1858 | 0.382 |
| 4 | 217 | 0.003121 | 0.1628 | 0.205 | 0.4117 |
| 5 | 124 | 0.003485 | 0.1777 | 0.2291 | 0.4228 |
| 6 | 83 | 0.004448 | 0.1869 | 0.2357 | 0.4259 |
| 7 | 52 | 0.005336 | 0.2021 | 0.2718 | 0.462 |
| 8 | 26 | 0.007958 | 0.2305 | 0.3069 | 0.466 |
| 9 | 31 | 0.008051 | 0.2572 | 0.3263 | 0.4691 |
| 10 | 12 | 0.009387 | 0.2596 | 0.3341 | 0.4692 |
| 11 | 8 | 0.01245 | 0.2598 | 0.3487 | 0.4763 |
| 12 | 10 | 0.01245 | 0.2598 | 0.3487 | 0.4763 |
| 13 | 6 | 0.01358 | 0.3326 | 0.3544 | 0.4776 |
| 14 | 2 | 0.01358 | 0.3326 | 0.3544 | 0.4776 |


## Diagnostics

### Doc FPR and per-window exceedance by window count, frozen 1% `max` threshold

| model_id | group | bin | n_docs | doc_fpr_% | window_exceed_% | n_windows |
|---|---|---|---|---|---|---|
| moe_base_s1_mean | calibration_human | all | 3000 | 1.00 | 0.38 | 8687 |
| moe_base_s1_mean | calibration_human | 1-1 | 566 | 0.88 | 0.88 | 566 |
| moe_base_s1_mean | calibration_human | 2-3 | 1863 | 1.02 | 0.45 | 4850 |
| moe_base_s1_mean | calibration_human | 4-7 | 476 | 1.26 | 0.26 | 2350 |
| moe_base_s1_mean | calibration_human | 8-inf | 95 | 0.00 | 0.00 | 921 |
| moe_base_s1_mean | audited_human_core | all | 5000 | 1.18 | 0.46 | 14394 |
| moe_base_s1_mean | audited_human_core | 1-1 | 1003 | 1.20 | 1.20 | 1003 |
| moe_base_s1_mean | audited_human_core | 2-3 | 3166 | 1.04 | 0.44 | 8266 |
| moe_base_s1_mean | audited_human_core | 4-7 | 639 | 1.72 | 0.43 | 3251 |
| moe_base_s1_mean | audited_human_core | 8-inf | 192 | 1.56 | 0.21 | 1874 |
| moe_base_s1_mean | student_human | all | 1000 | 2.30 | 0.72 | 3176 |
| moe_base_s1_mean | student_human | 1-1 | 70 | 1.43 | 1.43 | 70 |
| moe_base_s1_mean | student_human | 2-3 | 603 | 1.16 | 0.47 | 1492 |
| moe_base_s1_mean | student_human | 4-7 | 308 | 4.22 | 0.89 | 1459 |
| moe_base_s1_mean | student_human | 8-inf | 19 | 10.53 | 1.29 | 155 |
| moe_base_s1_mean | benchmark_source_matched_human_references | all | 720 | 3.75 | 1.53 | 1833 |
| moe_base_s1_mean | benchmark_source_matched_human_references | 1-1 | 226 | 6.19 | 6.19 | 226 |
| moe_base_s1_mean | benchmark_source_matched_human_references | 2-3 | 301 | 3.32 | 1.46 | 686 |
| moe_base_s1_mean | benchmark_source_matched_human_references | 4-7 | 179 | 1.12 | 0.26 | 772 |
| moe_base_s1_mean | benchmark_source_matched_human_references | 8-inf | 14 | 7.14 | 1.34 | 149 |
| moe_base_s1_mean | heterogeneous_matched_human_controls | all | 300 | 47.00 | 13.51 | 2147 |
| moe_base_s1_mean | heterogeneous_matched_human_controls | 1-1 | 2 | 100.00 | 100.00 | 2 |
| moe_base_s1_mean | heterogeneous_matched_human_controls | 2-3 | 103 | 48.54 | 26.85 | 257 |
| moe_base_s1_mean | heterogeneous_matched_human_controls | 4-7 | 66 | 39.39 | 16.42 | 335 |
| moe_base_s1_mean | heterogeneous_matched_human_controls | 8-inf | 129 | 48.84 | 10.56 | 1553 |
| moe_qwen_t21a2_mean | calibration_human | all | 3000 | 1.00 | 0.37 | 8687 |
| moe_qwen_t21a2_mean | calibration_human | 1-1 | 566 | 0.53 | 0.53 | 566 |
| moe_qwen_t21a2_mean | calibration_human | 2-3 | 1863 | 1.40 | 0.58 | 4850 |
| moe_qwen_t21a2_mean | calibration_human | 4-7 | 476 | 0.00 | 0.00 | 2350 |
| moe_qwen_t21a2_mean | calibration_human | 8-inf | 95 | 1.05 | 0.11 | 921 |
| moe_qwen_t21a2_mean | audited_human_core | all | 5000 | 1.44 | 0.54 | 14394 |
| moe_qwen_t21a2_mean | audited_human_core | 1-1 | 1003 | 1.10 | 1.10 | 1003 |
| moe_qwen_t21a2_mean | audited_human_core | 2-3 | 3166 | 1.67 | 0.71 | 8266 |
| moe_qwen_t21a2_mean | audited_human_core | 4-7 | 639 | 1.10 | 0.22 | 3251 |
| moe_qwen_t21a2_mean | audited_human_core | 8-inf | 192 | 0.52 | 0.05 | 1874 |
| moe_qwen_t21a2_mean | student_human | all | 1000 | 0.20 | 0.06 | 3176 |
| moe_qwen_t21a2_mean | student_human | 1-1 | 70 | 0.00 | 0.00 | 70 |
| moe_qwen_t21a2_mean | student_human | 2-3 | 603 | 0.17 | 0.07 | 1492 |
| moe_qwen_t21a2_mean | student_human | 4-7 | 308 | 0.32 | 0.07 | 1459 |
| moe_qwen_t21a2_mean | student_human | 8-inf | 19 | 0.00 | 0.00 | 155 |
| moe_qwen_t21a2_mean | benchmark_source_matched_human_references | all | 720 | 2.08 | 1.04 | 1833 |
| moe_qwen_t21a2_mean | benchmark_source_matched_human_references | 1-1 | 226 | 1.33 | 1.33 | 226 |
| moe_qwen_t21a2_mean | benchmark_source_matched_human_references | 2-3 | 301 | 1.33 | 0.73 | 686 |
| moe_qwen_t21a2_mean | benchmark_source_matched_human_references | 4-7 | 179 | 4.47 | 1.42 | 772 |
| moe_qwen_t21a2_mean | benchmark_source_matched_human_references | 8-inf | 14 | 0.00 | 0.00 | 149 |
| moe_qwen_t21a2_mean | heterogeneous_matched_human_controls | all | 300 | 23.33 | 5.50 | 2147 |
| moe_qwen_t21a2_mean | heterogeneous_matched_human_controls | 1-1 | 2 | 0.00 | 0.00 | 2 |
| moe_qwen_t21a2_mean | heterogeneous_matched_human_controls | 2-3 | 103 | 6.80 | 3.11 | 257 |
| moe_qwen_t21a2_mean | heterogeneous_matched_human_controls | 4-7 | 66 | 9.09 | 2.39 | 335 |
| moe_qwen_t21a2_mean | heterogeneous_matched_human_controls | 8-inf | 129 | 44.19 | 6.57 | 1553 |
| moe_nemo_t21a2_mean | calibration_human | all | 3000 | 1.00 | 0.35 | 8687 |
| moe_nemo_t21a2_mean | calibration_human | 1-1 | 566 | 0.71 | 0.71 | 566 |
| moe_nemo_t21a2_mean | calibration_human | 2-3 | 1863 | 1.07 | 0.41 | 4850 |
| moe_nemo_t21a2_mean | calibration_human | 4-7 | 476 | 0.84 | 0.17 | 2350 |
| moe_nemo_t21a2_mean | calibration_human | 8-inf | 95 | 2.11 | 0.22 | 921 |
| moe_nemo_t21a2_mean | audited_human_core | all | 5000 | 1.16 | 0.44 | 14394 |
| moe_nemo_t21a2_mean | audited_human_core | 1-1 | 1003 | 0.30 | 0.30 | 1003 |
| moe_nemo_t21a2_mean | audited_human_core | 2-3 | 3166 | 1.52 | 0.64 | 8266 |
| moe_nemo_t21a2_mean | audited_human_core | 4-7 | 639 | 0.78 | 0.15 | 3251 |
| moe_nemo_t21a2_mean | audited_human_core | 8-inf | 192 | 1.04 | 0.11 | 1874 |
| moe_nemo_t21a2_mean | student_human | all | 1000 | 0.10 | 0.03 | 3176 |
| moe_nemo_t21a2_mean | student_human | 1-1 | 70 | 0.00 | 0.00 | 70 |
| moe_nemo_t21a2_mean | student_human | 2-3 | 603 | 0.17 | 0.07 | 1492 |
| moe_nemo_t21a2_mean | student_human | 4-7 | 308 | 0.00 | 0.00 | 1459 |
| moe_nemo_t21a2_mean | student_human | 8-inf | 19 | 0.00 | 0.00 | 155 |
| moe_nemo_t21a2_mean | benchmark_source_matched_human_references | all | 720 | 2.08 | 0.98 | 1833 |
| moe_nemo_t21a2_mean | benchmark_source_matched_human_references | 1-1 | 226 | 0.88 | 0.88 | 226 |
| moe_nemo_t21a2_mean | benchmark_source_matched_human_references | 2-3 | 301 | 1.66 | 0.87 | 686 |
| moe_nemo_t21a2_mean | benchmark_source_matched_human_references | 4-7 | 179 | 3.91 | 1.17 | 772 |
| moe_nemo_t21a2_mean | benchmark_source_matched_human_references | 8-inf | 14 | 7.14 | 0.67 | 149 |
| moe_nemo_t21a2_mean | heterogeneous_matched_human_controls | all | 300 | 21.33 | 4.10 | 2147 |
| moe_nemo_t21a2_mean | heterogeneous_matched_human_controls | 1-1 | 2 | 0.00 | 0.00 | 2 |
| moe_nemo_t21a2_mean | heterogeneous_matched_human_controls | 2-3 | 103 | 9.71 | 3.89 | 257 |
| moe_nemo_t21a2_mean | heterogeneous_matched_human_controls | 4-7 | 66 | 7.58 | 1.79 | 335 |
| moe_nemo_t21a2_mean | heterogeneous_matched_human_controls | 8-inf | 129 | 37.98 | 4.64 | 1553 |
| open_pangram_llama | calibration_human | all | 3000 | 1.00 | 0.37 | 8687 |
| open_pangram_llama | calibration_human | 1-1 | 566 | 0.35 | 0.35 | 566 |
| open_pangram_llama | calibration_human | 2-3 | 1863 | 0.97 | 0.39 | 4850 |
| open_pangram_llama | calibration_human | 4-7 | 476 | 1.68 | 0.38 | 2350 |
| open_pangram_llama | calibration_human | 8-inf | 95 | 2.11 | 0.22 | 921 |
| open_pangram_llama | audited_human_core | all | 5000 | 0.56 | 0.22 | 14394 |
| open_pangram_llama | audited_human_core | 1-1 | 1003 | 0.20 | 0.20 | 1003 |
| open_pangram_llama | audited_human_core | 2-3 | 3166 | 0.66 | 0.29 | 8266 |
| open_pangram_llama | audited_human_core | 4-7 | 639 | 0.47 | 0.12 | 3251 |
| open_pangram_llama | audited_human_core | 8-inf | 192 | 1.04 | 0.11 | 1874 |
| open_pangram_llama | student_human | all | 1000 | 2.00 | 0.72 | 3176 |
| open_pangram_llama | student_human | 1-1 | 70 | 0.00 | 0.00 | 70 |
| open_pangram_llama | student_human | 2-3 | 603 | 1.33 | 0.54 | 1492 |
| open_pangram_llama | student_human | 4-7 | 308 | 3.57 | 0.96 | 1459 |
| open_pangram_llama | student_human | 8-inf | 19 | 5.26 | 0.65 | 155 |
| open_pangram_llama | benchmark_source_matched_human_references | all | 720 | 2.08 | 0.87 | 1833 |
| open_pangram_llama | benchmark_source_matched_human_references | 1-1 | 226 | 1.77 | 1.77 | 226 |
| open_pangram_llama | benchmark_source_matched_human_references | 2-3 | 301 | 1.66 | 0.73 | 686 |
| open_pangram_llama | benchmark_source_matched_human_references | 4-7 | 179 | 3.35 | 0.91 | 772 |
| open_pangram_llama | benchmark_source_matched_human_references | 8-inf | 14 | 0.00 | 0.00 | 149 |
| open_pangram_llama | heterogeneous_matched_human_controls | all | 300 | 19.67 | 6.85 | 2147 |
| open_pangram_llama | heterogeneous_matched_human_controls | 1-1 | 2 | 0.00 | 0.00 | 2 |
| open_pangram_llama | heterogeneous_matched_human_controls | 2-3 | 103 | 0.00 | 0.00 | 257 |
| open_pangram_llama | heterogeneous_matched_human_controls | 4-7 | 66 | 3.03 | 1.79 | 335 |
| open_pangram_llama | heterogeneous_matched_human_controls | 8-inf | 129 | 44.19 | 9.08 | 1553 |

### heterogeneous_matched_human_controls by subdomain: human FPR (%) per rule at 1%

| model | subdomain | n_docs | median_windows | window_exceed_%(max thr) | max | binned_max | per_window_n | top2_mean | mean |
|---|---|---|---|---|---|---|---|---|---|
| moe_base_s1_mean | economic_report | 49 | 9.0 | 27.5 | 89.8 | 89.8 | 67.3 | 93.9 | 75.5 |
| moe_base_s1_mean | historical_fiction | 106 | 9.0 | 1.0 | 9.4 | 3.8 | 0.0 | 15.1 | 2.8 |
| moe_base_s1_mean | ml_paper_body | 145 | 3.0 | 26.9 | 60.0 | 57.2 | 53.8 | 63.4 | 58.6 |
| moe_qwen_t21a2_mean | economic_report | 49 | 9.0 | 13.7 | 69.4 | 81.6 | 20.4 | 89.8 | 44.9 |
| moe_qwen_t21a2_mean | historical_fiction | 106 | 9.0 | 3.3 | 19.8 | 19.8 | 1.9 | 21.7 | 1.9 |
| moe_qwen_t21a2_mean | ml_paper_body | 145 | 3.0 | 3.2 | 10.3 | 12.4 | 7.6 | 16.6 | 9.0 |
| moe_nemo_t21a2_mean | economic_report | 49 | 9.0 | 11.1 | 65.3 | 59.2 | 20.4 | 79.6 | 40.8 |
| moe_nemo_t21a2_mean | historical_fiction | 106 | 9.0 | 1.6 | 14.2 | 10.4 | 1.9 | 21.7 | 2.8 |
| moe_nemo_t21a2_mean | ml_paper_body | 145 | 3.0 | 3.3 | 11.7 | 11.7 | 12.4 | 22.8 | 18.6 |
| open_pangram_llama | economic_report | 49 | 9.0 | 25.7 | 79.6 | 65.3 | 61.2 | 85.7 | 51.0 |
| open_pangram_llama | historical_fiction | 106 | 9.0 | 2.6 | 18.9 | 6.6 | 3.8 | 37.7 | 3.8 |
| open_pangram_llama | ml_paper_body | 145 | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The 5% operating point tables are in `tables.md`.
