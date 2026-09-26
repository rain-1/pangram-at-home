# Vast winner and local Repeat2 comparison

Validation-only development results. No blind test was used.

Token training was deferred pending realistic, reviewed span data. Highlighting uses coarse overlapping-window classification scores.

| Run | Steps | Best step | Validation pAUC | Training hours | Peak allocated GB |
| --- | --- | --- | --- | --- | --- |
| Vast selected model (existing) | 3200 | 3200 | 0.96808 | remote | remote |
| qwen3_hpo_single_local_v1 | 3200 | 2800 | 0.97006 | 2.67 | 3.90 |
| qwen3_hpo_repeat2_local_v1 | 3200 | 2400 | 0.97510 | 2.65 | 4.07 |

The saved Vast model is the practical benchmark for improvement. The two fresh local runs isolate Repeat2: compare their passage metrics only if both completed the requested 3,200 steps. Vast used microbatch 2 × accumulation 4; local runs use 1 × 8, with the same effective batch. The token pilot, if enabled, uses a different synthetic task, so its token pAUC is not directly comparable.

Synthetic joins and inherited source labels limit realism. An AI-assisted class is not trained. HPO settings and the selected adapter are starting points; no claim is made that they are optimal for token training.

Whole-document span diagnostics (including the passage classifiers as coarse sliding-window baselines) are in the JSON report. Their thresholds are calibrated on the same development controls, so they do not estimate blind-test FPR.
