# Local 4080 adaptive run

Queue state: **complete**.
Choices used development validation only. Held-out sets were scored after the choice was made.

| Run | Steps | Validation pAUC (≤5% FPR) | Validation FPR | Validation AI recall | Window AI recall | Human false-highlight rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vast_hpo_selected_v3 | — | 96.81% | 2.00% | 93.50% | — | — |
| qwen3_hpo_single_local_v1 | 3200 | 96.99% | 2.00% | 94.75% | 63.13% | 27.85% |
| qwen3_hpo_repeat2_local_v1 | 3200 | 97.54% | 2.00% | 95.75% | 51.68% | 18.44% |
| qwen3_hpo_256_repeat2_pilot_v1 | 1600 | 96.53% | 2.00% | 94.00% | 77.80% | 33.49% |

The window scores use synthetic joins and a threshold calibrated on that same development set.
They are localization diagnostics, not a measured real-world span error rate.

Validation operating points were recalibrated after correcting float32 threshold rounding.
The saved training-time recall-at-2%-FPR metric used a threshold that admitted one extra human validation example;
partial AUROC and the selected checkpoints were unaffected.

Short-window pilot chosen: **qwen3_hpo_256_repeat2_pilot_v1**.
Its 1,600-step result is exploratory and has a smaller training budget than the full runs.

## Readout

Repeat2 improved development pAUC, but on the frozen diverse test its FPR/AI recall was 5.20%/93.80%; single-copy was 3.20%/93.60%, and the Vast checkpoint was 2.20%/93.40%.

On the social/Q&A test category, Repeat2 flagged 8 of 50 human passages; Vast flagged 1 of 50. This category is small, but the difference matches the broader false-positive concern.

At a stricter 0.5% validation-FPR threshold, frozen test FPR/AI recall was 1.80%/91.80% for Repeat2 and 1.60%/89.60% for Vast.

The external paraphrase set remains difficult: AI recall was 28.67% for Repeat2 and 37.00% for Vast.

## Held-out evaluation: vast_hpo_selected_v3

| Set | Rows | pAUC (≤5% FPR) | AUROC | FPR | AI recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 800 | 96.81% | 99.46% | 2.00% | 93.50% |
| test | 1000 | 95.93% | 99.25% | 2.20% | 93.40% |
| raid_external | 1600 | 94.23% | 97.74% | 0.75% | 85.75% |
| enron_external | 3600 | 99.92% | 99.99% | 2.61% | 99.89% |
| gpt4_ood | 1200 | 98.33% | 99.24% | 0.67% | 95.67% |
| paraphrase | 1200 | 70.52% | 79.73% | 0.67% | 37.00% |
| standard_ebooks_human | 1860 | — | — | 0.00% | — |
| persuade_essays_human | 1000 | — | — | 1.40% | — |
| federal_reserve_human | 1622 | — | — | 1.73% | — |
| stackexchange_writers_human | 1000 | — | — | 4.60% | — |
| pmc_full_body_human | 261 | — | — | 2.68% | — |

## Held-out evaluation: qwen3_hpo_single_local_v1

| Set | Rows | pAUC (≤5% FPR) | AUROC | FPR | AI recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 800 | 96.99% | 99.38% | 2.00% | 94.75% |
| test | 1000 | 95.68% | 99.19% | 3.20% | 93.60% |
| raid_external | 1600 | 94.17% | 98.00% | 1.38% | 86.62% |
| enron_external | 3600 | 99.84% | 99.98% | 4.11% | 99.89% |
| gpt4_ood | 1200 | 98.24% | 99.21% | 1.50% | 96.50% |
| paraphrase | 1200 | 66.12% | 78.81% | 1.50% | 31.67% |
| standard_ebooks_human | 1860 | — | — | 0.00% | — |
| persuade_essays_human | 1000 | — | — | 0.90% | — |
| federal_reserve_human | 1622 | — | — | 0.12% | — |
| stackexchange_writers_human | 1000 | — | — | 4.40% | — |
| pmc_full_body_human | 261 | — | — | 3.07% | — |

## Held-out evaluation: qwen3_hpo_repeat2_local_v1

| Set | Rows | pAUC (≤5% FPR) | AUROC | FPR | AI recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 800 | 97.54% | 99.51% | 2.00% | 95.75% |
| test | 1000 | 95.43% | 98.99% | 5.20% | 93.80% |
| raid_external | 1600 | 94.37% | 97.53% | 0.88% | 85.38% |
| enron_external | 3600 | 99.91% | 99.99% | 4.44% | 99.94% |
| gpt4_ood | 1200 | 98.47% | 99.53% | 0.67% | 95.83% |
| paraphrase | 1200 | 67.64% | 77.00% | 0.50% | 28.67% |
| standard_ebooks_human | 1860 | — | — | 0.11% | — |
| persuade_essays_human | 1000 | — | — | 0.60% | — |
| federal_reserve_human | 1622 | — | — | 0.86% | — |
| stackexchange_writers_human | 1000 | — | — | 5.40% | — |
| pmc_full_body_human | 261 | — | — | 1.53% | — |

## Held-out evaluation: qwen3_hpo_256_repeat2_pilot_v1

| Set | Rows | pAUC (≤5% FPR) | AUROC | FPR | AI recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 800 | 96.53% | 99.03% | 2.00% | 94.00% |
| test | 1000 | 95.27% | 99.18% | 2.80% | 92.60% |
| raid_external | 1600 | 94.68% | 97.95% | 0.88% | 87.38% |
| enron_external | 3600 | 99.86% | 99.99% | 2.83% | 99.89% |
| gpt4_ood | 1200 | 97.96% | 99.13% | 1.83% | 96.00% |
| paraphrase | 1200 | 68.40% | 83.21% | 1.83% | 36.67% |
| standard_ebooks_human | 1860 | — | — | 0.16% | — |
| persuade_essays_human | 1000 | — | — | 0.20% | — |
| federal_reserve_human | 1622 | — | — | 11.04% | — |
| stackexchange_writers_human | 1000 | — | — | 3.60% | — |
| pmc_full_body_human | 261 | — | — | 0.38% | — |

## Whole-paper human audit

Held-out pre-2023 PMC bodies, scored in overlapping windows with
the passage-calibrated threshold. The same documents are used for each model.
False highlights come from averaged window scores; no token head was trained.

| Run | Papers | Tokens | False-highlight tokens (2% val) | False-highlight tokens (0.5% val) | Papers with any false highlight (2% val) |
| --- | ---: | ---: | ---: | ---: | ---: |
| vast_hpo_selected_v3 | 100 | 685096 | 0.79% | 0.09% | 9 |
| qwen3_hpo_single_local_v1 | 100 | 685096 | 1.77% | 0.04% | 17 |
| qwen3_hpo_repeat2_local_v1 | 100 | 685096 | 1.20% | 0.00% | 15 |

The original PMC body chunk audit shares paper IDs with the diverse splits (73 of 87 paper IDs). Its broad-evaluation FPR is therefore not fully document-independent. This whole-paper audit excludes every overlapping ID.

