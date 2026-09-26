# Independent publication calibration of balanced Qwen v6

Each source calibration uses its own 5% document false-alarm threshold; source-aware uses their maximum. No locked test or external stress article is used for threshold selection.

## Thresholds

| Human calibration source | Threshold |
| --- | ---: |
| original_1120_human | 5.5469 |
| pmc_200 | 3.6992 |
| cnn_300 | 1.0858 |
| source_aware | 5.5469 |

## Locked evaluation

| Decision rule | External attributed-human false alarms | External AI articles caught | PMC human false alarms | CNN/Daily Mail human false alarms |
| --- | ---: | ---: | ---: | ---: |
| original | 64/150 | 150/150 | 5/346 | 1/500 |
| source_aware | 64/150 | 150/150 | 5/346 | 1/500 |

These thresholds operate on each document’s maximum token score. A threshold can lower false alarms at the cost of AI recall; consult the Qwen/baseline ROC and mixed-token charts as well. The 150 external human articles have attributed authors but their workflows were not independently verified as AI-free.
