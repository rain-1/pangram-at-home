# Paired student essay v10 comparison

All rows below use a threshold selected for 2% document false alarms on the separate generic-human calibration split. v8 and v9 were rescored from saved token scores; v10 was evaluated directly at its frozen threshold. The external article and PERSUADE sets have informed development and are now development tests rather than blind tests.

## Human documents with any false highlight

| Source | v8 | v9 | v10 |
|---|---:|---:|---:|
| External articles | 63/150 | 20/150 | 16/150 |
| PERSUADE essays | 0/3000 | 33/3000 | 2/3000 |
| Writers Stack Exchange | 3/579 | 9/579 | 7/579 |
| CNN articles | 1/500 | 9/500 | 4/500 |
| PMC papers | 2/346 | 0/346 | 0/346 |
| Archived EPA | 4/151 | 0/151 | 0/151 |
| Archived Smithsonian | 1/21 | 0/21 | 0/21 |
| ASAP 2.0 essays | 0/200 | 4/200 | 1/200 |

## AI-token recall on mixed documents

| Source | v8 | v9 | v10 |
|---|---:|---:|---:|
| LLMTrace | 43.3% recall / 0.0% FPR | 66.2% recall / 0.4% FPR | 64.8% recall / 0.3% FPR |
| AITDNA | 82.6% recall / 5.7% FPR | 89.5% recall / 7.7% FPR | 87.3% recall / 7.2% FPR |

## External AI articles

| Model | AI-token recall | AI articles with any highlight | Document AUROC |
|---|---:|---:|---:|
| v8 | 92.7% | 150/150 | 0.9836 |
| v9 | 96.5% | 150/150 | 0.9891 |
| v10 | 95.9% | 150/150 | 0.9921 |

The two open Pangram baselines appear on the external article ROC chart. They make whole-document decisions, so their article alarm counts are not directly equivalent to any-token highlights from these span models. V8/v9 saved-score threshold sweeps may differ by one document at a score tie because those older exports were float32; v10 exports retain float64.

[Download charts](essay_paired_v10_comparison.pdf)
