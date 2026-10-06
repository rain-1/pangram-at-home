# Paired student essay v10 comparison

All models use thresholds selected for 2% document false alarms on the same separate generic-human calibration documents. The Qwen models flag a document when any token is highlighted; the two open Pangram EditLens models make document decisions. For genuinely mixed documents only, EditLens window scores are broadcast across overlapping windows to form coarse token scores, with separate 2% calibration of that span rule. The external article and PERSUADE sets have informed development and are now development tests rather than blind tests.

## Shared human and AI document tests

| Model | External human alarms | PERSUADE human alarms | Writers human alarms | External AI articles detected | LLMTrace pure AI detected | External AUROC | LLMTrace AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8 | 63/150 | 0/3,000 | 3/579 | 150/150 | 442/516 | 0.9836 | 0.9872 |
| v9 | 20/150 | 33/3,000 | 9/579 | 150/150 | 487/516 | 0.9891 | 0.9878 |
| v10 | 16/150 | 2/3,000 | 7/579 | 150/150 | 479/516 | 0.9921 | 0.9879 |
| Pangram RoBERTa | 0/150 | 0/3,000 | 1/579 | 118/150 | 353/516 | 0.9989 | 0.9506 |
| Pangram Llama | 0/150 | 0/3,000 | 1/579 | 149/150 | 453/516 | 0.9999 | 0.9834 |

## Additional human sources (all five models)

| Source | v8 | v9 | v10 | Pangram RoBERTa | Pangram Llama |
|---|---:|---:|---:|---:|---:|
| CNN articles | 1/500 | 9/500 | 4/500 | 0/500 | 0/500 |
| PMC papers | 2/346 | 0/346 | 0/346 | 0/346 | 0/346 |
| Archived EPA | 4/151 | 0/151 | 0/151 | 0/151 | 0/151 |
| Archived Smithsonian | 1/21 | 0/21 | 0/21 | 0/21 | 0/21 |
| ASAP 2.0 essays | 0/200 | 4/200 | 1/200 | 0/200 | 0/200 |

## Mixed-document localization (mixed records only)

| Source | Model | AI-token recall | Human-token FPR |
|---|---|---:|---:|
| LLMTrace | v8 | 16.6% | 0.1% |
| LLMTrace | v9 | 42.4% | 0.9% |
| LLMTrace | v10 | 38.6% | 0.8% |
| LLMTrace | Pangram RoBERTa | 12.1% | 2.9% |
| LLMTrace | Pangram Llama | 22.4% | 6.8% |
| AITDNA | v8 | 82.5% | 10.7% |
| AITDNA | v9 | 89.5% | 14.5% |
| AITDNA | v10 | 87.3% | 13.4% |
| AITDNA | Pangram RoBERTa | 57.2% | 20.5% |
| AITDNA | Pangram Llama | 90.9% | 57.1% |

## External AI-token coverage from our models

| Model | AI-token recall | AI articles with any highlight |
|---|---:|---:|
| v8 | 92.7% | 150/150 |
| v9 | 96.5% | 150/150 |
| v10 | 95.9% | 150/150 |

LLMTrace evaluation IDs are disjoint from training, but LLMTrace contributes some training records to v8–v10. EditLens document decisions cannot provide an AI-token recall for external articles. The mixed-document EditLens scores above are coarse window broadcasts, not native token predictions. The document detection counts also use different decision units: any highlighted Qwen token versus a single EditLens document score. V8/v9 saved-score threshold sweeps may differ by one document at a score tie because those older exports were float32; v10 exports retain float64.

Download charts (PDF on the `main` branch: `reports/essay_paired_v10_comparison.pdf`)
