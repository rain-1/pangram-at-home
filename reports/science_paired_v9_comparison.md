# Paired science v9 comparison

The v9 run changes paired science article windows while retaining the v8 architecture, tuned hyperparameters, 20k-document scale, and existing mixed examples. Thresholds for each Qwen model were selected on the same separate human-calibration protocol, not these article tests.

| Model | External human articles flagged | External AI articles caught | Document AUROC | External human-token FPR |
|---|---:|---:|---:|---:|
| Qwen balanced v6 | 64/150 | 150/150 | 0.9778 | 9.22% |
| Qwen publication v8 | 77/150 | 150/150 | 0.9836 | 12.94% |
| Qwen paired science v9 | 39/150 | 150/150 | 0.9891 | 5.40% |
| Pangram RoBERTa | 1/150 | 146/150 | 0.9989 | — |
| Pangram Llama | 7/150 | 150/150 | 0.9999 | — |

An article is counted as falsely flagged if Qwen highlights any human token. This is sensitive to isolated errors; Pangram makes a whole-article decision. The AI-generated article count is analogously any highlighted token.

## Human false-highlight rate by source

| Source | v8 | v9 |
|---|---:|---:|
| External articles | 12.94% | 5.40% |
| Generic held-out human | 0.01% | 0.44% |
| CNN news | 0.01% | 0.24% |
| PMC articles | 0.04% | 0.01% |
| Archived EPA | 0.38% | 0.01% |
| Archived magazine | 0.20% | 0.00% |
| AITDNA collaboration | 9.07% | 13.47% |

## Mixed-document AI-token recall

| Source | v8 | v9 |
|---|---:|---:|
| LLMTrace | 61.9% | 81.2% |
| AITDNA | 90.6% | 95.8% |

## External human false alarms by publisher

| Publisher | v8 | v9 |
|---|---:|---:|
| Associated Press | 4/15 | 0/15 |
| Discover | 11/20 | 7/20 |
| National Geographic | 19/25 | 10/25 |
| New York Times | 7/20 | 2/20 |
| Reader's Digest | 8/15 | 4/15 |
| Scientific American | 5/15 | 2/15 |
| Smithsonian Magazine | 16/25 | 8/25 |
| Wall Street Journal | 7/15 | 6/15 |

## Calibration target and operating point

Each threshold below is selected using only the separate generic-human calibration split. The target is the fraction of calibration documents allowed to have any false highlight.

| Model | Calibration target | Held-out human docs flagged /3,579 | External human articles flagged /150 | External AI-token recall |
|---|---:|---:|---:|---:|
| v8 | 0.5% | 1 | 31 | 80.3% |
| v8 | 1.0% | 2 | 49 | 88.4% |
| v8 | 2.0% | 3 | 63 | 92.7% |
| v8 | 5.0% | 8 | 77 | 97.4% |
| v9 | 0.5% | 4 | 9 | 87.2% |
| v9 | 1.0% | 16 | 15 | 93.4% |
| v9 | 2.0% | 42 | 20 | 96.5% |
| v9 | 5.0% | 265 | 39 | 99.5% |

The external Human Detectors human labels have attributed bylines but no independently verified AI-free workflow. The archived EPA and magazine extracts were captured before 2023. External article results have informed development and are no longer a pristine blind test. The magazine test is author-exclusive from its candidate training pool. The original Human Detectors source IDs repeat, although all 300 text hashes are distinct across 150 source article URLs; numeric results use row order, and new prediction exports include row index and text hash for unambiguous case review.

For v8 and v9, retrospective threshold sweeps use saved float32 scores. The frozen 5% rows use the original float64 evaluation summaries; near-threshold ties can shift a retrospective count by one document.

The generic held-out human split contains 3,000 PERSUADE 2.0 student essays and 579 Writers Stack Exchange documents, whereas the calibration split contains neither student essays nor CNN articles. The v9 generic-human regression is driven mainly by the student essays (239/3,000 documents with any false highlight at the original 5% calibration target versus 2/3,000 for v8). PERSUADE remains evaluation-only; its source license in our ingested version restricts training.

[Download the comparison charts](science_paired_v9_comparison.pdf)
