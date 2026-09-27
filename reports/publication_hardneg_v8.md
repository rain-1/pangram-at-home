# Publication hard-negative pilot v8

The v8 model changes 500 of 20,000 training documents from DAMASHA mixed examples to dated, attributed human publication prose from five CC BY publishers. The architecture, tuned hyperparameters, and initialization match v6. Original article stress results have guided this experiment and are now development evidence, not an untouched final test.

LLMTrace is 600/20,000 documents (3.0%), 2.32% of training windows, and 0.92% of supervised token positions. DAMASHA is the largest source at 24.5% of documents and 27.1% of supervised positions. The final token supervision is 57.1% human and 42.9% AI.

## Plain-language verdict

**Published articles: FAIL.** At the separately calibrated threshold, v8 falsely flags 77/150 attributed-human articles (51.3%), versus 64/150 for v6. It catches 150/150 AI articles (100.0%). The stated goal is at most 10% human false alarms with at least 95% AI recall.

The any-highlight document rule includes small isolated errors. A stricter descriptive view counts articles with at least 10% of tokens falsely highlighted: v6 38/150; v8 52/150. This is not a recalibrated operating threshold.

**Mixed documents: FAIL.** A useful passage highlighter should both find AI passages and leave human passages unmarked across datasets. The current check requires ≥75% AI-token recall on LLMTrace, ≤5% human-token false marks on AITDNA, and ≥50% AI-token recall on CoAuthor. This is a practical gate, not a published benchmark standard.

“AI-token recall” means the fraction of truly AI-written word/token positions the model highlights. “Human-token false marks” means the fraction of human-written positions it incorrectly highlights. “AI spans half covered” counts an AI passage only if the model highlights at least half of it; this is stricter than merely touching its edge.

Qwen v6 threshold: `5.5469`; v8 threshold: `5.0313`. Both use the same independent 1,120-document calibration protocol.

## Fully human and fully AI documents

| Dataset | Model | Human false alarms | AI caught | Document AUROC |
| --- | --- | ---: | ---: | ---: |
| LLMTrace test | Qwen balanced v6 | 0/720 | 462/516 | 0.9878 |
| LLMTrace test | Qwen publication v8 | 2/720 | 476/516 | 0.9872 |
| LLMTrace test | Pangram RoBERTa | 10/720 | 423/516 | 0.9506 |
| LLMTrace test | Pangram Llama | 19/720 | 486/516 | 0.9834 |
| Synthetic v4 validation | Qwen balanced v6 | 0/152 | 141/148 | 0.9926 |
| Synthetic v4 validation | Qwen publication v8 | 0/152 | 138/148 | 0.9892 |
| Synthetic v4 validation | Pangram RoBERTa | 2/152 | 52/148 | 0.7398 |
| Synthetic v4 validation | Pangram Llama | 0/152 | 75/148 | 0.8523 |
| External articles | Qwen balanced v6 | 64/150 | 150/150 | 0.9778 |
| External articles | Qwen publication v8 | 77/150 | 150/150 | 0.9836 |
| External articles | Pangram RoBERTa | 1/150 | 146/150 | 0.9989 |
| External articles | Pangram Llama | 7/150 | 150/150 | 0.9999 |
| Locked human test | Qwen balanced v6 | 15/3579 | — | — |
| Locked human test | Qwen publication v8 | 8/3579 | — | — |
| Locked human test | Pangram RoBERTa | 11/3579 | — | — |
| Locked human test | Pangram Llama | 11/3579 | — | — |
| AITDNA human controls | Qwen balanced v6 | 4/103 | — | — |
| AITDNA human controls | Qwen publication v8 | 8/103 | — | — |
| AITDNA human controls | Pangram RoBERTa | 2/103 | — | — |
| AITDNA human controls | Pangram Llama | 2/103 | — | — |

## Mixed documents

| Dataset | Model | AI-token recall | Human-token false marks | AI spans half covered |
| --- | --- | ---: | ---: | ---: |
| LLMTrace test | Qwen balanced v6 | 35.8% | 0.57% | 20.3% |
| LLMTrace test | Qwen publication v8 | 34.3% | 0.67% | 19.9% |
| LLMTrace test | Pangram RoBERTa | 18.1% | 5.52% | 9.8% |
| LLMTrace test | Pangram Llama | 37.0% | 15.35% | 24.3% |
| Synthetic v4 validation | Qwen balanced v6 | 76.9% | 0.06% | 79.3% |
| Synthetic v4 validation | Qwen publication v8 | 79.6% | 0.07% | 81.7% |
| Synthetic v4 validation | Pangram RoBERTa | 15.7% | 3.73% | 13.6% |
| Synthetic v4 validation | Pangram Llama | 37.2% | 16.36% | 30.3% |
| AITDNA collaboration | Qwen balanced v6 | 89.8% | 15.03% | 53.2% |
| AITDNA collaboration | Qwen publication v8 | 90.6% | 17.21% | 55.7% |
| AITDNA collaboration | Pangram RoBERTa | 78.6% | 34.54% | 54.1% |
| AITDNA collaboration | Pangram Llama | 95.5% | 66.20% | 82.7% |
| CoAuthor collaboration | Qwen balanced v6 | 9.7% | 0.30% | 2.4% |
| CoAuthor collaboration | Qwen publication v8 | 8.2% | 0.42% | 2.7% |
| CoAuthor collaboration | Pangram RoBERTa | 0.0% | 0.00% | 0.0% |
| CoAuthor collaboration | Pangram Llama | 0.0% | 0.00% | 0.0% |

AI-token recall weighs long passages more heavily. The half-covered span rate gives each AI passage one vote, including short insertions. These sets differ substantially in passage length:

| Mixed set | AI passages | Median AI passage length |
| --- | ---: | ---: |
| LLMTrace | 2261 | 22 words |
| Synthetic v4 | 603 | 90 words |
| AITDNA | 4172 | 1 words |
| CoAuthor | 407 | 14 words |

## Human article false alarms by publication

| Publication | Human articles | Qwen v6 | Qwen v8 | Pangram RoBERTa | Pangram Llama |
| --- | ---: | ---: | ---: | ---: | ---: |
| Associated Press | 15 | 2/15 | 4/15 | 0/15 | 0/15 |
| Discover | 20 | 10/20 | 11/20 | 0/20 | 1/20 |
| National Geographic | 25 | 18/25 | 19/25 | 1/25 | 4/25 |
| New York Times | 20 | 5/20 | 7/20 | 0/20 | 0/20 |
| Reader's Digest | 15 | 7/15 | 8/15 | 0/15 | 1/15 |
| Scientific American | 15 | 3/15 | 5/15 | 0/15 | 0/15 |
| Smithsonian Magazine | 25 | 12/25 | 16/25 | 0/25 | 0/25 |
| Wall Street Journal | 15 | 7/15 | 7/15 | 0/15 | 1/15 |

## Separate publication controls

| Human-only source | Qwen v6 false alarms | Qwen v8 false alarms |
| --- | ---: | ---: |
| PMC pre-2023 | 5/346 | 5/346 |
| CNN/Daily Mail historical | 1/500 | 2/500 |
| Common Pile four unseen publishers | — | 0/200 |

## Independent publication threshold check

Common Pile human calibration threshold for 5% FPR: `-2.8984`; source-aware maximum with original controls: `5.0313`.

At that source-aware threshold: external human false alarms 77/150; external AI caught 150/150; new held-out Common Pile human false alarms 0/200.

Common Pile calibration and locked-test articles come from publishers absent in v8 training. Dates/bylines provide strong but not absolute authorship evidence. The external Human Detectors set has attributed authors but AI-free workflows were not independently verified. EditLens has coarse broadcast window scores for mixed text.
