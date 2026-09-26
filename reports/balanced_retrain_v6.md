# Source-balanced retrain evaluation

Old Qwen threshold: `6.7344`; new Qwen threshold: `5.5469`. Both are independently calibrated to 5% document false alarms on the same 1,120 human controls. The external article set was not used to select either threshold.

## Fully human and fully AI documents

| Dataset | Model | Human false alarms | AI documents caught | Document AUROC |
| --- | --- | ---: | ---: | ---: |
| LLMTrace test | Qwen old | 0/720 (0.0%) | 500/516 (96.9%) | 0.9949 |
| LLMTrace test | Qwen balanced | 0/720 (0.0%) | 462/516 (89.5%) | 0.9878 |
| LLMTrace test | Pangram RoBERTa | 10/720 (1.4%) | 423/516 (82.0%) | 0.9506 |
| LLMTrace test | Pangram Llama | 19/720 (2.6%) | 486/516 (94.2%) | 0.9834 |
| Synthetic v4 validation | Qwen old | 0/152 (0.0%) | 118/148 (79.7%) | 0.9917 |
| Synthetic v4 validation | Qwen balanced | 0/152 (0.0%) | 141/148 (95.3%) | 0.9926 |
| Synthetic v4 validation | Pangram RoBERTa | 2/152 (1.3%) | 52/148 (35.1%) | 0.7398 |
| Synthetic v4 validation | Pangram Llama | 0/152 (0.0%) | 75/148 (50.7%) | 0.8523 |
| External articles | Qwen old | 74/150 (49.3%) | 150/150 (100.0%) | 0.9370 |
| External articles | Qwen balanced | 64/150 (42.7%) | 150/150 (100.0%) | 0.9778 |
| External articles | Pangram RoBERTa | 1/150 (0.7%) | 146/150 (97.3%) | 0.9989 |
| External articles | Pangram Llama | 7/150 (4.7%) | 150/150 (100.0%) | 0.9999 |
| Locked human test | Qwen old | 0/3579 (0.0%) | — | — |
| Locked human test | Qwen balanced | 15/3579 (0.4%) | — | — |
| Locked human test | Pangram RoBERTa | 11/3579 (0.3%) | — | — |
| Locked human test | Pangram Llama | 11/3579 (0.3%) | — | — |
| AITDNA human controls | Qwen old | 1/103 (1.0%) | — | — |
| AITDNA human controls | Qwen balanced | 4/103 (3.9%) | — | — |
| AITDNA human controls | Pangram RoBERTa | 2/103 (1.9%) | — | — |
| AITDNA human controls | Pangram Llama | 2/103 (1.9%) | — | — |

## Mixed documents

| Dataset | Model | AI token recall | Human token false alarms | AI spans half covered |
| --- | --- | ---: | ---: | ---: |
| LLMTrace test | Qwen old | 24.9% | 0.0% | 14.8% |
| LLMTrace test | Qwen balanced | 35.8% | 0.6% | 20.3% |
| LLMTrace test | Pangram RoBERTa | 18.1% | 5.5% | 9.8% |
| LLMTrace test | Pangram Llama | 37.0% | 15.3% | 24.3% |
| Synthetic v4 validation | Qwen old | 67.2% | 0.1% | 61.6% |
| Synthetic v4 validation | Qwen balanced | 76.9% | 0.1% | 79.3% |
| Synthetic v4 validation | Pangram RoBERTa | 15.7% | 3.7% | 13.6% |
| Synthetic v4 validation | Pangram Llama | 37.2% | 16.4% | 30.3% |
| AITDNA collaboration | Qwen old | 86.6% | 15.2% | 50.8% |
| AITDNA collaboration | Qwen balanced | 89.8% | 15.0% | 53.2% |
| AITDNA collaboration | Pangram RoBERTa | 78.6% | 34.5% | 54.1% |
| AITDNA collaboration | Pangram Llama | 95.5% | 66.2% | 82.7% |
| CoAuthor collaboration | Qwen old | 0.0% | 0.0% | 0.0% |
| CoAuthor collaboration | Qwen balanced | 9.7% | 0.3% | 2.4% |
| CoAuthor collaboration | Pangram RoBERTa | 0.0% | 0.0% | 0.0% |
| CoAuthor collaboration | Pangram Llama | 0.0% | 0.0% | 0.0% |

## Verdict

Published human articles: old Qwen 74/150 false alarms; balanced Qwen 64/150. AI article recall: 150/150 old; 150/150 balanced.

Article AUROC: 0.9370 old versus 0.9778 balanced.

The 150 external human articles have attributed human authors but their production workflows were not independently verified as AI-free. EditLens baseline span scores are coarse window broadcasts, not native token heads.

## Article publication breakdown

| Publication | Human articles | Qwen old false alarms | Qwen balanced false alarms | Pangram RoBERTa | Pangram Llama |
| --- | ---: | ---: | ---: | ---: | ---: |
| Associated Press | 15 | 4/15 | 2/15 | 0/15 | 0/15 |
| Discover | 20 | 8/20 | 10/20 | 0/20 | 1/20 |
| National Geographic | 25 | 17/25 | 18/25 | 1/25 | 4/25 |
| New York Times | 20 | 7/20 | 5/20 | 0/20 | 0/20 |
| Reader's Digest | 15 | 10/15 | 7/15 | 0/15 | 1/15 |
| Scientific American | 15 | 5/15 | 3/15 | 0/15 | 0/15 |
| Smithsonian Magazine | 25 | 15/25 | 12/25 | 0/25 | 0/25 |
| Wall Street Journal | 15 | 8/15 | 7/15 | 0/15 | 1/15 |
