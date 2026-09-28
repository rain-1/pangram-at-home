# Essay/forum hard-negative pilot (v14): results

V14 adds 1,200 balanced human and AI essay/forum documents to the 20,000-document v13 mix. The Qwen3-1.7B Repeat2 architecture and fine-tuning settings are unchanged. Every threshold is calibrated independently on the same separate human set to target 2% of documents receiving any false AI highlight. The two Pangram EditLens baselines use their separately calibrated thresholds. All rows below use the same locked evaluations.

## Pure human and AI documents

| Model | Broad human false alarms | External article human false alarms | CNN human false alarms | PMC human false alarms | External AI caught | LLMTrace AI caught |
|---|---:|---:|---:|---:|---:|---:|
| v13 | 19/3,579 | 13/150 | 8/500 | 0/346 | 150/150 | 479/516 |
| v14 | 25/3,579 | 11/150 | 25/500 | 1/346 | 150/150 | 488/516 |
| Pangram RoBERTa | 1/3,579 | 0/150 | 0/500 | 0/346 | 118/150 | 353/516 |
| Pangram Llama | 1/3,579 | 0/150 | 0/500 | 0/346 | 149/150 | 453/516 |

## Mixed documents

| Model | LLMTrace AI recall / human FPR | AITDNA AI recall / human FPR | New-source AI recall / human FPR | New-source human false alarms |
|---|---:|---:|---:|---:|
| v13 | 48.7% / 2.0% | 87.6% / 15.7% | 64.0% / 0.3% | 15/590 |
| v14 | 58.9% / 3.4% | 92.0% / 19.1% | 73.8% / 1.1% | 20/590 |
| Pangram RoBERTa | 12.1% / 2.9% | 57.2% / 20.5% | 7.0% / 2.9% | 5/590 |
| Pangram Llama | 22.4% / 6.8% | 90.9% / 57.1% | 14.6% / 11.4% | 19/590 |

## Additional held-out human checks

| Model | LLMTrace pure human false alarms | ASAP student essays false alarms |
|---|---:|---:|
| v13 | 5/720 | 1/200 |
| v14 | 28/720 | 1/200 |
| Pangram RoBERTa | 2/720 | 0/200 |
| Pangram Llama | 2/720 | 0/200 |

## Interpretation

- V14 finished 3,352 steps in 3.27 training hours. [W&B run](https://wandb.ai/eac-adsf/pangram-at-home/runs/c31nuvg5).
- V14 improves mixed AI-token recall, especially LLMTrace (48.7% to 58.9%) and new-source spans (64.0% to 73.8%), but also raises human-token false positives on those sets (2.0% to 3.4% and 0.3% to 1.1%, respectively).
- The broad human alarm count rises from 19 to 25 of 3,579; CNN rises from 8 to 25 of 500; LLMTrace pure human rises from 5 to 28 of 720. This is not an across-the-board win.
- The Pangram baselines retain much lower false-positive counts on the pure-human sets. Their mixed-span scores come from coarse window broadcasts and should be read with that limitation.
- Pure-document detection and mixed-span localization are different operating conditions; report both rather than averaging them into one score.
