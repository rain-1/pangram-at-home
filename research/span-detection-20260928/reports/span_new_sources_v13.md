# Half-dose GRADTEX follow-up (v13)

V13 keeps the v12 human additions and restores 500 DAMASHA mixed examples in place of 500 GRADTEX mixed examples. It retains the 20,000-document budget, v10/v12 validation and new-source holdout files, Qwen3-1.7B Repeat2 architecture, initialization adapter, and tuned hyperparameters. AI-labeled tokens are 42.8% of supervised positions. Each Qwen threshold is independently chosen on the same separate human calibration set for 2% document-any false highlights; the Pangram EditLens scores use the same calibration text.

## Human and AI documents

| Model | Broad human alarms | External human alarms | CNN human alarms | PMC human alarms | External AI caught | LLMTrace pure-AI caught |
|---|---:|---:|---:|---:|---:|---:|
| v10 | 9/3,579 | 16/150 | 4/500 | 0/346 | 150/150 | 479/516 |
| v12 | 45/3,579 | 11/150 | 5/500 | 0/346 | 150/150 | 483/516 |
| v13 | 19/3,579 | 13/150 | 8/500 | 0/346 | 150/150 | 479/516 |
| Pangram RoBERTa | 1/3,579 | 0/150 | 0/500 | 0/346 | 118/150 | 353/516 |
| Pangram Llama | 1/3,579 | 0/150 | 0/500 | 0/346 | 149/150 | 453/516 |

## Mixed-document localization

| Model | LLMTrace AI-token recall / human-token FPR | AITDNA AI-token recall / human-token FPR | New GRADTEX AI-token recall / human-token FPR | New-source human alarms |
|---|---:|---:|---:|---:|
| v10 | 38.6% / 0.76% | 87.3% / 13.4% | 44.5% / 0.18% | 19/590 |
| v12 | 51.8% / 2.1% | 90.1% / 16.5% | 66.5% / 0.38% | 15/590 |
| v13 | 48.7% / 2.0% | 87.6% / 15.7% | 64.0% / 0.29% | 15/590 |
| Pangram RoBERTa | 12.1% / 2.9% | 57.2% / 20.5% | 7.0% / 2.9% | 5/590 |
| Pangram Llama | 22.4% / 6.8% | 90.9% / 57.1% | 14.6% / 11.4% | 19/590 |

## New human-source false alarms

| Qwen model | Dolly (150) | Historical fiction (100) | Named pre-LLM essays (90) | GRADTEX human (250) |
|---|---:|---:|---:|---:|
| v10 | 15/150 | 0/100 | 0/90 | 4/250 |
| v12 | 11/150 | 0/100 | 0/90 | 4/250 |
| v13 | 9/150 | 0/100 | 0/90 | 6/250 |

## Training and interpretation

- Completed 3,202 steps in 1.70 hours. Best validation partial AUROC: 0.9577.
- [Weights & Biases run](https://wandb.ai/eac-adsf/pangram-at-home/runs/nma7wgo0).
- V13 trades some of v12's mixed-span recall for fewer broad-human false alarms: 19/3,579 versus 45/3,579. V10 still has the fewest broad-human alarms among our checkpoints (9/3,579), while v13 improves LLMTrace mixed AI-token recall from 38.6% to 48.7%. Keep v10 as the conservative default until a calibration-only threshold sweep shows whether v13 can retain that gain at an acceptable false-positive rate.
- The Pangram baselines produce fewer pure-human false alarms on these sets, but their mixed-span recall is lower. Their window-broadcast localization and our token-level output represent different resolution levels.
- These evaluations have informed development and are not untouched final tests. The new-source holdout is disjoint by work/group, not by author. GRADTEX span boundaries are inferred from preserved context.

Download comparison charts (PDF on the `main` branch: `reports/span_new_sources_v13.pdf`)
