# New-source v12 span retrain

The model uses the v10 Qwen3-1.7B Repeat2 architecture, initialization adapter,
LoRA settings, effective batch size, validation set, and 20,000-document budget.
Exactly 1,860 documents were replaced. New sources are 600 Dolly responses,
200 historical fiction excerpts, 60 pre-LLM author essay excerpts, and
1,000 GRADTEX mixed completions. Training exposure is 42.4% AI-labeled tokens;
the largest source is DAMASHA at 21.4%.

All four models below use thresholds chosen on the same separate human
calibration set for 2% document-any false highlights. EditLens span scores
broadcast window-level decisions; the Qwen scores are native token predictions.
The new-source holdout is a development evaluation, disjoint by source group
from this training mix. GRADTEX boundaries are inferred from exact preserved
context, so they are weaker labels than independently logged provenance.

## New-source holdout: human false alarms

| Source | Documents | v10 | v12 | Pangram RoBERTa | Pangram Llama |
|---|---:|---:|---:|---:|---:|
| Dolly employee responses | 150 | 15/150 (10.0%) | 11/150 (7.3%) | 4/150 (2.7%) | 16/150 (10.7%) |
| Historical fiction | 100 | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 2/100 (2.0%) |
| Pre-LLM author essays | 90 | 0/90 (0.0%) | 0/90 (0.0%) | 0/90 (0.0%) | 1/90 (1.1%) |
| GRADTEX human | 250 | 4/250 (1.6%) | 4/250 (1.6%) | 1/250 (0.4%) | 0/250 (0.0%) |

## New-source holdout: mixed GRADTEX token localization

| Model | AI-token recall | Human-token FPR | Token AUROC |
|---|---:|---:|---:|
| v10 | 44.5% | 0.2% | 0.915 |
| v12 | 66.5% | 0.4% | 0.955 |
| Pangram RoBERTa | 7.0% | 2.9% | 0.609 |
| Pangram Llama | 14.6% | 11.4% | 0.573 |

## Established held-out evaluations

| Model | External human alarms | External AI documents caught | LLMTrace AI documents caught | Locked human alarms | LLMTrace mixed AI-token recall / human-token FPR | AITDNA mixed AI-token recall / human-token FPR |
|---|---:|---:|---:|---:|---:|---:|
| v10 | 16/150 | 150/150 | 479/516 | 9/3579 | 38.6% / 0.8% | 87.3% / 13.4% |
| v12 | 11/150 | 150/150 | 483/516 | 45/3579 | 51.8% / 2.1% | 90.1% / 16.5% |
| Pangram RoBERTa | 0/150 | 118/150 | 353/516 | 1/3579 | 12.1% / 2.9% | 57.2% / 20.5% |
| Pangram Llama | 0/150 | 149/150 | 453/516 | 1/3579 | 22.4% / 6.8% | 90.9% / 57.1% |

## Training and limits

- Completed 3,202 optimizer steps in 1.70 hours; best validation partial AUROC: 0.9657.
- [Weights & Biases run](https://wandb.ai/eac-adsf/pangram-at-home/runs/wv9s1zne).
- New-source holdout labels have different evidence levels: Dolly employees were instructed not to use AI; historical fiction and pre-LLM essays have publication-era evidence; GRADTEX mixed boundaries are reconstructed.
- The new holdout is work/group-disjoint, not author-disjoint: writers may appear on both sides. Its author rows test held-out prose, not unknown authors.
- External articles and other established evaluations have informed model development. They are useful comparisons, not untouched final tests.

## Interpretation

At the shared 2% generic-human calibration target, v12 substantially improves mixed GRADTEX and LLMTrace AI-token recall and lowers external-article false alarms. It also raises broad human false alarms from 9/3,579 to 45/3,579, especially on student essays. This checkpoint is therefore an informative data-mixture experiment, not a replacement for v10 at this operating point. A separate, disjoint essay-calibration threshold analysis is the next check; if that cannot retain the recall gain at acceptable false alarms, the next mixture should restore more student-essay human coverage.

[Download comparison charts](span_new_sources_v12.pdf)
