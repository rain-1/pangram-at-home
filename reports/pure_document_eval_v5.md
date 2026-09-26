# Fully human and fully AI evaluation

AI-token recall means the fraction of tokens known to be AI that receive an AI highlight. It is not balanced accuracy or document accuracy. A random model that highlights half the AI tokens would also highlight about half the human tokens, while these models are calibrated to highlight few human tokens.

## Held-out LLMTrace pure documents

| Model | AI tokens highlighted, 516 pure AI docs | AI docs with any highlight | Human docs with any false highlight |
| --- | ---: | ---: | ---: |
| Earlier v4 | 72.9% | 451/516 (87.4%) | 2/720 |
| 5k × 1 | 71.6% | 458/516 (88.8%) | 0/720 |
| 10k × 1 | 85.1% | 488/516 (94.6%) | 0/720 |
| 20k × 1 | 90.8% | 500/516 (96.9%) | 0/720 |
| 5k × 4 | 91.2% | 500/516 (96.9%) | 1/720 |

The any-highlight rule is deliberately permissive. The token-recall column shows whether the model highlights most of an AI document rather than a stray word. The older v4 used a different training mixture; the 5k, 10k, and 20k tiers are nested. The 5k × 4 trial has the same optimizer-step budget as 20k × 1.

## Shared synthetic v4 evaluation: models and baselines

| Model | AI-token recall, 148 pure AI docs | Human docs with any false highlight, 152 pure human docs |
| --- | ---: | ---: |
| Earlier v4 | 78.2% | 0/152 |
| 5k × 1 | 52.6% | 0/152 |
| 10k × 1 | 72.1% | 0/152 |
| 20k × 1 | 72.6% | 0/152 |
| 5k × 4 | 74.7% | 0/152 |
| Token v3 | 17.5% | 0/152 |
| Qwen passage | 96.1% | 5/152 |
| Char TF-IDF | 54.5% | 2/152 |
| Word TF-IDF | 54.9% | 4/152 |

The Qwen passage baseline has high pure-AI recall, but its previously measured human-token FPR on mixed documents is much larger. Pure-document performance alone is therefore insufficient for the span-localization goal. The held-out LLMTrace test has no TF-IDF or Qwen passage baseline scores yet, so those comparisons use the shared synthetic evaluation.
