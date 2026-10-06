# Span data-size curve

Nested 5k, 10k, and 20k training documents use the same Qwen3-1.7B Repeat2 token architecture, Vast-selected LoRA settings, initialization adapter, fixed validation, and frozen test sets. The 5k × 4 run matches the 20k run's 3,269 optimizer steps to separate exposure to new data from additional updates. The 5k and 10k runs used Vast RTX 4090s; the 20k and matched-compute 5k runs were completed on the local RTX 4080 after Vast credit ran out. All training hyperparameters and BF16 precision were held fixed, but GPU hardware is a minor remaining experimental difference. Thresholds are calibrated separately on the same pure-human calibration set at 5% document-any false highlight.

| Run | Documents | Optimizer steps | LLMTrace heldout AUROC | LLMTrace AI recall | LLMTrace mixed AI recall | LLMTrace human FPR | LLMTrace recall at v4 FPR* | AITDNA mixed AI recall | AITDNA mixed human FPR | Locked human any-highlight | CoAuthor AI recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k · 1 epoch | 5,000 | 816 | 0.978 | 51.0% | 16.3% | 0.011% | 83.1% | 77.8% | 9.4% | 0.00% | 0.0% |
| 10k · 1 epoch | 10,000 | 1,634 | 0.983 | 63.1% | 26.0% | 0.011% | 86.6% | 83.7% | 11.2% | 0.03% | 0.6% |
| 20k · 1 epoch | 20,000 | 3,269 | 0.987 | 66.3% | 24.9% | 0.008% | 88.8% | 86.6% | 15.2% | 0.00% | 0.0% |
| 5k · 4 epochs | 5,000 | 3,269 | 0.983 | 67.9% | 28.4% | 0.020% | 86.6% | 88.5% | 16.2% | 0.06% | 0.8% |

The previous v4 checkpoint has 0.937 AUROC and recalls 58.5% of AI tokens at 0.67% human-token FPR on the same held-out LLMTrace test. It trained on the older 5k synthetic mix and was not part of this controlled nested-mixture sweep.

## LLMTrace held-out domains

| Domain | Previous v4 recall | 5k recall | 10k recall | 20k recall | 5k × 4 recall | 20k human FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| article | 52.8% | 45.7% | 59.3% | 61.2% | 62.2% | 0.00% |
| factual | 55.5% | 47.8% | 61.4% | 65.1% | 64.6% | 0.00% |
| news | 52.3% | 44.8% | 53.8% | 58.9% | 59.8% | 0.00% |
| paper_abstract | 65.3% | 52.1% | 72.4% | 74.7% | 81.2% | 0.01% |
| poetry | 66.4% | 65.4% | 77.0% | 76.1% | 75.9% | 0.06% |
| question | 57.9% | 50.3% | 60.0% | 63.2% | 65.9% | 0.00% |
| review | 54.6% | 49.1% | 62.4% | 67.0% | 69.9% | 0.00% |
| short_form | 74.8% | 64.1% | 72.1% | 79.0% | 78.3% | 0.00% |
| story | 57.5% | 51.4% | 59.8% | 63.2% | 64.0% | 0.01% |

*ROC-interpolated recall at the previous v4 checkpoint's 0.67% LLMTrace human-token FPR. This is a retrospective test-set tradeoff, not a deployable threshold. AUROC and recall at the frozen threshold answer different questions. The ROC chart shows the available recall/FPR tradeoff, while the other table columns show the prespecified calibration rule.

A recall near 50% alone is not equivalent to random chance. Chance-level scores have AUROC around 0.5; a random detector that labels half the AI tokens also labels about half the human tokens. Report recall together with human false-positive rate, and inspect mixed-document recall separately because pure-AI documents can dominate the aggregate.

The frozen pure-human calibration set contains social Q&A, professional finance, and creative writing. It does not cover all nine LLMTrace domain labels. Large differences between AUROC and recall at its calibrated threshold may therefore reflect score calibration across domains; a broader independent human calibration set is the next threshold study.

The 20k tier consists of 4,964 unique earlier synthetic composites and 15,036 substantial English LLMTrace documents. The 5k and 10k tiers are subsets of it. Labeled AI characters comprise 48.3–48.6% across tiers. Train, validation, and test texts are exact-hash disjoint; LLMTrace topic groups overlapping the earlier validation and frozen diverse test were excluded. These experiments do not establish a 50k-data result or guarantee generalization beyond the tested generators and domains.
