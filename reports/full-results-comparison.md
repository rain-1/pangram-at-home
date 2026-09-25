# Full pilot results comparison

![Comparison of all saved baseline results and the first Qwen3 LoRA run](charts/full_results.png)

[Download the vector PDF](charts/full_results.pdf) · [Download the plotted values as CSV](charts/full_results.csv)

The chart now shows **two cutoffs for the same Qwen3 LoRA model**. The original 3.7% AI-score cutoff detects 631/631 mixed-test AI texts but flags 19/631 humans (3.0%). The validation-selected 86.4% midpoint also detects 631/631 AI texts and flags 4/631 humans (0.6%). Its paper subset flags 2/221 humans (0.9%) and its PMC full-body audit flags 4/261 human chunks (1.5%). The [threshold sweep](threshold-tradeoff.md) explains the selection and the stricter option. The EditLens RoBERTa reference flags 8/631 humans (1.3%) on the mixed test, while detecting 617/631 AI texts (97.8%).

Each row uses its saved validation-selected threshold. The Qwen3 paper row is the paper subset of the mixed test at its **mixed** validation threshold. Paper-only baselines used paper validation; the PMC-only and EditLens-only sections likewise use their respective validation sets. The published reference checkpoints had substantially more external training data than the local baselines. In the general EditLens and Enron sections, the Llama reference was run on smaller 500-human/500-AI test tiers. The paper AI samples were generated from titles by two small local models; these results do not measure detection of frontier generators or edited writing.

The [load-bearing PR vocabulary row](load-bearing-transfer.md) uses a frozen, unsupervised GitHub writing-style model. Its score identifies membership in a particular PR vocabulary cluster; it was included to test transfer to our domains, not because its author claimed an authorship detector.

The original PMC-only test is a separate split: 61 of its 87 source papers occur in Qwen3's mixed training set and 15 in its validation set. A Qwen3 score on that split is excluded because it would not be a clean held-out comparison. The chart instead includes a **mixed-test PMC subset** with 87 papers held out from Qwen3 training, scored by Qwen3 and every mixed-set baseline.

We filled the other missing Qwen3 evaluations at the original mixed-validation threshold: **2,000/2,000 AI detected and 41/2,000 human false positives (2.05%)** on the general EditLens test; **1,800/1,800 AI detected and 38/1,800 human false positives (2.11%)** on Enron; and **383/17,560 human false positives (2.18%)** on the ACL abstract audit. At the midpoint cutoff, those false-positive counts fall to **7/2,000**, **2/1,800**, and **97/17,560**, with one missed AI text on each binary test.

The ACL audits use different human pools: paper and mixed model audits exclude every sampled paper work (17,560 abstracts), whereas the general EditLens model audits used 18,287 abstracts. The swapped-generator test reuses the same held-out human works, so its human false-positive result is correlated with the paper test. The source metrics are in [`metrics/`](metrics), and [`scripts/chart_full_results.py`](../scripts/chart_full_results.py) regenerates the chart and CSV.
