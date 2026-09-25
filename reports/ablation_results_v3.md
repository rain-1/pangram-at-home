# Dataset mixture pilots — diverse v3

Nine completed training runs, no Ray trial errors. All used the selected tuning parameters, 4,000 balanced training rows, 12,800 source examples seen, the same Qwen backbone/seed, and the same 800-row validation set. The control is 35% paper data within each class.

| Mixture | Partial AUROC at ≤5% FPR | Change vs control | AI recall at validation ≤2% FPR | Worst-domain recall |
| --- | ---: | ---: | ---: | ---: |
| Control: 35% paper | 95.99% | +0.00 pp | 91.25% | 83.75% |
| 20% paper | 96.15% | +0.17 pp | 92.75% | 88.75% |
| 50% paper | 95.81% | -0.18 pp | 92.00% | 85.00% |
| Without papers | 91.10% | -4.88 pp | 84.00% | 77.14% |
| Without creative writing | 91.46% | -4.53 pp | 86.00% | 75.00% |
| Without reference / education | 95.18% | -0.80 pp | 91.25% | 82.50% |
| Without social / Q&A | 94.26% | -1.72 pp | 89.75% | 85.00% |
| Without reviews | 94.85% | -1.14 pp | 91.00% | 82.50% |
| Without news | 95.15% | -0.84 pp | 91.00% | 82.50% |

Removing papers or creative writing produced the largest drops on this validation set. Removing the other categories also lowered partial AUROC, but by less. Moving paper share from 35% to 20% or 50% changed partial AUROC by under 0.2 percentage points. Those small differences should not select a final mix by themselves.

This study compares mixtures after a fixed example budget, not a source's intrinsic quality. Removing one category changes the proportions of the others. Every row comes from existing training sources, and all nine runs use one seed and the same validation examples. Selection and repeated inspection make these development results optimistic. Confirm promising mixes on independent source families, with a fresh blind test after choosing the design.

See [chart](ablation_results_v3.pdf) and the private run summaries on F: for full metrics and adapters.
