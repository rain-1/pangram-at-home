# Diverse baseline results

All thresholds were chosen on the 800-row diverse validation split to allow at most 2% human false positives.
The 1,000-row mixed test contains six writing categories. RAID is an independent source-family test (1,600 rows, eight domains, 11 generators).

| Baseline | Mixed AUROC | Mixed FPR | Mixed AI recall | RAID AUROC | RAID FPR | RAID AI recall | Enron AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Character TF-IDF | 0.882 | 1.8% | 46.0% | 0.826 | 0.8% | 25.5% | 0.940 |
| Word TF-IDF | 0.862 | 1.0% | 41.2% | 0.803 | 0.5% | 24.6% | 0.884 |
| MiniLM + logistic | 0.696 | 1.6% | 9.6% | 0.697 | 0.5% | 7.4% | 0.813 |
| Load Bearing PR cluster | 0.481 | 3.6% | 3.6% | 0.513 | 6.4% | 4.6% | 0.296 |

Human-only false-positive audits:

| Audit | Rows | Character | Word | MiniLM |
| --- | ---: | ---: | ---: | ---: |
| Classic fiction | 1860 | 0.3% | 0.0% | 0.4% |
| Student essays | 1000 | 1.2% | 0.1% | 4.6% |
| Federal Reserve | 1622 | 0.9% | 0.8% | 2.3% |
| Writers Stack Exchange | 1000 | 3.8% | 1.1% | 6.7% |
| PMC full-body prose | 261 | 1.5% | 0.4% | 1.1% |

The Load Bearing score is a vocabulary-cluster transfer probe, not a purpose-trained AI-authorship detector.
EditLens reference checkpoints and Qwen are scored after the GPU training run.
See [the PDF](diverse_baselines_v1.pdf) for per-domain AUROC and AI recall.
