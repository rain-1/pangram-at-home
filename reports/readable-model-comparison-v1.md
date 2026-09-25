# Readable model comparison

Open the [eight-page chart PDF](readable_model_comparison_v1.pdf). The first page summarizes performance; pages 2–5 show category-level recall and false positives; pages 6–7 show ROC curves; page 8 shows additional human-only checks. Separate images: [overview](readable_overview_v1.png), [diverse ROC](readable_roc_diverse_v1.png), [RAID ROC](readable_roc_raid_v1.png).

## How to read the summary

The average gives **equal weight** to the 1,000-row diverse test and the 1,600-row independent RAID test. Both have equal numbers of human and AI passages. AI recall is the share of AI passages caught; human false-positive rate is the share of human passages wrongly flagged. Balanced accuracy is the mean of AI recall and human specificity. Every model's decision threshold was selected on the *same* diverse validation split with a target of at most 2% human false positives. No test labels were used to set thresholds.

| Model | Mean AI recall | Mean balanced accuracy | Mean human false positives |
| --- | ---: | ---: | ---: |
| Our Qwen model | **83.0%** | **90.8%** | 1.3% |
| Character TF-IDF | 35.8% | 67.2% | 1.3% |
| Word TF-IDF | 32.9% | 66.1% | **0.8%** |
| MiniLM + logistic | 8.5% | 53.7% | 1.1% |
| EditLens RoBERTa | 44.3% | 71.1% | 2.2% |
| EditLens Llama | 58.0% | 78.6% | **0.8%** |
| Load Bearing PR cluster | 4.1% | 49.6% | 5.0% |

These are averages of two test results, not estimates over all text on the internet. Load Bearing is a PR-language cluster probe, not a purpose-trained authorship detector.

## Where our model struggles

| Test | Category | AI recall | Human false positives | Passages per label |
| --- | --- | ---: | ---: | ---: |
| Diverse | Social / Q&A | 88% | **6%** | 50 |
| Diverse | Reviews | 86% | **4%** | 50 |
| RAID | Recipes | **60%** | 1% | 100 |
| RAID | News | **69%** | 0% | 100 |
| RAID | Reddit | **72%** | 1% | 100 |

The category charts include every model, including stronger Qwen categories. The diverse social/Q&A false-positive figure is three mistakes among 50 human examples, so the exact percentage is uncertain. On separate human-only audits, Qwen wrongly flagged 2.1% of Writers Stack Exchange posts and 2.3% of held-out PMC paper body passages. Those checks do not measure AI recall.

## What the ROC curves add

The ROC pages show the full threshold tradeoff for every model, with a second panel focused on 0–10% human false positives. Qwen's AUROC is 0.989 on the diverse test and 0.974 on RAID. A high AUROC means the model usually ranks AI above human passages; it does not guarantee a particular deployed false-positive rate. At the fixed validation-selected threshold, Qwen's observed false-positive rates are 2.2% on the diverse test and 0.4% on RAID, while AI recall is 89.2% and 76.8%, respectively.

False positives matter more when AI writing is rare. For illustration, if only 1% of passages were AI and these test rates held, approximately 29% of Qwen's positive flags would be correct under the diverse-test rates, versus approximately 67% under the RAID rates. These are scenario calculations, not measured real-world precision. They show why human review and setting-specific evaluation matter before acting on an individual flag.

RAID uses a distinct source family and is the stronger transfer check. The diverse test shares source families with training, though it uses held-out rows. Some human-source labels, notably older social posts, are inferred from provenance rather than verified author by author. Small category samples and possible near duplicates also limit certainty; see the [data report](diverse-data-v1.md).

The ROC script recreates per-example baseline scores from frozen inputs and checks each AUROC against the published metric before drawing a curve. Score caches stay on the external drive; the chart script is [cache_diverse_roc_scores.py](../scripts/cache_diverse_roc_scores.py) and [chart_readable_comparison.py](../scripts/chart_readable_comparison.py).
