# Wider comparison of the three Qwen passage classifiers

All figures use each model's frozen threshold chosen on the same diverse validation split for nominal 2% human false-positive rate. No threshold was tuned on these evaluation sets. Scores refer to the first 512 source tokens (Repeat2 sees two copies of those tokens).

| Model | Quantization | Microbatch × accumulation | Effective batch | Validation logit-margin threshold |
|---|---|---:|---:|---:|
| Vast single | BF16, unquantized | 2 × 4 | 8 | -0.469 |
| Local single | BF16, unquantized | 1 × 8 | 8 | -1.717 |
| Local Repeat2 | BF16, unquantized | 1 × 8 | 8 | -1.086 |

All three runs share the same training and validation file hashes, seed 42, learning rate, LoRA rank/dropout, 512 source-token cap, and 3,200 optimizer-step budget. The local microbatch change was made without a memory probe. It preserves effective batch but means Vast versus local single is not an exact training replication. The binary sequence-classification Repeat2 run also does not implement [Pangram 4's tokenwise Repeat2 objective](https://pangram-public.s3.us-east-1.amazonaws.com/pdf/pangram_4_technical_report.pdf).

## Local microbatch-2 memory check

A subsequent one-step BF16 LoRA backward and AdamW optimizer probe on the 16 GiB RTX 4080 succeeded with the sweep's microbatch 2 for both input lengths. The probe used synthetic maximum-length inputs; it does not establish sustained-run memory behavior, but the original microbatch reduction had no measured need.

| Input mode | Model tokens/example | PyTorch peak reserved |
|---|---:|---:|
| Single copy | 512 | 4.23 GiB |
| Repeat2 | 1024 | 4.55 GiB |

## Results at the fixed operating point

False-positive rate (FPR) measures human text incorrectly called AI; AI recall measures AI text correctly called AI. A dash means the set has only human examples.

| Dataset | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diverse held-out | 500 / 500 | 2.2% | 3.2% | 5.2% | 93.4% | 93.6% | 93.8% |
| RAID | 800 / 800 | 0.8% | 1.4% | 0.9% | 85.8% | 86.6% | 85.4% |
| Enron | 1800 / 1800 | 2.6% | 4.1% | 4.4% | 99.9% | 99.9% | 99.9% |
| GPT-4 | 600 / 600 | 0.7% | 1.5% | 0.7% | 95.7% | 96.5% | 95.8% |
| Paraphrased AI | 600 / 600 | 0.7% | 1.5% | 0.5% | 37.0% | 31.7% | 28.7% |
| Standard Ebooks | 1860 / 0 | 0.0% | 0.0% | 0.1% | — | — | — |
| Persuade essays | 1000 / 0 | 1.4% | 0.9% | 0.6% | — | — | — |
| Federal Reserve | 1622 / 0 | 1.7% | 0.1% | 0.9% | — | — | — |
| Stack Exchange | 1000 / 0 | 4.6% | 4.4% | 5.4% | — | — | — |
| PMC full body | 261 / 0 | 2.7% | 3.1% | 1.5% | — | — | — |
| EditLens original | 1878 / 1879 | 0.5% | 1.0% | 0.8% | 99.8% | 99.8% | 99.9% |
| EditLens Llama | 1916 / 1917 | 0.5% | 0.9% | 0.8% | 100.0% | 100.0% | 100.0% |
| MAGE holdout | 1520 / 1520 | 9.4% | 11.4% | 15.2% | 78.3% | 82.2% | 83.5% |
| Paper generator swap | 82 / 82 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## Ranking quality (AUROC)

AUROC uses every possible threshold; partial AUROC focuses on the 0–5% false-positive region. Neither replaces the fixed-threshold error rates above.

| Dataset | Vast AUROC / partial | Local single AUROC / partial | Repeat2 AUROC / partial |
|---|---:|---:|---:|
| Diverse held-out | 0.993 / 0.959 | 0.992 / 0.957 | 0.990 / 0.954 |
| RAID | 0.977 / 0.942 | 0.980 / 0.942 | 0.975 / 0.944 |
| Enron | 1.000 / 0.999 | 1.000 / 0.998 | 1.000 / 0.999 |
| GPT-4 | 0.992 / 0.983 | 0.992 / 0.982 | 0.995 / 0.985 |
| Paraphrased AI | 0.797 / 0.705 | 0.788 / 0.661 | 0.770 / 0.676 |
| EditLens original | 1.000 / 1.000 | 1.000 / 0.999 | 1.000 / 1.000 |
| EditLens Llama | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| MAGE holdout | 0.924 / 0.801 | 0.932 / 0.790 | 0.925 / 0.788 |
| Paper generator swap | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

The two EditLens sets reuse source prompts and must not be pooled as independent evidence. For the four new probes, source IDs and normalized exact texts occurring in any diverse train, validation, or test split were removed. MAGE source IDs are hashes of individual texts, so this exclusion establishes exact-text separation but not independent prompts or authors. The EditLens Llama probe includes only `human_written` and `ai_generated`; assisted edits were excluded because the current models are binary classifiers. MAGE contains some closed-ended source tasks and strongly mismatched human/AI lengths; interpret its source breakdown and aggregate with those caveats.

## Source breakdown on the diverse held-out set

| Source | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| editlens:amazon_reviews | 17 / 17 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| editlens:fineweb_edu | 34 / 34 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| editlens:google_reviews | 17 / 17 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| editlens:news | 13 / 13 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| editlens:reddit_writing_prompts | 40 / 40 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| mage:cmv | 25 / 25 | 0.0% | 0.0% | 4.0% | 96.0% | 96.0% | 96.0% |
| mage:eli5 | 33 / 33 | 3.0% | 6.1% | 6.1% | 78.8% | 78.8% | 87.9% |
| mage:roct | 21 / 21 | 9.5% | 14.3% | 14.3% | 76.2% | 81.0% | 71.4% |
| mage:sci | 57 / 57 | 7.0% | 8.8% | 12.3% | 89.5% | 87.7% | 93.0% |
| mage:squad | 33 / 33 | 6.1% | 9.1% | 9.1% | 90.9% | 87.9% | 81.8% |
| mage:tldr | 25 / 25 | 4.0% | 8.0% | 28.0% | 88.0% | 92.0% | 92.0% |
| mage:wp | 39 / 39 | 0.0% | 0.0% | 0.0% | 97.4% | 100.0% | 97.4% |
| mage:xsum | 12 / 12 | 0.0% | 0.0% | 8.3% | 100.0% | 100.0% | 100.0% |
| mage:yelp | 16 / 16 | 0.0% | 6.2% | 6.2% | 56.2% | 56.2% | 56.2% |
| paper:acl_anthology | 59 / 59 | 0.0% | 0.0% | 1.7% | 100.0% | 100.0% | 100.0% |
| paper:pmc_oa | 59 / 59 | 1.7% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## New-probe source breakdown

| Dataset / source | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EditLens original / amazon_reviews | 253 / 243 | 0.0% | 0.8% | 0.4% | 99.6% | 99.2% | 99.6% |
| EditLens original / fineweb_edu | 474 / 426 | 1.1% | 1.3% | 1.1% | 99.8% | 100.0% | 100.0% |
| EditLens original / google_reviews | 188 / 191 | 1.6% | 3.2% | 3.7% | 100.0% | 99.5% | 100.0% |
| EditLens original / news | 472 / 499 | 0.0% | 0.6% | 0.4% | 100.0% | 100.0% | 100.0% |
| EditLens original / reddit_writing_prompts | 491 / 520 | 0.2% | 0.2% | 0.0% | 99.6% | 99.8% | 99.8% |
| EditLens Llama / amazon_reviews | 260 / 260 | 0.0% | 0.8% | 0.4% | 100.0% | 100.0% | 100.0% |
| EditLens Llama / fineweb_edu | 487 / 487 | 1.0% | 1.2% | 1.0% | 100.0% | 100.0% | 100.0% |
| EditLens Llama / google_reviews | 190 / 190 | 1.6% | 3.2% | 3.7% | 100.0% | 100.0% | 100.0% |
| EditLens Llama / news | 480 / 480 | 0.0% | 0.6% | 0.4% | 100.0% | 100.0% | 100.0% |
| EditLens Llama / reddit_writing_prompts | 499 / 500 | 0.2% | 0.2% | 0.0% | 100.0% | 100.0% | 100.0% |
| MAGE holdout / cmv | 155 / 155 | 0.6% | 0.0% | 0.0% | 76.1% | 78.1% | 83.2% |
| MAGE holdout / eli5 | 140 / 140 | 4.3% | 4.3% | 6.4% | 73.6% | 75.0% | 77.1% |
| MAGE holdout / hswag | 200 / 200 | 50.5% | 59.0% | 73.0% | 87.5% | 88.5% | 88.5% |
| MAGE holdout / roct | 157 / 157 | 7.0% | 8.9% | 14.0% | 79.6% | 87.9% | 86.6% |
| MAGE holdout / sci | 96 / 96 | 1.0% | 3.1% | 4.2% | 94.8% | 93.8% | 97.9% |
| MAGE holdout / squad | 141 / 141 | 3.5% | 3.5% | 6.4% | 73.8% | 71.6% | 80.1% |
| MAGE holdout / tldr | 155 / 155 | 5.8% | 9.0% | 15.5% | 88.4% | 92.9% | 94.2% |
| MAGE holdout / wp | 127 / 127 | 0.0% | 0.8% | 0.0% | 74.0% | 84.3% | 80.3% |
| MAGE holdout / xsum | 178 / 178 | 0.0% | 0.6% | 0.6% | 75.3% | 77.5% | 78.7% |
| MAGE holdout / yelp | 171 / 171 | 5.3% | 7.0% | 9.4% | 63.7% | 74.9% | 72.5% |
| Paper generator swap / acl_anthology | 57 / 57 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| Paper generator swap / pmc_oa | 25 / 25 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## Length breakdown on new probes

| Dataset / characters | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAGE holdout / under_500_chars | 524 / 584 | 25.6% | 30.3% | 39.5% | 84.1% | 86.5% | 88.2% |
| MAGE holdout / 500_to_999_chars | 357 / 259 | 2.5% | 3.9% | 6.4% | 85.3% | 91.9% | 88.4% |
| MAGE holdout / 1000_to_1999_chars | 337 / 261 | 0.0% | 0.0% | 0.3% | 78.5% | 81.2% | 82.0% |
| MAGE holdout / 2000_plus_chars | 302 / 416 | 0.0% | 0.3% | 0.0% | 65.6% | 70.7% | 74.8% |
| EditLens original / under_500_chars | 159 / 11 | 3.1% | 5.7% | 5.0% | 100.0% | 100.0% | 100.0% |
| EditLens original / 500_to_999_chars | 443 / 161 | 0.5% | 1.1% | 1.4% | 99.4% | 98.1% | 99.4% |
| EditLens original / 1000_to_1999_chars | 478 / 437 | 0.4% | 0.4% | 0.2% | 99.8% | 99.8% | 99.8% |
| EditLens original / 2000_plus_chars | 798 / 1270 | 0.0% | 0.3% | 0.0% | 99.8% | 100.0% | 100.0% |

## Paired human-error counts

These counts use the same human rows for each pair. `A only` means model A falsely called AI while model B did not; `B only` is the reverse.

| Dataset | Pair (A / B) | A only | B only |
|---|---|---:|---:|
| Diverse held-out | Vast single / Local single | 2 | 7 |
| Diverse held-out | Vast single / Local Repeat2 | 1 | 16 |
| Diverse held-out | Local single / Local Repeat2 | 6 | 16 |
| MAGE holdout | Vast single / Local single | 19 | 50 |
| MAGE holdout | Vast single / Local Repeat2 | 13 | 101 |
| MAGE holdout | Local single / Local Repeat2 | 23 | 80 |
| RAID | Vast single / Local single | 3 | 8 |
| RAID | Vast single / Local Repeat2 | 3 | 4 |
| RAID | Local single / Local Repeat2 | 7 | 3 |
| Paraphrased AI | Vast single / Local single | 1 | 6 |
| Paraphrased AI | Vast single / Local Repeat2 | 2 | 1 |
| Paraphrased AI | Local single / Local Repeat2 | 6 | 0 |

## Interpretation

The main diverse held-out set contains only 500 human examples, so a few additional false positives move FPR by whole percentage points. Reported gaps should be read alongside the larger human-only sets and the per-source rows. Very high scores on datasets with similar generation pipelines are weaker evidence of real-world generalization than the harder RAID and paraphrase sets.

On MAGE, under-500-character human FPR is Vast single 25.6%, Local single 30.3%, Local Repeat2 39.5%. The HSwag subset contains short, formatted human passages and accounts for Vast single 101, Local single 118, Local Repeat2 146 human false positives out of 200 per model. This is an observed concentration, not proof that length or formatting alone causes the errors.

Paraphrased AI recall is Vast single 37.0%, Local single 31.7%, Local Repeat2 28.7%; this remains a major false-negative weakness even where ordinary AI recall is high.

The current window classifier provides one score per 512-token window. It does not label tokens or identify exact authorship boundaries. On long documents, overlapping windows can produce a coarse heatmap; full-paper human audits are in `reports/metrics/*_pmc_fullpaper.json`.

## Full-paper human-only audit

The same 100 source-disjoint PMC papers contain 685,096 source tokens. Windows use size 512 and stride 256; a token is falsely highlighted if a covering window crosses that model's frozen threshold.

| Model | Human tokens falsely highlighted | Papers with any false highlight |
|---|---:|---:|
| Vast single | 0.8% | 9 / 100 |
| Local single | 1.8% | 17 / 100 |
| Local Repeat2 | 1.2% | 15 / 100 |

## Provenance

- EditLens original: 3757 rows after filtering; 242 source-overlap and 1 exact-text-overlap rows removed; 2513 source IDs; SHA256 `7300b92f24b7a822f96d7e24aaafd76530633e0e6b7dc7a1b023f055856dabf8`.
- EditLens Llama: 3833 rows after filtering; 242 source-overlap and 1 exact-text-overlap rows removed; 1917 source IDs; SHA256 `69171eea06694945fce4b9963779af269b6e2a4dbd4a2660091ea12a36586034`.
- MAGE holdout: 3040 rows after filtering; 960 source-overlap and 0 exact-text-overlap rows removed; 3040 source IDs; SHA256 `8bfdaeb2b1a99c1af73971128e302d0b3428404daf7cdf821180f96fdf33a977`.
- Paper generator swap: 164 rows after filtering; 200 source-overlap and 0 exact-text-overlap rows removed; 82 source IDs; SHA256 `9f05ae8a68abd320598103fd3aacdc3b736bad863074b50fda3d155d9d91b04e`.
