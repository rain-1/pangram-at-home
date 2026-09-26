# Binary token-localization pilot

The models predict **human** or **AI-generated** at each token. They do not predict AI-assisted text. The token models use a Qwen3-1.7B backbone with the selected Vast learning rate 7.607757e-5, LoRA rank 32/alpha 64/dropout 0.068837, attention and feed-forward targets, BF16, microbatch 2 × accumulation 4, 512 source tokens, 256 stride, and 300 steps. Both start with the same selected Vast backbone adapter; their token heads are newly initialized. Repeat2 masks all first-copy token labels and supervises the second copy. W&B logged both runs.

Data: 2400 synthetic training documents and 400 validation documents, with 500 source-group-disjoint confirmation documents and 120 long composites. Training and validation cover paper, reviews, creative writing, news, reference/education, and social Q&A. Joins use exact character offsets and varied separators; the underlying source text stays on the external drive.

| Model | Input / output | W&B |
|---|---|---|
| Vast window | Single 512-token input; one score per window broadcast over its tokens | Prior HPO run |
| Token single | One input copy; token scores | [run](https://wandb.ai/eac-adsf/pangram-at-home/runs/2jrklgmg) |
| Token Repeat2 | Two input copies; second-copy token scores | [run](https://wandb.ai/eac-adsf/pangram-at-home/runs/053oc7dr) |

## Whole-document scores at frozen thresholds

Each model's threshold is set to 2% false positives on pure-human *validation tokens*, then held fixed for the confirmation and long-document diagnostics. FPR and recall below include all labeled tokens; AI-span recall asks whether at least half the characters in a known AI span are highlighted.

| Set | Model | Human token FPR | AI token recall | Mixed AI spans ≥half covered | Fully human docs with any false highlight | Mixed AI-fraction MAE |
|---|---|---:|---:|---:|---:|---:|
| Validation | Vast window | 31.0% | 59.7% | 50.0% | 4.5% | 0.449 |
| Validation | Token single | 3.7% | 72.6% | 79.9% | 47.8% | 0.121 |
| Validation | Token Repeat2 | 3.1% | 86.8% | 89.8% | 11.9% | 0.065 |
| Confirmation | Vast window | 28.3% | 60.5% | 45.4% | 6.0% | 0.455 |
| Confirmation | Token single | 4.1% | 71.8% | 80.8% | 46.4% | 0.134 |
| Confirmation | Token Repeat2 | 3.6% | 89.3% | 94.0% | 17.9% | 0.065 |
| Long composite | Vast window | 14.3% | 60.5% | 35.9% | 0.0% | 0.317 |
| Long composite | Token single | 4.3% | 63.2% | 65.6% | 100.0% | 0.134 |
| Long composite | Token Repeat2 | 1.4% | 75.4% | 70.1% | 33.3% | 0.116 |

## Stricter document-safe operating point

For each token model, a second threshold allows at most 3 of 67 fully human validation documents to receive any highlight (4.5%). It is frozen for the other sets. This is a development calibration, not a guarantee on new writing.

| Set | Token model | Human token FPR | AI token recall | Fully human docs with any false highlight |
|---|---|---:|---:|---:|
| Validation | Token single | 0.5% | 54.5% | 4.5% |
| Validation | Token Repeat2 | 1.7% | 82.7% | 4.5% |
| Confirmation | Token single | 0.5% | 50.8% | 6.0% |
| Confirmation | Token Repeat2 | 1.7% | 84.2% | 7.1% |
| Long composite | Token single | 0.6% | 45.2% | 56.7% |
| Long composite | Token Repeat2 | 0.6% | 69.2% | 10.0% |

## Confirmation-set source construction

Matched pairs share a source group/prompt where available. Unmatched same-source joins also contain MAGE sources absent from the matched stratum, so construction and source family are confounded.

| Construction | Model | Human FPR | AI recall | Documents |
|---|---|---:|---:|---:|
| matched_source_pair | Vast window | 36.7% | 49.4% | 249 |
| matched_source_pair | Token single | 4.3% | 81.3% | 249 |
| matched_source_pair | Token Repeat2 | 1.9% | 93.5% | 249 |
| unaltered_source_excerpt | Vast window | 2.1% | 90.3% | 168 |
| unaltered_source_excerpt | Token single | 2.4% | 64.9% | 168 |
| unaltered_source_excerpt | Token Repeat2 | 2.1% | 86.2% | 168 |
| unmatched_same_source_join | Vast window | 42.4% | 47.5% | 83 |
| unmatched_same_source_join | Token single | 6.4% | 47.0% | 83 |
| unmatched_same_source_join | Token Repeat2 | 14.2% | 77.7% | 83 |

### Construction × source family

| Construction / family | Model | Human FPR | AI recall | Documents |
|---|---|---:|---:|---:|
| matched_source_pair:editlens | Vast window | 39.5% | 49.7% | 169 |
| matched_source_pair:editlens | Token single | 4.3% | 78.7% | 169 |
| matched_source_pair:editlens | Token Repeat2 | 2.0% | 92.5% | 169 |
| matched_source_pair:paper | Vast window | 30.9% | 48.7% | 80 |
| matched_source_pair:paper | Token single | 4.3% | 87.6% | 80 |
| matched_source_pair:paper | Token Repeat2 | 1.7% | 96.2% | 80 |
| unaltered_source_excerpt:editlens | Vast window | 1.4% | 93.6% | 45 |
| unaltered_source_excerpt:editlens | Token single | 1.9% | 81.6% | 45 |
| unaltered_source_excerpt:editlens | Token Repeat2 | 0.3% | 93.4% | 45 |
| unaltered_source_excerpt:mage | Vast window | 3.7% | 84.5% | 99 |
| unaltered_source_excerpt:mage | Token single | 3.4% | 45.0% | 99 |
| unaltered_source_excerpt:mage | Token Repeat2 | 4.9% | 77.8% | 99 |
| unaltered_source_excerpt:paper | Vast window | 0.0% | 100.0% | 24 |
| unaltered_source_excerpt:paper | Token single | 1.0% | 89.7% | 24 |
| unaltered_source_excerpt:paper | Token Repeat2 | 0.0% | 96.7% | 24 |
| unmatched_same_source_join:editlens | Vast window | 17.5% | 29.1% | 17 |
| unmatched_same_source_join:editlens | Token single | 3.3% | 82.6% | 17 |
| unmatched_same_source_join:editlens | Token Repeat2 | 2.7% | 96.3% | 17 |
| unmatched_same_source_join:mage | Vast window | 48.8% | 49.7% | 55 |
| unmatched_same_source_join:mage | Token single | 7.5% | 26.3% | 55 |
| unmatched_same_source_join:mage | Token Repeat2 | 22.7% | 67.1% | 55 |
| unmatched_same_source_join:paper | Vast window | 51.8% | 68.2% | 11 |
| unmatched_same_source_join:paper | Token single | 6.2% | 80.6% | 11 |
| unmatched_same_source_join:paper | Token Repeat2 | 0.4% | 94.4% | 11 |

## Confirmation-set domains

| Domain | Model | Human FPR | AI recall | Documents |
|---|---|---:|---:|---:|
| creative | Vast window | 16.0% | 51.7% | 78 |
| creative | Token single | 5.0% | 60.6% | 78 |
| creative | Token Repeat2 | 9.5% | 81.4% | 78 |
| news | Vast window | 28.0% | 71.9% | 61 |
| news | Token single | 5.2% | 72.9% | 61 |
| news | Token Repeat2 | 2.5% | 89.9% | 61 |
| paper | Vast window | 28.6% | 65.1% | 130 |
| paper | Token single | 4.2% | 83.1% | 130 |
| paper | Token Repeat2 | 1.4% | 94.9% | 130 |
| reference_education | Vast window | 42.3% | 68.1% | 90 |
| reference_education | Token single | 3.3% | 67.2% | 90 |
| reference_education | Token Repeat2 | 3.3% | 90.3% | 90 |
| reviews | Vast window | 21.1% | 43.0% | 106 |
| reviews | Token single | 2.2% | 75.7% | 106 |
| reviews | Token Repeat2 | 1.6% | 89.6% | 106 |
| social_qa | Vast window | 24.7% | 63.8% | 35 |
| social_qa | Token single | 6.5% | 29.7% | 35 |
| social_qa | Token Repeat2 | 11.6% | 70.9% | 35 |

## Learning curve

| Token model | Best step | Best validation partial AUROC ≤5% FPR | Training minutes | Peak PyTorch allocation |
|---|---:|---:|---:|---:|
| Token single | 225 | 0.829 | 11.0 | 4.06 GiB |
| Token Repeat2 | 225 | 0.913 | 12.4 | 4.39 GiB |

## Run inference

`scripts/predict_token_spans.py` reads a UTF-8 text file, scores overlapping 512-token windows, averages token logits, and writes character-offset human/AI spans. For this pilot's stricter document threshold:

```bash
python scripts/predict_token_spans.py --run-name qwen3_token_repeat2_v3_pilot1 --text-file /path/to/document.txt --threshold 1.8671876192092896 --output /path/to/spans.json
```

## What this pilot can establish

All mixed spans have known origin because they were assembled from separately labeled human and AI source text with recorded character offsets. The joins are synthetic and may reveal formatting or source cues. The confirmation set is source-group-disjoint from pilot train/validation, but its parent diverse test was examined in earlier passage-model work, so it is not a blind final test. The long composite set reuses validation excerpts and tests window stitching and length effects only. The Vast hyperparameters were tuned for passage classification, not this token objective. A realistic reviewed span set remains necessary before claiming real-world localization quality.

Raw text, token scores, and predicted spans stay on the external drive. This report contains aggregate metrics only.
