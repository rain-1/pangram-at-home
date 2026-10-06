# Repeat2 span model: v4 data experiment

The v3 and v4 token models were calibrated separately on the same independent,
pure-human calibration set. Each threshold allows at most 5% of calibration
documents to receive any false highlight. The thresholds were frozen before
scoring the held-out human set. Both models use the same Qwen3-1.7B base,
Vast passage-adapter initialization, Repeat2, LoRA modules, learning rate,
and effective batch size. V4 trains on the expanded synthetic span mix.

| Metric | v3 Repeat2 | v4 Repeat2 |
| --- | ---: | ---: |
| Calibration threshold (logit margin) | 7.219 | 6.750 |
| Held-out human token FPR | 0.0% | 0.0% |
| Held-out human documents with any false highlight | 0.0% | 0.7% |
| V4 synthetic validation: AI token recall | 18.0% | 73.6% |
| V4 synthetic validation: human token FPR | 0.0% | 0.0% |
| Prior synthetic validation: AI token recall | 31.5% | 62.0% |
| Prior synthetic validation: human token FPR | 0.0% | 0.7% |
| Realistic mixed-document evaluation: AI token recall | 0.0% | 3.3% |
| Realistic mixed-document evaluation: human token FPR | 0.0% | 0.4% |

CoAuthor metrics score 51.3% of source tokens. Prompts, ambiguous pasted text, and edited AI regions are masked; these are insertion-provenance metrics, not complete assisted-writing labels.

## Held-out human breakdown

| Domain | v3 token FPR | v4 token FPR | v3 any-highlight | v4 any-highlight | Human documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| social_qa | 0.0% | 0.1% | 0.0% | 0.7% | 579 |
| student_argumentative_essay | 0.0% | 0.0% | 0.0% | 0.7% | 3000 |

## Human-document length

| Source-token length | v3 any-highlight | v4 any-highlight | Human documents |
| --- | ---: | ---: | ---: | ---: |
| 0001-0512 | 0.0% | 0.9% | 2134 |
| 0513-1024 | 0.0% | 0.3% | 1216 |
| 1025-2048 | 0.0% | 0.4% | 229 |

## Alternate operating point: 2% calibration token FPR

This is a retrospective threshold calculation from cached scores, not
a new training run. The threshold is set only on pure-human calibration
tokens and then applied unchanged to the other sets.

| Metric | v3 Repeat2 | v4 Repeat2 |
| --- | ---: | ---: |
| Threshold (logit margin) | 3.742 | 2.602 |
| Held-out human: human token FPR | 0.0% | 0.7% |
| Held-out human: any false highlight per document | 1.3% | 7.7% |
| V4 synthetic: human token FPR | 0.1% | 0.8% |
| V4 synthetic: AI token recall | 55.8% | 90.5% |
| CoAuthor: human token FPR | 0.0% | 4.4% |
| CoAuthor: AI token recall | 1.3% | 20.7% |

The held-out human set has 3,579 documents, with possible clustering
by author or writing prompt. The table gives descriptive rates rather than
uncertainty estimates across independent authors.

## Interpretation limits

The v4 training data is built from exact known-origin excerpts, but its long
mixed documents are synthetic joins. This experiment tests whether that
training improves localization and reduces false highlights. It does not
establish performance on reviewed real human edits of AI documents. The
human test data is independent by record from the span-training parent,
with provenance and rights limitations described in its manifest. No
threshold or checkpoint was selected using the locked human set.
