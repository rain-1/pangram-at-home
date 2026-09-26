# Span localization: v4 and baselines

All methods use a separately calibrated threshold allowing 5% of documents in the same 1,120-document pure-human calibration set to receive any false highlight. All methods are then evaluated without threshold adjustment. Load Bearing is omitted from this new comparison as requested. The char and word TF-IDF baselines train on 10,000 labeled passage documents and broadcast sliding-window predictions across tokens; the Qwen passage baseline uses the selected Vast checkpoint the same way.

| Model | V4 synthetic AI token recall | V4 synthetic human token FPR | AITDNA mixed AI token recall | AITDNA mixed human token FPR | Locked human docs with any false highlight | CoAuthor AI token recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Token v4 | 73.6% | 0.0% | 88.4% | 18.0% | 0.7% | 3.3% |
| Token v3 | 18.0% | 0.0% | 39.1% | 2.5% | 0.0% | 0.0% |
| Qwen passage | 79.5% | 19.2% | 96.2% | 73.4% | 2.0% | 2.0% |
| Char TF-IDF | 43.6% | 5.8% | 57.1% | 30.2% | 4.3% | 0.2% |
| Word TF-IDF | 44.3% | 6.9% | 57.1% | 34.9% | 2.2% | 0.0% |

The synthetic set contains 600 documents; the locked human set contains 3,579. AITDNA has 258 mixed documents; its mixed human-token FPR is the most direct indicator of false span marking within actual collaborative writing. CoAuthor has short accepted suggestions and substantial provenance masking, so its token recall tests a harder and narrower use case than paragraph localization.

The passage Qwen baseline has strong AI recall but marks substantially more human material inside mixed documents. V4 is the clearest overall tradeoff at this fixed calibration rule. These are descriptive rates on fixed datasets, not a guarantee for unseen authors, prompts, or generator models.
