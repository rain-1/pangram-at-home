# What the current detector does, and the source-balanced retrain

See [the four-page chart](plain_results_v6.pdf). The charts compare the **existing Qwen 20k checkpoint** with Pangram EditLens RoBERTa and Llama at thresholds fixed on an independent 1,120-document human calibration set. They are not results from the new training run.

## Plain-language verdict

- **Whole documents:** Our model catches 150/150 AI articles, but falsely flags 74/150 attributed-human articles (49.3%). The Pangram RoBERTa and Llama baselines falsely flag 1/150 and 7/150. That is a serious source-transfer failure.
- **Mixed documents:** The current model highlights 24.9% of AI tokens in LLMTrace mixtures, 67.2% in synthetic v4 mixtures, 86.6% in AITDNA real collaboration, and 0% in CoAuthor. It falsely highlights 0.02%, 0.15%, 15.2%, and 0% of human tokens in those respective sets. It is **not dependable as a general passage highlighter**. AITDNA performance is encouraging, but one successful dataset does not cancel the failures on others.
- **Span success:** Requiring at least half of an AI span to be highlighted, the current model finds 14.8% on LLMTrace, 61.6% on synthetic v4, 50.8% on AITDNA, and 0% on CoAuthor. CoAuthor AI inserts are unusually short, which makes it a particularly hard test.
- **ROC:** On published articles, Qwen's document AUROC is 0.937 versus 0.999 for RoBERTa and approximately 1.000 for Llama. This means threshold adjustment alone cannot close the entire gap in ranking quality.

The external 150-human/150-AI article set has attributed human authors, but its writing workflows were not independently verified as AI-free. It remains a frozen stress test, not calibration or training material. The mixed-span baseline scores come from window classifiers broadcast to tokens, so their localization is inherently coarse.

A diagnostic check on the old checkpoint shows that changing the whole-document score aggregator does not by itself repair this failure. With each aggregator calibrated to 5% false alarms on the same independent human controls, using the maximum, 95th percentile, 90th percentile, or mean token score falsely flags 74, 75, 74, or 72 of the 150 human articles respectively; all four catch 150/150 AI articles. This is evidence of a broad source/style error rather than a few stray high-scoring tokens. These aggregators were inspected on the article stress set and are **not** new validated operating rules.

## What we changed in the data

The old 20,000-document mix used 15,036 LLMTrace documents (75.2%), accounting for 71.1% of its training windows. The new [source-capped builder](../scripts/build_span_balanced_v6.py) produced a fresh 20,000-document mixture:

| Component | Documents | Share |
| --- | ---: | ---: |
| 16 named families of original pure human/AI text | 10,000 | 50% |
| Synthetic mixtures from the same named families | 4,000 | 20% |
| DAMASHA clean mixed documents | 5,400 | 27% |
| LLMTrace | 600 | **3%** |

LLMTrace contributes **2.43% of 512-token/256-stride training windows**. The largest single named source is DAMASHA at 27%; among the 16 named families, `mage:sci` is highest at about 12%. The new set contains 5,925 pure human, 5,912 pure AI, and 8,163 mixed documents. Labeled characters are 47.7% human and 52.3% AI. Exact duplicate documents were removed. The 41 DAMASHA candidates with sampled 24-word overlap against protected evaluations were excluded before selection.

The source cap is real, but diversity still has limits. The 4,000 synthetic examples reuse passages from the 10,000 pure-parent pool, and DAMASHA does not provide its original document or prompt IDs. We therefore cannot prove full source independence for DAMASHA using the published aggregate alone. We kept AITDNA and CoAuthor for evaluation and did not train on the external article stress set.

A subsequent read-only check found zero exact-text and zero sampled 24-word phrase matches between the 20,000 training records and each of the external articles, locked human, CoAuthor, AITDNA, and LLMTrace held-out sets. This screens direct text reuse; it does not establish independent provenance where source IDs are absent.

## How we are testing the false-positive fix

The new Qwen Repeat2 run uses the same base model, tuned LoRA hyperparameters, and initialization adapter as the previous 20k run; its principal change is the data mixture. It reports training to W&B. The [run controller](../scripts/run_span_balanced_v6.py) evaluates the new checkpoint at a threshold set on the separate 1,120-document human calibration set, then checks LLMTrace, synthetic validation, external articles, locked human documents, AITDNA, and CoAuthor.

We also prepared 200 pre-2023 open-license PMC article bodies for **publication calibration** and 346 distinct PMC articles for a locked human-publication test. Their PMC article IDs are excluded from every diverse-pyramid train/validation/test split; records with sampled 24-word overlap against selected DAMASHA/LLMTrace training records were also removed. These provide a source-aware threshold check without touching the 150-article external stress set. Threshold adjustments must report the resulting AI-recall loss.

**Success gate:** reducing the 49.3% external article false-positive rate matters most. We should compare article FPR, article AI recall, article AUROC, human-token false highlights, and mixed AI-span detection at the same time. If article FPR remains high, the next data change should add more verified pre-2023 published nonfiction to training, and calibrate by source/length without using either locked publication test.
