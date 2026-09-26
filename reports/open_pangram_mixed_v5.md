# Mixed-document comparison with Open Pangram EditLens

EditLens is a document/window classifier. Its scores are broadcast across overlapping source-token windows and averaged where windows overlap. This gives a coarse span baseline, not a native EditLens token prediction. Our Qwen model directly predicts token labels. A threshold for each model was frozen using the same separate 1,120 pure-human calibration documents at no more than 5% document-any false highlights.

**Interpretation limit:** Our 20k model trained on 15,036 LLMTrace train records; LLMTrace mixed test is from the same corpus, although train/test texts and group IDs are disjoint. AITDNA and CoAuthor are different sources. CoAuthor AI insertions are often very short and every model misses them at these thresholds.

| Set | Model | AI-token recall | Human-token FPR | Token AUROC |
| --- | --- | ---: | ---: | ---: |
| LLMTrace mixed | Our 20k token model | 24.9% | 0.02% | 0.962 |
| AITDNA mixed | Our 20k token model | 86.6% | 15.23% | 0.920 |
| CoAuthor mixed | Our 20k token model | 0.0% | 0.00% | 0.759 |
| LLMTrace mixed | EditLens RoBERTa windows | 18.1% | 5.52% | 0.642 |
| AITDNA mixed | EditLens RoBERTa windows | 78.6% | 34.54% | 0.772 |
| CoAuthor mixed | EditLens RoBERTa windows | 0.0% | 0.00% | 0.476 |
| LLMTrace mixed | EditLens Llama windows | 37.0% | 15.35% | 0.656 |
| AITDNA mixed | EditLens Llama windows | 95.5% | 66.20% | 0.753 |
| CoAuthor mixed | EditLens Llama windows | 0.0% | 0.00% | 0.512 |

The two EditLens checkpoints are [RoBERTa-large](https://huggingface.co/pangram/editlens_roberta-large) and [Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B).
