# External article stress test

The [Human Detectors dataset](https://github.com/jenna-russell/human_detectors) provides 150 human and 150 AI nonfiction articles. Our 20k model trained on LLMTrace and earlier synthetic sources; these articles are from a separate dataset. The local audit found no exact text or normalized 24-word shingle overlap with training. These are whole-article labels, so they do not test mixed-span localization.

Each model uses its previously fixed threshold, calibrated to at most 5% document false alarms on a separate set of 1,120 human documents. Our Qwen article score is its highest token logit; EditLens uses the mean of overlapping native-window scores. The thresholds and score aggregation were chosen before looking at this stress set. No thresholds were retuned here.

**Provenance limit:** Source articles have named authors and publication links, but their writing workflows were not independently verified as AI-free. Treat them as attributed-human articles, rather than a gold-standard known-human corpus.

| Model | AI detected / 150 | Human false alarms / 150 | Article AUROC |
| --- | ---: | ---: | ---: |
| Our Qwen 20k | 150/150 (100.0%) | 74/150 (49.3%) | 0.9370 |
| EditLens RoBERTa | 146/150 (97.3%) | 1/150 (0.7%) | 0.9989 |
| EditLens Llama | 150/150 (100.0%) | 7/150 (4.7%) | 0.9999 |

Our Qwen model highlights 96.9% of AI tokens and falsely highlights 18.4% of human tokens on these fully labeled articles. The 74 human-document false alarms therefore include broad false highlights, not just one stray token in each article.
Among falsely flagged human articles, the median highlighted share is 31.7% of tokens; 20 of 74 have more than half their tokens highlighted.

## AI detection by generator

| Generator | Our Qwen 20k | EditLens RoBERTa | EditLens Llama |
| --- | ---: | ---: | ---: |
| claude | 30/30 | 30/30 | 30/30 |
| gpt-4o | 30/30 | 30/30 | 30/30 |
| humanized_o1-pro | 30/30 | 30/30 | 30/30 |
| o1-pro | 30/30 | 30/30 | 30/30 |
| paraphrased_gpt-4o | 30/30 | 26/30 | 30/30 |

## Human article false alarms by publication

These are document-level false alarms for our Qwen model at its frozen threshold. They are spread across sources rather than coming from one duplicate or publication.

| Publication | Human articles | Falsely flagged |
| --- | ---: | ---: |
| Associated Press | 15 | 4 |
| Discover | 20 | 8 |
| National Geographic | 25 | 17 |
| New York Times | 20 | 7 |
| Reader's Digest | 3 | 3 |
| Readers Digest | 12 | 7 |
| Scientific American | 15 | 5 |
| Smithsonian Magazine | 25 | 15 |
| Wall Street Journal | 15 | 8 |

This is a stress test of source transfer, not a representative estimate of deployment accuracy. It contains one domain (nonfiction news articles) and five AI generation/rewrite modes.
