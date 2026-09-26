# Model evaluation dashboard: Qwen 20k versus Open Pangram

**Read the source rows before the averages.** The LLMTrace test is disjoint from training by text and group ID, but 15,036 of our 20,000 training documents came from other LLMTrace records. Synthetic v4 is a development validation set. External articles, AITDNA, CoAuthor, and the locked human test probe different sources.

Document decision: our Qwen model is positive if any token crosses its frozen threshold; EditLens uses its mean native-window score on the main pure-document sets. For AITDNA human controls, all models use the span-window decoder and its document-any threshold. EditLens has no native token head, so mixed-document token scores broadcast each EditLens window prediction across its source tokens. Every threshold was calibrated on the same separate 1,120 pure-human documents to allow at most 5% document-any false alarms. This calibration set has social Q&A, professional finance, and creative writing, with no published nonfiction articles.

## Source-balanced overview

These are unweighted averages of the named source rows below, so the large locked human set cannot conceal a failure on external articles. They describe this test collection, not deployment prevalence.

| Model | Mean pure-human accuracy across 5 sets | Mean pure-AI accuracy across 3 sets | Mean mixed-token balanced accuracy across 4 sets |
| --- | ---: | ---: | ---: |
| Our Qwen 20k | 89.9% | 92.2% | 70.4% |
| EditLens RoBERTa | 98.9% | 71.5% | 58.6% |
| EditLens Llama | 98.1% | 81.6% | 59.0% |

## 1. Fully human and fully AI documents

Human-only accuracy is **specificity** (`1 − FPR`); AI-only accuracy is **recall** (`1 − FNR`). “FPR on AI-only text” is undefined because false positives require human examples. Combined accuracy is shown only where both classes exist.

| Dataset | Model | Human correct / total | Human FPR | AI correct / total | AI FNR | Combined accuracy | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLMTrace test | Our Qwen 20k | 720/720 | 0.0% | 500/516 | 3.1% | 98.7% | 0.9949 |
| LLMTrace test | EditLens RoBERTa | 710/720 | 1.4% | 423/516 | 18.0% | 91.7% | 0.9506 |
| LLMTrace test | EditLens Llama | 701/720 | 2.6% | 486/516 | 5.8% | 96.0% | 0.9834 |
| Synthetic v4 validation | Our Qwen 20k | 152/152 | 0.0% | 118/148 | 20.3% | 90.0% | 0.9917 |
| Synthetic v4 validation | EditLens RoBERTa | 150/152 | 1.3% | 52/148 | 64.9% | 67.3% | 0.7398 |
| Synthetic v4 validation | EditLens Llama | 152/152 | 0.0% | 75/148 | 49.3% | 75.7% | 0.8523 |
| External articles | Our Qwen 20k | 76/150 | 49.3% | 150/150 | 0.0% | 75.3% | 0.9370 |
| External articles | EditLens RoBERTa | 149/150 | 0.7% | 146/150 | 2.7% | 98.3% | 0.9989 |
| External articles | EditLens Llama | 143/150 | 4.7% | 150/150 | 0.0% | 97.7% | 0.9999 |
| Locked human test | Our Qwen 20k | 3579/3579 | 0.0% | — | — | — | — |
| Locked human test | EditLens RoBERTa | 3568/3579 | 0.3% | — | — | — | — |
| Locked human test | EditLens Llama | 3568/3579 | 0.3% | — | — | — | — |
| AITDNA human controls | Our Qwen 20k | 102/103 | 1.0% | — | — | — | — |
| AITDNA human controls | EditLens RoBERTa | 101/103 | 1.9% | — | — | — | — |
| AITDNA human controls | EditLens Llama | 101/103 | 1.9% | — | — | — | — |

## 2. Mixed documents: token and span localization

These rows contain both human and AI text. “Mixed documents found” requires at least one correctly highlighted AI token. Human FPR is the fraction of human **tokens within mixed documents** falsely highlighted. Token balanced accuracy averages AI-token recall and human-token specificity; it prevents the large human portions from hiding missed AI spans.

| Dataset | Model | Mixed documents found | Token balanced accuracy | AI-token recall | Human-token FPR | Token F1 | Token AUROC | AI spans ≥50% covered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLMTrace test | Our Qwen 20k | 412/764 | 62.5% | 24.9% | 0.02% | 0.399 | 0.962 | 14.8% (2261 spans) |
| LLMTrace test | EditLens RoBERTa | 102/764 | 56.3% | 18.1% | 5.52% | 0.284 | 0.642 | 9.8% (2261 spans) |
| LLMTrace test | EditLens Llama | 196/764 | 60.8% | 37.0% | 15.35% | 0.453 | 0.656 | 24.3% (2261 spans) |
| Synthetic v4 validation | Our Qwen 20k | 249/300 | 83.5% | 67.2% | 0.15% | 0.803 | 0.986 | 61.6% (323 spans) |
| Synthetic v4 validation | EditLens RoBERTa | 46/300 | 56.0% | 15.7% | 3.73% | 0.263 | 0.637 | 13.6% (323 spans) |
| Synthetic v4 validation | EditLens Llama | 102/300 | 60.4% | 37.2% | 16.36% | 0.482 | 0.657 | 30.3% (323 spans) |
| AITDNA collaboration | Our Qwen 20k | 237/258 | 85.7% | 86.6% | 15.23% | 0.900 | 0.920 | 50.8% (2015 spans) |
| AITDNA collaboration | EditLens RoBERTa | 188/258 | 72.0% | 78.6% | 34.54% | 0.819 | 0.772 | 54.1% (2015 spans) |
| AITDNA collaboration | EditLens Llama | 229/258 | 64.7% | 95.5% | 66.20% | 0.865 | 0.753 | 82.7% (2015 spans) |
| CoAuthor collaboration | Our Qwen 20k | 0/119 | 50.0% | 0.0% | 0.00% | 0.000 | 0.759 | 0.0% (375 spans) |
| CoAuthor collaboration | EditLens RoBERTa | 0/119 | 50.0% | 0.0% | 0.00% | 0.000 | 0.476 | 0.0% (375 spans) |
| CoAuthor collaboration | EditLens Llama | 0/119 | 50.0% | 0.0% | 0.00% | 0.000 | 0.512 | 0.0% (375 spans) |

A model that predicts every token as human can have high raw token accuracy on human-heavy documents. CoAuthor exposes that failure: all three models have 0% AI-token recall at the frozen thresholds. EditLens window spans are deliberately coarse, so compare their localization with that limitation in mind.

## 3. Category breakdown: LLMTrace pure documents

| Domain | Model | Human FPR | AI recall |
| --- | --- | ---: | ---: |
| article (166) | Our Qwen 20k | 0.00% | 98.4% |
| article (166) | EditLens RoBERTa | 1.94% | 81.0% |
| article (166) | EditLens Llama | 3.88% | 96.8% |
| factual (165) | Our Qwen 20k | 0.00% | 92.1% |
| factual (165) | EditLens RoBERTa | 0.00% | 77.8% |
| factual (165) | EditLens Llama | 0.98% | 95.2% |
| news (146) | Our Qwen 20k | 0.00% | 95.2% |
| news (146) | EditLens RoBERTa | 0.96% | 90.5% |
| news (146) | EditLens Llama | 0.00% | 90.5% |
| paper_abstract (153) | Our Qwen 20k | 0.00% | 98.6% |
| paper_abstract (153) | EditLens RoBERTa | 5.00% | 83.6% |
| paper_abstract (153) | EditLens Llama | 2.50% | 93.2% |
| poetry (118) | Our Qwen 20k | 0.00% | 100.0% |
| poetry (118) | EditLens RoBERTa | 0.00% | 83.3% |
| poetry (118) | EditLens Llama | 15.71% | 97.9% |
| question (151) | Our Qwen 20k | 0.00% | 98.3% |
| question (151) | EditLens RoBERTa | 0.00% | 78.3% |
| question (151) | EditLens Llama | 0.00% | 93.3% |
| review (115) | Our Qwen 20k | 0.00% | 91.8% |
| review (115) | EditLens RoBERTa | 0.00% | 77.0% |
| review (115) | EditLens Llama | 0.00% | 90.2% |
| short_form (61) | Our Qwen 20k | 0.00% | 97.5% |
| short_form (61) | EditLens RoBERTa | 0.00% | 85.0% |
| short_form (61) | EditLens Llama | 0.00% | 97.5% |
| story (161) | Our Qwen 20k | 0.00% | 100.0% |
| story (161) | EditLens RoBERTa | 3.16% | 84.8% |
| story (161) | EditLens Llama | 1.05% | 93.9% |

## 4. Category breakdown: LLMTrace mixed documents

| Domain | Model | AI-token recall | Human-token FPR | Token balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| article (104) | Our Qwen 20k | 22.0% | 0.00% | 61.0% |
| article (104) | EditLens RoBERTa | 7.0% | 2.75% | 52.1% |
| article (104) | EditLens Llama | 33.7% | 15.59% | 59.1% |
| factual (102) | Our Qwen 20k | 32.9% | 0.00% | 66.5% |
| factual (102) | EditLens RoBERTa | 25.1% | 6.30% | 59.4% |
| factual (102) | EditLens Llama | 45.2% | 25.98% | 59.6% |
| news (107) | Our Qwen 20k | 31.1% | 0.00% | 65.6% |
| news (107) | EditLens RoBERTa | 23.9% | 9.69% | 57.1% |
| news (107) | EditLens Llama | 23.5% | 9.64% | 56.9% |
| paper_abstract (92) | Our Qwen 20k | 36.8% | 0.01% | 68.4% |
| paper_abstract (92) | EditLens RoBERTa | 31.8% | 7.61% | 62.1% |
| paper_abstract (92) | EditLens Llama | 61.0% | 24.08% | 68.5% |
| poetry (76) | Our Qwen 20k | 36.9% | 0.15% | 68.4% |
| poetry (76) | EditLens RoBERTa | 44.7% | 10.56% | 67.1% |
| poetry (76) | EditLens Llama | 59.5% | 17.08% | 71.2% |
| question (99) | Our Qwen 20k | 11.4% | 0.01% | 55.7% |
| question (99) | EditLens RoBERTa | 0.3% | 0.26% | 50.0% |
| question (99) | EditLens Llama | 16.6% | 3.57% | 56.5% |
| review (53) | Our Qwen 20k | 17.9% | 0.00% | 58.9% |
| review (53) | EditLens RoBERTa | 3.8% | 1.92% | 51.0% |
| review (53) | EditLens Llama | 6.9% | 2.60% | 52.2% |
| short_form (25) | Our Qwen 20k | 29.1% | 0.00% | 64.6% |
| short_form (25) | EditLens RoBERTa | 43.9% | 8.79% | 67.5% |
| short_form (25) | EditLens Llama | 76.9% | 17.88% | 79.5% |
| story (106) | Our Qwen 20k | 15.0% | 0.03% | 57.5% |
| story (106) | EditLens RoBERTa | 12.1% | 4.87% | 53.6% |
| story (106) | EditLens Llama | 39.8% | 18.81% | 60.5% |

## 5. Real collaboration categories

| Dataset and category | Model | AI-token recall | Human-token FPR | Token balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| AITDNA collaboration: Argumentative Writing (88) | Our Qwen 20k | 85.9% | 20.41% | 82.7% |
| AITDNA collaboration: Argumentative Writing (88) | EditLens RoBERTa | 79.2% | 41.79% | 68.7% |
| AITDNA collaboration: Argumentative Writing (88) | EditLens Llama | 96.0% | 79.32% | 58.3% |
| AITDNA collaboration: Creative Writing (84) | Our Qwen 20k | 89.0% | 10.40% | 89.3% |
| AITDNA collaboration: Creative Writing (84) | EditLens RoBERTa | 78.3% | 30.28% | 74.0% |
| AITDNA collaboration: Creative Writing (84) | EditLens Llama | 94.0% | 58.08% | 68.0% |
| AITDNA collaboration: Explanatory Writing (81) | Our Qwen 20k | 85.3% | 16.40% | 84.4% |
| AITDNA collaboration: Explanatory Writing (81) | EditLens RoBERTa | 79.1% | 38.41% | 70.3% |
| AITDNA collaboration: Explanatory Writing (81) | EditLens Llama | 97.7% | 74.41% | 61.6% |
| AITDNA collaboration: Peer Review (5) | Our Qwen 20k | 28.2% | 11.33% | 58.4% |
| AITDNA collaboration: Peer Review (5) | EditLens RoBERTa | 0.0% | 0.00% | 50.0% |
| AITDNA collaboration: Peer Review (5) | EditLens Llama | 5.7% | 0.57% | 52.6% |
| CoAuthor collaboration: argumentative (36) | Our Qwen 20k | 0.0% | 0.00% | 50.0% |
| CoAuthor collaboration: argumentative (36) | EditLens RoBERTa | 0.0% | 0.00% | 50.0% |
| CoAuthor collaboration: argumentative (36) | EditLens Llama | 0.0% | 0.00% | 50.0% |
| CoAuthor collaboration: creative (83) | Our Qwen 20k | 0.0% | 0.00% | 50.0% |
| CoAuthor collaboration: creative (83) | EditLens RoBERTa | 0.0% | 0.00% | 50.0% |
| CoAuthor collaboration: creative (83) | EditLens Llama | 0.0% | 0.00% | 50.0% |

## 6. External article categories

The [separate external article report](external_articles_v5.md) includes generator and publication charts. The same breakdown is tabulated here.

| AI generator | Our Qwen 20k | EditLens RoBERTa | EditLens Llama |
| --- | ---: | ---: | ---: |
| claude | 30/30 | 30/30 | 30/30 |
| gpt-4o | 30/30 | 30/30 | 30/30 |
| humanized_o1-pro | 30/30 | 30/30 | 30/30 |
| o1-pro | 30/30 | 30/30 | 30/30 |
| paraphrased_gpt-4o | 30/30 | 26/30 | 30/30 |

| Human publication | Our Qwen FPR | EditLens RoBERTa FPR | EditLens Llama FPR |
| --- | ---: | ---: | ---: |
| Associated Press | 4/15 | 0/15 | 0/15 |
| Discover | 8/20 | 0/20 | 1/20 |
| National Geographic | 17/25 | 1/25 | 4/25 |
| New York Times | 7/20 | 0/20 | 0/20 |
| Reader's Digest | 3/3 | 0/3 | 0/3 |
| Readers Digest | 7/12 | 0/12 | 1/12 |
| Scientific American | 5/15 | 0/15 | 0/15 |
| Smithsonian Magazine | 15/25 | 0/25 | 0/25 |
| Wall Street Journal | 8/15 | 0/15 | 1/15 |

Our Qwen model flags 74/150 attributed-human articles, versus 1/150 for EditLens RoBERTa and 7/150 for EditLens Llama. Its human-token FPR is 18.4%; median highlighted share among falsely flagged human articles is 31.7%.

## What failed

Our 20k mixture contains 15,036 LLMTrace documents (75.2%). It has only 20 pure-human `editlens:news` documents and 28 pure-human `mage:xsum` documents; 548 LLMTrace human documents carry the `news` domain label. The 1,120-document human calibration set has no published nonfiction articles. External article human FPR remains about 50% across 500–1,000-word bands, while the similarly long calibration documents are near the intended 5% rate. Length and the “any token” document rule contribute, but cannot explain the 18.4% human-token FPR or near-whole-article false highlights. The evidence supports a source shift and inadequate human coverage/calibration; the specific learned shortcut is not yet identified.

Next evaluation gate: use an external, source-balanced calibration set assembled without these 300 stress articles; add separate human nonfiction to training, preserve this article set as a holdout, and report these source rows and mixed-span metrics for every future checkpoint. Also retain CoAuthor as a short-insertion challenge.

Sources: [Human Detectors](https://github.com/jenna-russell/human_detectors), [EditLens RoBERTa](https://huggingface.co/pangram/editlens_roberta-large), [EditLens Llama](https://huggingface.co/pangram/editlens_Llama-3.2-3B).
