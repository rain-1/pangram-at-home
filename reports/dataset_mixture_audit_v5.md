# Training mixture audit: Qwen3-1.7B Repeat2 token model v5

This audit reads the exact 20,000-document training JSONL and applies the trainer’s Qwen tokenizer, offset labels, 512-source-token windows, and 256-token stride. The trainer samples shuffled **windows** uniformly; Repeat2 first-copy labels are masked, so supervised-token counts refer to the second copy only. Raw training text is not included here.

## Two training stages

1. **Initialization adapter:** a binary sequence classifier trained from the diverse pyramid on 10,000 documents (5,000 human and 5,000 AI), with a 512-token truncation. Its configuration sets 8 epochs but `max_steps=3,200` overrides that; 25,600 example draws equal about 2.56 passes over this corpus. Only its LoRA backbone weights were transferred to the token model; the token head was newly initialized.
2. **Token training:** 20,000 documents became 26,145 windows. 3,269 optimizer steps × effective batch 8 = 26,152 window draws, approximately one pass over the 26,145 windows. The model processed 18,505,540 Repeat2 tokens and supervised 9,224,733 second-copy token positions.

These stage proportions cannot be combined into one percentage: the stages use different objectives and the first stage truncates each document to 512 tokens.

**Underlying-source reuse:** all 4,964 earlier-v4 token-stage composites reference only source groups already present in initialization training (8,102/8,102 distinct groups). Only 1 composite is an exact full-text match; the others are new joins or edits of previously seen source material. Thus these 4,964 rows add span supervision and new combinations, but little new underlying source diversity.

## Token-stage source concentration

| Source family | Documents | Share | Windows | Share | Supervised token positions | Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLMTrace | 15,036 | 75.2% | 18,582 | 71.1% | 6,326,486 | 68.6% |
| Earlier v4 composites | 4,964 | 24.8% | 7,563 | 28.9% | 2,898,247 | 31.4% |

The token labels are 5,033,985 human (54.6%) and 4,190,748 AI (45.4%). This class balance is reasonable; **source balance is the main problem**. The 4,964 earlier-v4 rows are themselves composites of 16 named sources, so each contributes only a small slice of token training.

There are only 48 pure-human documents from the two non-LLMTrace news sources (`editlens:news` and `mage:xsum`). LLMTrace contributes 3,686 pure-human documents overall, but its shared construction pipeline dominates their examples.

## Detailed source mix in the token stage

| Source | Documents | Windows | Window share | Supervised AI share within source |
| --- | ---: | ---: | ---: | ---: |
| LLMTrace_detection | 15,036 | 18,582 | 71.07% | 43.7% |
| mage:sci | 602 | 1,102 | 4.21% | 51.8% |
| editlens:fineweb_edu | 430 | 723 | 2.77% | 48.7% |
| editlens:reddit_writing_prompts | 426 | 721 | 2.76% | 53.9% |
| paper:acl_anthology | 376 | 654 | 2.50% | 45.3% |
| mage:wp | 284 | 595 | 2.28% | 46.0% |
| mage:cmv | 368 | 490 | 1.87% | 49.3% |
| paper:pmc_oa | 268 | 483 | 1.85% | 39.3% |
| editlens:amazon_reviews | 264 | 438 | 1.68% | 53.8% |
| editlens:google_reviews | 257 | 427 | 1.63% | 55.6% |
| mage:tldr | 375 | 404 | 1.55% | 49.3% |
| mage:squad | 281 | 333 | 1.27% | 44.9% |
| mage:eli5 | 283 | 292 | 1.12% | 49.0% |
| mage:roct | 286 | 287 | 1.10% | 51.3% |
| editlens:news | 143 | 276 | 1.06% | 49.8% |
| mage:yelp | 218 | 218 | 0.83% | 50.5% |
| mage:xsum | 103 | 120 | 0.46% | 44.5% |

## Domains and construction patterns in the token stage

The domain labels are broad. For example, most `news` and all `article` rows here are LLMTrace records; the `news` label alone does not establish publisher-style coverage.

| Domain | Documents | Windows | Window share |
| --- | ---: | ---: | ---: |
| story | 2,099 | 3,001 | 11.5% |
| article | 2,000 | 2,888 | 11.0% |
| news | 2,039 | 2,795 | 10.7% |
| factual | 1,960 | 2,445 | 9.4% |
| paper | 1,246 | 2,239 | 8.6% |
| question | 1,869 | 2,192 | 8.4% |
| paper_abstract | 1,880 | 1,998 | 7.6% |
| poetry | 1,541 | 1,645 | 6.3% |
| creative | 996 | 1,603 | 6.1% |
| reference_education | 994 | 1,348 | 5.2% |
| review | 1,193 | 1,230 | 4.7% |
| reviews | 739 | 1,083 | 4.1% |
| social_qa | 743 | 894 | 3.4% |
| short_form | 701 | 784 | 3.0% |

| Construction | Documents | Windows | Window share |
| --- | ---: | ---: | ---: |
| fill_gaps | 5,954 | 6,882 | 26.3% |
| source_human | 3,686 | 4,968 | 19.0% |
| same_source_long_mix | 1,250 | 2,677 | 10.2% |
| expand | 2,034 | 2,616 | 10.0% |
| create | 1,705 | 2,378 | 9.1% |
| same_label_long_join | 1,000 | 2,138 | 8.2% |
| delete | 1,362 | 1,393 | 5.3% |
| matched_pair_short_mix | 1,080 | 1,081 | 4.1% |
| unaltered_source | 964 | 995 | 3.8% |
| same_label_short_join | 500 | 502 | 1.9% |
| human_modification | 295 | 345 | 1.3% |
| same_source_short_mix | 170 | 170 | 0.7% |

## Initialization-stage mix

| Domain | Documents | Share |
| --- | ---: | ---: |
| paper | 3,500 | 35.0% |
| creative | 2,000 | 20.0% |
| reference_education | 2,000 | 20.0% |
| reviews | 1,000 | 10.0% |
| social_qa | 1,000 | 10.0% |
| news | 500 | 5.0% |

The initialization stage contained no LLMTrace, but it was paper-heavy (35%) and news-light (5%). The token stage then shifted sharply toward LLMTrace. Its category labels do not represent independent source diversity.

## What this explains, and what it does not

The source concentration makes the excellent LLMTrace holdout scores a narrow result. It plausibly encouraged reliance on patterns of that dataset and left the calibration set without published nonfiction. This is an evidence-backed hypothesis, not proof of a particular learned shortcut. The external article stress test found 74/150 human-document false alarms, spread across publications; the long-document length effect alone cannot explain its 18.4% human-token FPR.

For the next training recipe, set **per-source and per-domain window quotas** before training, report both document and window/token shares, and keep independent human nonfiction articles in training and calibration. Preserve the 300 external stress articles as a holdout. Evaluate by source after every run so a high score on one construction pipeline cannot dominate the headline.

The exact per-group counts, including source × kind and source-family × domain, are in [dataset_mixture_audit_v5.json](dataset_mixture_audit_v5.json).
