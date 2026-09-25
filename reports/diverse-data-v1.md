# Diverse data pilot v1

This replaces the three-source mixed pilot for the next Qwen run. Raw text and
Parquet files live under `/mnt/f/pangram-at-home/data/`, outside Git.

## What is in the pyramid

| Split | Tiny | Small | Medium | Full |
| --- | ---: | ---: | ---: | ---: |
| Train | 200 | 1,000 | 4,000 | 10,000 |
| Validation | 200 | 500 | — | 800 |
| Test | 200 | 600 | — | 1,000 |

Every tier is balanced 50/50 human and AI and nested within the larger tier.
The full train split has 5,000 human and 5,000 AI passages. Category shares are
35% paper/scientific writing, 20% reference/education, 20% creative writing,
10% social/Q&A, 10% consumer reviews, and 5% news. Within each category the
builder alternates contributing sources. Paired passages are limited to a 2:1
word-count ratio where source pairing is available; MAGE rows are matched by
word-count bucket. Text, source ID and group ID are disjoint across train,
validation and test. This does not prove that MAGE prompts are independent:
its public CSV has no common prompt ID for all generations.

Full training sources, counted as human/AI pairs:

| Category | Source | Pairs |
| --- | --- | ---: |
| Paper | PMC OA published abstracts | 309 |
| Paper | ACL Anthology published abstracts | 462 |
| Paper | MAGE SciGen scientific writing | 979 |
| Reference/education | EditLens FineWeb Edu | 334 |
| Reference/education | MAGE ELI5 | 333 |
| Reference/education | MAGE SQuAD | 333 |
| Creative | EditLens Reddit WritingPrompts | 334 |
| Creative | MAGE WritingPrompts | 333 |
| Creative | MAGE ROCStories | 333 |
| Social/Q&A | MAGE ChangeMyView | 250 |
| Social/Q&A | MAGE TLDR | 250 |
| Reviews | EditLens Amazon reviews | 167 |
| Reviews | EditLens Google reviews | 167 |
| Reviews | MAGE Yelp reviews | 166 |
| News | EditLens news | 125 |
| News | MAGE XSum | 125 |

MAGE's published train CSV supplies ten domain labels and many generators.
Its human rows are benchmark-labeled human, but we do **not** have individual
authorship/date/rights records for each row. EditLens is CC BY-NC-SA research
data. This is a local research training run and neither the mixed raw dataset
nor a commercially cleared model is ready for public release. The strictly
dated human corpus remains separate: ten public-domain Standard Ebooks works
at Git commits from no later than 2022-12-31 yield 1,860 passages for an
independent human false-positive audit. The original PMC/ACL rows retain their
per-item/publisher rights metadata.

The original target in `notes/datasets.md` also includes essays, general web,
and professional/finance. This training pilot has no defensible *paired* source
for those categories yet. Independent human false-positive sets now cover
15,594 PERSUADE student essays (fixed 1,000-row audit), 1,622 Federal Reserve
Beige Book passages, and 5,000 Writers Stack Exchange posts (fixed 1,000-row
audit). The latter gives us a second Q&A platform beyond Reddit, but its
authorship labels and CC BY-SA rights need further review before training.
AO3 is not used:
its archive allows AI-written works, so an AO3 page alone cannot certify human
authorship. Amazon/Google/Yelp review labels are benchmark labels, not
verified-purchase proof.

## Evaluation roles

The diverse validation/test splits combine separate upstream paper and
EditLens splits with non-overlapping halves of MAGE's published test CSV. Since
MAGE train is in the new training mix, these are *in-family* evaluations, even
though their text is disjoint. Report each source and domain, not just the
aggregate. MAGE GPT-4 and paraphrase subsets are challenge tests from the same
benchmark family. RAID is frozen separately as an untouched source-family
evaluation: 1,600 balanced passages across abstracts, books, news, poetry,
recipes, Reddit, reviews, and Wikipedia, with 11 generators. No RAID text is
in training or validation. Standard Ebooks and the three new human corpora
probe false positives only. Enron from EditLens is a paired professional-email
test.

The first Qwen adapter was trained before MAGE was added to any training set.
On a frozen 4,000-row MAGE ten-domain test it scores **0.6333 AUROC**, **16.3%
AI recall**, and **1.1% human FPR** at the old fixed threshold. The MAGE GPT-4
OOD subset scores 0.9948 AUROC; the paraphrase subset 0.9440. This wide spread
is why the earlier near-perfect pilot ROC should not be treated as a general
result. Full first-adapter numbers are in
`reports/metrics/segment_qwen3_mage_external_v1.json`.

## Reproduce

```bash
python scripts/build_mage_external_eval.py
python scripts/fetch_standard_ebooks.py
python scripts/build_diverse_pyramid.py
python scripts/run_baselines.py --dataset diverse --train-tier full --model char
python scripts/run_baselines.py --dataset diverse --train-tier full --model word
python scripts/run_baselines.py --dataset diverse --train-tier full --model embedding
```

The Qwen run uses Qwen3-1.7B QLoRA, 512 tokens, rank 16, dropout 0.1, batch 2,
gradient accumulation 8, learning rate 5e-5 with cosine decay and 5% warmup,
step-200 validation/checkpointing, four epochs, and a 9.5-hour deadline. It
logs loss and validation metrics to W&B. Run name:
`qwen3_17b_diverse_v1`; [live W&B run](https://wandb.ai/eac-adsf/pangram-at-home/runs/ogc1ve1i).

Sources: [MAGE](https://github.com/yafuly/MAGE),
[RAID](https://github.com/liamdugan/raid),
[Standard Ebooks public-domain statement](https://standardebooks.org/about/standard-ebooks-and-the-public-domain),
[AO3 data scraping statement](https://www.transformativeworks.org/ai-and-data-scraping-on-the-archive/).
