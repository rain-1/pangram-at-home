# First data build and baselines — 2026-09-24

## What is frozen

The raw corpora, split Parquet files, generated text, and model weights are under `/mnt/f/pangram-at-home` (about 14 GB for this run). Small [manifests](../manifests) and [metric JSON files](metrics) in Git pin source revisions, input hashes, exact split counts, Parquet hashes, and results. `scripts/verify_splits.py` passed on all four nested pyramids and the swapped-generator test. It checked balanced labels, nesting, work and exact-text separation, and paper test venue separation. A separate word 3–5-gram similarity check found no near-duplicate human abstracts across paper splits above 0.26 cosine similarity.

| Data | Train tiers, rows | Validation tiers, rows | Test tiers, rows |
| --- | --- | --- | --- |
| General EditLens research | 1,000 / 5,000 / 20,000 / 37,568 | 200 / 800 / 1,504 | 200 / 1,000 / 4,000, plus an Enron test up to 3,600 |
| Licensed PMC + ACL paper abstracts | 100 / 400 / 1,000 / 1,542 | 100 / 200 / 436 | 100 / 200 / 442 |
| Mixed: 35% paper per label | 200 / 1,000 / 3,000 / 4,404 | 200 / 1,000 / 1,244 | 200 / 1,000 / 1,262 |

Every binary tier is 50% human and 50% AI, and each smaller tier is contained in the larger one. The paper pool contains 1,210 source-matched pairs: 483 PMC articles and 727 ACL papers. Its 221-pair test holds out journals or venues from paper training. We also reserved 261 human body chunks from 87 PMC test papers for a full-paper false-positive audit.

The human paper sources are 963 dated, full-text [PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) articles with per-item CC BY 4.0 in JATS XML, and 20,214 dated [ACL Anthology](https://aclanthology.org/faq/copyright/) abstracts from official 2016–2022 metadata. The paper AI texts were made locally from titles with [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct). Pair filters enforce source hash, date, rights, approximate length matching, and a journal cap. The mixed pool uses [EditLens ICLR](https://huggingface.co/datasets/pangram/editlens_iclr) for general text; its CC BY-NC-SA terms make the mixed split and anything trained on it **noncommercial research assets**.

## Test results

All models use a threshold set **only on the matching full validation split** to allow at most 2% observed false positives there. AI recall is the fraction of AI texts flagged; false positives are human texts flagged as AI.

The classical baselines were fitted on the listed local train tiers (full paper tier or 3,000-row mixed medium tier). Pangram's published checkpoints already had their own much larger training process, so this is a reference comparison, not a matched-training-data contest.
Our EditLens inference normalizes text and scores the first 510 tokenizer tokens; the n-gram models see the full input. This differs slightly from the authors' full preprocessing and should be treated as a local reference run.

| Baseline | Paper AI recall | Paper human false positives | Mixed AI recall | Mixed human false positives |
| --- | ---: | ---: | ---: | ---: |
| Character TF-IDF 3–5 grams + logistic regression | 219/221 (99.1%) | 7/221 (3.2%) | 601/631 (95.2%) | 13/631 (2.1%) |
| Word TF-IDF 1–2 grams + logistic regression | 219/221 (99.1%) | 5/221 (2.3%) | 573/631 (90.8%) | 12/631 (1.9%) |
| MiniLM embeddings + logistic regression | 129/221 (58.4%) | 6/221 (2.7%) | 191/631 (30.3%) | 9/631 (1.4%) |
| EditLens RoBERTa-large | 203/221 (91.9%) | 2/221 (0.9%) | 617/631 (97.8%) | 8/631 (1.3%) |
| EditLens Llama-3.2-3B | 218/221 (98.6%) | 3/221 (1.4%) | 629/631 (99.7%) | 16/631 (2.5%) |

The mixed character model's 2.1% overall false-positive rate hides a paper failure: it flagged **8 of 87 PMC** humans but 1 of 134 ACL humans. The word model flagged 6 of 87 PMC humans. RoBERTa flagged 1/87 PMC and 5/134 ACL humans on the mixed test, while its paper AI recall was 78/87 PMC and 130/134 ACL. Llama flagged 4/87 PMC and 9/134 ACL humans, even though it caught 85/87 PMC and all 134 ACL AI texts. Paper-specific evaluation should always show each source, not just a single aggregate.

At 8 false positives among 631 mixed-test humans, RoBERTa's observed rate is 1.27%; the exact 95% binomial interval is about **0.55%–2.48%**. This run therefore does not establish a population false-positive rate below 2%.

## Shift and human-only checks

The 182-pair swapped-generator paper test uses the **same held-out human works** with a new AI generator for each source: SmolLM on PMC and Qwen on ACL. It is a generator-shift check, not an independent human test.

| Baseline | Swapped AI recall | Human false positives on matched subset |
| --- | ---: | ---: |
| Character n-gram | 177/182 (97.3%) | 5/182 (2.7%) |
| Word n-gram | 182/182 (100%) | 5/182 (2.7%) |
| Embedding | 111/182 (61.0%) | 4/182 (2.2%) |
| EditLens RoBERTa | 164/182 (90.1%) | 2/182 (1.1%) |
| EditLens Llama | 182/182 (100%) | 2/182 (1.1%) |

Calibrating on general EditLens validation alone did not transfer to scholarly prose: the general character model falsely flagged **11,854/18,287 ACL human abstracts (64.8%)** and **91/261 PMC human body chunks (34.9%)**. After training with papers and calibrating on mixed validation, the mixed character model flagged **317/17,560 unseen ACL abstracts (1.81%)**; the mixed word model flagged 139/17,560 (0.79%). These ACL audits exclude every paper used in our paper train, validation, or test splits. The separate PMC body audit is small: a PMC-trained character model flagged 7/261 chunks (2.68%), and RoBERTa calibrated on PMC validation flagged 2/261 (0.77%).

## Limits of this run

- The training-ready paper set uses **abstracts**. We downloaded PMC full article bodies, but used only held-out body chunks for a human-only audit; full-paper balanced training still needs credible AI section counterparts and larger evaluation.
- Human authorship is supported by official, pre-2023 publication records. The date alone cannot prove every sentence was human-written, and current metadata may contain later corrections.
- The paper AI set has only two small local generators and title-only prompts. The swapped-generator test reduces one confound, but performance against frontier models, AI edits, and subtle mixed authorship is unmeasured.
- The entire upstream PMC/PLOS/Gutenberg collections were **not** downloaded. PMC alone contains millions of articles, and rights differ by item. This run is a bounded, reproducible, licensed first corpus; the source shortlist in [datasets.md](../notes/datasets.md) records expansion candidates.
- The EditLens benchmark and reference models are gated and noncommercial. We keep them out of the reusable PMC/ACL paper corpus and Git.

The next model experiment can start from the mixed tiny/small/medium tiers, but the main evaluation target should be the paper test and separate PMC/ACL human false-positive counts. Hold the frozen test manifest fixed while selecting thresholds and model variants on validation.
