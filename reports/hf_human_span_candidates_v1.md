# Hugging Face human-source candidates for span attribution

Reviewed 2026-09-27 for the basic human/AI span detector. The data are candidates, not training-ready labels. The current model's remaining article errors cluster in magazine science and explainer prose; generic historical CNN/PMC articles were already easy. Existing human sources include PERSUADE student essays, Writers Stack Exchange, Standard Ebooks classics, common-pile news and v9 science articles. These candidates are source-distinct from those known sources.

## Acquired candidates

| HF dataset (pinned revision) | Useful coverage / provenance | License and material risks | Acquired sample and recommendation |
| --- | --- | --- | --- |
| [`ronaldahmed/scitechnews`](https://huggingface.co/datasets/ronaldahmed/scitechnews/tree/955dfa3b4c7c5edf29ad019a7b91734a739ef3df) | Best fit for explanatory science/technology prose. Its card says all text was human-produced and says the ACM TechNews archive material was collected from 1999–2021. The train file has 26,638 rows; 13,903 have an empty `pr-article`, and 10,049 have a nonempty body of at least 300 whitespace words. The row has no date, byline, or canonical source URL, and not all material is magazine-style. | The HF card has no license. The dataset loader defines CC BY-SA 3.0 but comments out the actual `license=` declaration. Underlying article-level reuse rights therefore remain unresolved. Treat all rows as human candidates, not verified pre-2023 human positives, until source rights and provenance are checked. | Raw train JSONL: `/mnt/f/pangram-at-home/data/candidate_span_sources/human/scitechnews/train.json` (SHA-256 `7b8c72f587fa33e7f8a83915dc446d44976c46431a0fbf42b1fc4527a6930fe8`). Normalized sample: 400 train, 100 calibration, 100 test. **Highest priority for an article-style FPR diagnostic**, conditional on an item-level rights review. |
| [`virtualkevin/tell-me-a-story`](https://huggingface.co/datasets/virtualkevin/tell-me-a-story/tree/be75c1948d97f4d0d4ec8fa5f0dfc59398654d22) | 230 long prompt/target pairs. Dataset card calls the targets human-written fiction; upstream DeepMind README describes the human-written story dataset. Targets are 3,180–15,899 characters (median 8,076). | CC BY 4.0, with attribution. The associated paper first appeared in 2024; the dataset gives no story-level creation dates or author IDs, so it cannot substantiate a pre-LLM human label. Prompts are excluded from normalized text. | Normalized train/validation/test: 123/52/55 records. **Good small, distinct fiction stress set**; use as human-asserted recent material, not as the strongest clean-human class. |
| [`goosmanlei/amazon_reviews_multi`](https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi/tree/a7b1fa9703f1f930d5c8e1a65c5d2774aaab0c85) | English Amazon customer reviews collected 2015–2019; raw records include pseudonymous reviewer and product IDs. One review per reviewer was selected; reviewer groups do not cross the output partitions. | The mirror defers to original Amazon Reviews Multi terms: academic research only, no commercial use, and no redistribution. A top-level license on a derived HF mirror would not override those terms. Customer-submitted text is not independently verified as human-authored; paid, incentivized, copied, or assisted reviews can occur. No individual review dates survive in the selected source. | Normalized sample: 1,000 train, 250 calibration, 250 test, one unique reviewer per row, 80+ body characters. **Secondary consumer-review stress data only**; retain outside Git and avoid treating it as definitive clean-human evidence. |

For the scientific-journalism sample, the normalized `kind` is `human_candidate`; each row carries `source_id`, `group_id`, source split, length, hashes, and a full-text human span label for audit. The Amazon review sample also uses `human_candidate`. Story targets use `kind: human` in the normalized file because the dataset specifically asserts human-written targets, with a confidence note documenting the missing dates and authors.

## Overlap and quality checks

The preparation script samples at fixed seed `20260927` and stores all text outside Git. A sampled normalized 24-word-shingle audit found **zero candidate rows with a match** against local `span_balanced_v6`, `span_publication_hardneg_v8`, or `span_essay_paired_v10` training data, the v2 human calibration/test, the CoAuthor test, or the AI candidate test. This is a sampled fingerprint audit, not proof of zero overlap; do another exact and near-duplicate audit against any new target training parents before intake. Amazon is split by unique `reviewer_id` across its output partitions. Tell Me A Story keeps the upstream split and split-qualified example IDs. SciTechNews partitions by record ID from the upstream training split.

The Tell Me A Story source restarts `example_id` numbering in each upstream
split. Normalized record and group IDs therefore include the split name; all
230 IDs are unique across train, validation, and test. A separate sampled
24-word cross-split check found no text overlap.

The local normalized files, raw HF snapshots, and per-row metadata are under `/mnt/f/pangram-at-home/data/candidate_span_sources/human/normalized/`. Recreate normalized candidates with:

```bash
python scripts/prepare_hf_human_candidates_v1.py
```

The builder writes split JSONL plus manifests and the overlap audit. It does not train a model or upload data. Preserve source IDs and hashes if selecting rows later.

## Investigated and rejected fiction mirror

[`leftyfeep/fiction-chapters-24kmax`](https://huggingface.co/datasets/leftyfeep/fiction-chapters-24kmax/tree/e200391e1eef648a787e761cd135cefd9846bdd2) advertises PDDL and says its 5,625 rows are public-domain fiction from Project Gutenberg/Wikisource. Its source list and preview include John D. MacDonald's *A Bullet for Cinderella*, a mid-20th-century work that is not public domain. The top-level PDDL tag cannot grant rights in copied third-party works. I removed the downloaded text and excluded it from normalized candidates. Do not ingest this mirror without a full work-by-work rights audit.

## Next source to investigate

No suitable The Conversation corpus appeared in the Hugging Face search. Its articles are expert-written, close to magazine explainers, and could help with the remaining false alarms, but its official terms describe CC BY-ND; confirm whether the intended model-training use fits before collecting a batch. Prefer archived pre-2023 article versions with bylines and item-level license evidence.

## Follow-up HF search (2026-09-27)

No stronger training-ready HF source was found. The closest creative candidate was [`ray0rf1re/AO3-2020`](https://huggingface.co/datasets/ray0rf1re/AO3-2020/tree/5fa81ed59f6d75bd2aa8a593d56813316b1301c3): it claims a 2020 cutoff and CC BY-NC 4.0, and a 2.5M-token sample has 711 rows. I downloaded that 7.2 MB sample to inspect its fields, then removed the story text. The sample contains only `storyId`, chapter index, raw HTML, and text; it has no work date, author, or item-level license. The card says individual authors retain copyright, while AO3's [Terms of Service](https://secure.ao3.org/tos) say OTW does not claim copyright ownership. The dataset-wide tag does not establish author permission for these works. Do not ingest this corpus unless each selected work's license or permission is verified. This remains an attractive *discovery lead* for fanfiction because of its claimed cutoff, but it does not clear the rights/provenance bar.

[`ficsim/ficsim`](https://huggingface.co/datasets/ficsim/ficsim) says its 90 fanfiction works were collected with author consent, but its card explicitly bars training, fine-tuning, and input to models; it is not usable for this detector. [`The Conversation`](https://tc.tc-dev.net/us/republishing-guidelines) remains a potentially close science-explainer source, but the publisher's terms state CC BY-ND and no suitable Hugging Face corpus surfaced. I found no alternative contemporary fiction or science-explainer corpus with both row-level pre-2023 evidence and clear reuse permission that improves on the existing acquired candidates.
