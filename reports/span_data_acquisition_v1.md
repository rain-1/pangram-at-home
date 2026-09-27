# Span-attribution data intake, 27 September 2026

This is a **noncommercial, open-research project**. Sources carrying
noncommercial terms are therefore candidates for our experiments, subject to
attribution, source-document rights, and any other dataset-specific terms.
Dataset-card licenses do not automatically cover third-party texts inside a
corpus. No newly acquired candidate has yet been used to train or score a
checkpoint.

## Why these data

The v10 detector still misses many AI spans on locked LLMTrace mixed documents
(38.6% AI-token recall at 0.8% human-token FPR), and it is weak on AITDNA peer
reviews (16.5% AI-token recall). Human publication prose also remains a false
positive problem: the external article evaluation has 16 human false alarms in
150 documents, including 7 of 25 National Geographic articles. The next
experiment needs diverse mixed-span supervision, stronger all-human prose,
and AI text from more generator families, while retaining independent tests.

## Acquired and prepared

All text lives outside Git in
`/mnt/f/pangram-at-home/data/candidate_span_sources/`. The linked source
reports contain pinned revisions, file hashes, normalization details, and
limitations.

| Source | Prepared data | Best role | Important limit |
| --- | ---: | --- | --- |
| [OpAI-Bench reports](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench) | 12,572 deduplicated rows from 1,397 report groups; derived train/validation/test: 9,998/1,314/1,260 | First controlled mixed-span experiment, new government-report domain | Synthetic GPT-5.4 edit trajectories; eight versions of each seed are highly correlated. Source-document reuse rights need review. |
| [OpAI-Bench abstracts](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench) | 11,036 deduplicated rows from 1,235 abstract groups; derived splits: 8,938/1,014/1,084 | Additional mixed boundaries and science writing | Synthetic GPT-5.4-nano edits; academic overlap risk is higher. Four overlapping source groups and one invalid-offset row were excluded before this split. |
| [GEN](https://huggingface.co/datasets/szyszy/GEN) | 35,419 training rows from 3,989 prompt groups after overlap exclusions: 3,989 human and 31,430 AI | Fully AI text from eight named models in five domains, balanced with independent human sources | CC BY-NC 4.0 fits this project, but human-reference source licenses need review. Its 273,418 acquired AI edits have no gold character spans. Raw 8:1 AI ratio must not become the training ratio. |
| [PeerPrism](https://huggingface.co/datasets/Reviewerly/PeerPrism) | 674 human peer reviews and 4,044 AI-text reviews acquired | Peer-review domain positives and negatives | CC BY-NC 4.0 fits this project. “Hybrid” refers to idea origin; it provides no mixed-text boundaries. Preserve paper IDs when splitting. |
| [Tell Me A Story](https://huggingface.co/datasets/virtualkevin/tell-me-a-story) | 123/52/55 human-asserted fiction stories in train/validation/test | Small creative-writing stress set | CC BY 4.0 attribution; no per-story author or date, so not independently verified pre-2023 human text. |
| [SciTechNews](https://huggingface.co/datasets/ronaldahmed/scitechnews) | 400/100/100 sampled candidate articles | Investigate science-explainer false alarms | Missing row-level bylines/dates/URLs and unresolved item rights. Human-asserted only; keep diagnostic/conditional for now. |
| [Amazon Reviews Multi](https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi) | 1,000/250/250 sampled, disjoint reviewer groups | Secondary consumer-review stress set | Research-only, no redistribution; user-submitted text is not proven human. |
| [FAIDSet](https://huggingface.co/datasets/ngocminhta/FAIDSet) | 60,676 document-labeled rows acquired | Possible later academic domain check | No span labels and inconsistent card/file counts; inspect source rights and metadata first. |

The two OpAI slices came from **official upstream train files only**. Their
validation and test partitions are local derived holdouts, hashed by complete
seed-document group; they are not official benchmark splits. The original
snapshots remain untouched. The prepared files and manifest are under
`prepared_opai_v1/`. The splitter removes identical normalized texts within a
trajectory (one report row and 78 abstract rows) and one entire abstract
source group involved in a cross-group duplicate. It keeps each trajectory
together. Candidate-specific tests are useful for intake validation; existing
locked LLMTrace, AITDNA, article, and pure-text tests remain the primary
independent evaluation.

## Intake checks and selection

The normalized OpAI source files and GEN training slice passed full-span
coverage, unique-ID, and current-train/protected-evaluation overlap checks.
The reports' tagged text and AI intervals matched exactly on all 12,573
original rows. The abstract slice's four whole source groups with sampled
24-word or exact matches to current/protected data were excluded. GEN excluded
21 whole prompt groups with sampled phrase matches. The normalized human
samples also had valid full-text spans and no sampled matches to the checked
current/protected sets. The sampled-phrase audit is a screen, not a proof of
independence. Run it again on any final selected mix and resolve matching
groups before training.

For the next bounded data-mix ablation:

1. Keep the v10 model, tokenizer, windowing, and optimization fixed. Cap the
   existing DAMASHA synthetic source, then replace part of its token exposure
   with OpAI reports and a smaller share of OpAI abstracts. Sample complete
   seed groups before windows and cap versions per group, so nine correlated
   revisions do not dominate the objective.
2. Add GEN full-AI generations with model-family and domain caps. Match them
   to independently sourced human text rather than using GEN's raw 8:1 class
   distribution. Use PeerPrism only for full-human/full-AI peer-review windows,
   after checking source terms; do not label its hybrid rows as mixed spans.
3. Treat the fiction set as a separate human stress evaluation initially.
   Diagnose SciTechNews rights and provenance before relying on it as clean
   training or locked evaluation data. Keep Amazon reviews as a lower-confidence
   stress set.
4. Compare at a **fixed human-token FPR** and include human document FPR, AI
   document recall, mixed AI-token recall, boundary quality, and per-domain
   breakdowns against the current checkpoint and the two open Pangram
   baselines. Preserve the existing locked tests for the decisive comparison.

The strongest new labels are still **synthetic**; this search did not uncover
another well-documented, naturally observed collaboration corpus with gold
character boundaries. Better mixed-text realism remains a data gap. The GEN
edit file could later be aligned conservatively, with ambiguous unchanged
regions masked, but inferred alignments would not be gold supervision.

See [mixed-span candidates](hf-real-mixed-span-candidates-v1.md),
[AI-source candidates](hf-ai-span-candidates-v1.md), and
[human-source candidates](hf_human_span_candidates_v1.md) for source-level
evidence and rebuild commands.
