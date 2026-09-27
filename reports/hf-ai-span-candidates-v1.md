# Hugging Face AI text sources for span attribution

Researched and acquired 27 September 2026. Raw data and normalized candidates
are under `/mnt/f/pangram-at-home/data/candidate_span_sources/ai/`; none of the
corpus text is in Git. The pinned cards, raw-file hashes, and normalized-file
hashes are recorded in the external `source_manifest.json`.

## Recommended first intake: OpAI-Bench reports

[OpAI-Bench on Hugging Face](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench)
provides revision trajectories from human seed documents to AI-assisted
revisions, with character-level labels. I acquired only the official
`default/train/reports_gpt-5.4.csv` file at revision
`072c03c7051409346aaba05168d183cf6d863ee9`, preserving the source's train
role. The snapshot is 248,151,536 bytes. It contains 1,397 government-report
source groups, each with v0 (human seed) and eight mixed revision versions,
for 12,573 rows total. Each row records its generator, source ID, version,
operation, and AI character intervals.

All rows are normalized into full-coverage, alternating human/AI spans in
`opai_bench/default_train_reports_gpt54/normalized_train.jsonl` (68,246,191
bytes). The audit found all 12,573 span partitions valid, no duplicate IDs,
and no exact or sampled 24-word-window overlap with any current training or
protected evaluation reference checked. A separate label integrity check
verified every row: removing `<AI_Start>` and `</AI_End>` from `text_tagged`
reproduced `text` byte-for-byte, and the resulting intervals exactly matched
`ai_spans_char` (zero mismatches). The audit result is in the same external
directory as `intake_audit.json`.

The source card declares Apache-2.0 and documents four overall domains and
four generators: GPT-5.4, GPT-5.4-nano, Gemini 2.5 Flash, and Qwen3-8B. The
acquired slice adds only GPT-5.4 and the reports domain, which is a good
bounded first experiment. The card says source documents come from student
essays, news, government reports, and scientific abstracts; Croissant names
Kaggle AES2, XSum, GovReport, and arXiv abstracts as upstream sources. It
does not include the essay, news, or abstract files here. Government reports
are not a current project training source in the inventory, but keep
`group_id` intact and split by whole report before any chunking. v0-v8 for a
report must never cross train/dev/test. These are controlled synthetic edit
trajectories, not naturally observed writing sessions; use them to improve
span supervision while keeping real-use evaluation separate.

## Second candidate: GEN human references and generations

[GEN on Hugging Face](https://huggingface.co/datasets/szyszy/GEN) has 5,000
human references, 39,391 open-ended generations from eight named models, and
273,418 model edits of human seed text. Models include Gemma 3 12B/27B,
gpt-oss 20B/120B, Llama 3.1 8B/3.3 70B, and Qwen 2.5 7B/72B. Five domains
are scientific abstracts, Reddit answers, story generation, WikiHow, and
Wikipedia. Each AI generation has a `prompt_id` linking it to the human
reference and its prompt. The card reports temperature 0.7, top-p 0.9,
cleaning of common assistant artifacts, and CC-BY-NC-4.0. It also says human
references retain the licenses of their source corpora; those item-level
licenses are not identified in the downloaded rows, so preserve source IDs
and verify source terms before redistribution. The NC terms fit this
noncommercial research project.

Raw human, generation, and edit files are pinned at
`a0f143c1ae9c35b684898a5f9c0ab6efb49be486`. The normalized full-document
human/AI-generation candidates use group-hashed 80/10/10 splits by `prompt_id`
in `gen_noncommercial/normalized_by_prompt_split/`. The train partition has
3,989 prompt groups after excluding 21 entire groups with sampled phrase hits
against current training; that leaves 35,419 train rows (3,989 human and
31,430 AI generations). After exclusion, intake found all spans valid, no
duplicate IDs, and no exact or sampled-window matches to any checked current
train or protected evaluation reference. Its hash and details are in
`gen_noncommercial/train_intake_audit.json`. This source adds generator and
domain variety for fully human and fully AI documents; its document labels
alone do not supply mixed-text boundaries.

The large `ai_edited.jsonl` file is also acquired and preserved raw. It has
`seed_text`, edited text, edit prompt/category, editing model, and measured
edit ratio. The source labels the output as AI-edited but does not provide
span boundaries. We have therefore not converted it to gold span data.
Changed-word alignments against `seed_text` could serve as conservative
candidate supervision if edits are audited and unchanged or ambiguous
regions are masked; that would be an inferred alignment, not the source's
gold annotation. Group all variants by `seed_id` before splitting.

## Other reviewed leads

- [LEDE](https://huggingface.co/datasets/NeurIPS-2026-LEDE/LEDE-dataset) is a
  strong news-domain follow-up: 337,322 generated articles, 21 models, 17
  categories, and four generation strategies; English and Korean are
  separate. Its card declares CC-BY-NC-4.0 and names Newsroom, AI-Hub, Ayoobi,
  and ISOT as human source corpora. It currently exposes no direct span
  labels, no explicit source-group split, and an anonymous benchmark paper.
  It is a large all-AI benchmark candidate, not as ready for span training as
  OpAI.
- [Human-AI review deception](https://huggingface.co/datasets/gesis/human-ai-review-deception)
  labels human and LLM-written reviews and identifies GPT-4 and WizardLM
  generation. Hugging Face lists MIT, but the card gives no direct span
  annotation and leaves human-review source lineage unclear. The data also
  includes classifier predictions and linguistic features. This is worth a
  separate item-level source/license review for the model's review recall
  gap; it is not currently acquired as training data.

## Rebuild and audit

```bash
python scripts/prepare_candidate_span_sources_v1.py
python scripts/audit_span_candidate_intake.py \
  /mnt/f/pangram-at-home/data/candidate_span_sources/ai/opai_bench/default_train_reports_gpt54/normalized_train.jsonl
python scripts/audit_span_candidate_intake.py \
  /mnt/f/pangram-at-home/data/candidate_span_sources/ai/gen_noncommercial/normalized_by_prompt_split/train.jsonl
```

The normalization script stores only source hashes, metadata, and code in
Git. It does not train a model or use a GPU.
