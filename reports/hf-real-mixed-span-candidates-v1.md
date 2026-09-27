# Hugging Face mixed-authorship span data candidates

Reviewed 2026-09-27 for the basic span-attribution model. The target is data
that improves AI span recall without teaching source-specific shortcuts. In
particular, v10 has weak AI-token recall on LLMTrace (38.6% at 0.8% human-token
FPR) and peer-review text (16.5% recall; AUROC .755), so authentic mixed
revision and peer-review provenance are high-value gaps.

## Recommendation

No newly found Hugging Face source in this pass provides **observed human–AI
co-writing with trustworthy character/token boundaries**. CoAuthor remains the
project's existing real interaction source and is excluded from this search by
request. The strongest immediately usable span candidate is synthetic
[OpAI-Bench](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench): it records
AI-generated characters through progressive edits, often with many boundaries,
across four domains and multiple generators. Treat it as controlled span
augmentation and keep its source groups intact. Its abstracts may overlap
existing academic corpora, so complete the source/group overlap audit before
training. The root agent is auditing the normalized candidate against protected
sources and evaluations.

[PeerPrism](https://huggingface.co/datasets/Reviewerly/PeerPrism) is the best
peer-review-domain candidate, especially given the current peer-review recall
gap. It contains genuine OpenReview human reviews and AI-produced rewrites or
hybrids, but every transformed review is labeled AI at the document level; its
`hybrid` rows combine human and AI *ideas* while the text itself is AI-produced.
It does not provide boundaries for text written by each party. Use it only for
document-level or fully-human/fully-AI domain augmentation, not mixed-span
supervision.

[FAIDSet](https://huggingface.co/datasets/ngocminhta/FAIDSet) also labels
human–LLM collaboration only at document level. Its academic texts could help
domain coverage, but the merged train file has a stale/inconsistent card count,
omits advertised language/domain/subtype fields, and does not expose spans.
Neither PeerPrism nor FAIDSet should be described as real mixed-span data.

## Acquired candidates

| Source (pinned revision) | Acquired files and counts | Label meaning | Assessment |
| --- | --- | --- | --- |
| [OpAI-Bench](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench) (`072c03c7051409346aaba05168d183cf6d863ee9`), Apache-2.0 dataset card | One train source file only: `abstracts_gpt-5.4-nano.csv`, 11,160 rows across 1,240 document groups and v0–v8 trajectories. After boundary and overlap exclusions, normalized candidate: 11,123 rows from 1,236 groups (1,236 human seeds and 9,887 mixed revisions). | Card defines `ai_spans_char` / `ai_spans_token` as AI-generated spans in synthetic AI-assisted revisions. Normalized spans partition the full text into AI/human labels. | Best span-training candidate, but synthetic. Source-document families are listed broadly as Kaggle essay data, XSum, government reports, and Kaggle arXiv abstracts; there is no per-row upstream source URI or license. The dataset card's Apache-2.0 tag does not resolve upstream terms. |
| [PeerPrism](https://huggingface.co/datasets/Reviewerly/PeerPrism) (`578a8e2c4a37d4b420429969a6128eaa8e625a73`), CC BY-NC 4.0 | Acquired `human-train.parquet` (674 human reviews) and `hybrid-train.parquet` (4,044 AI-text reviews with mixed idea origin). | `text_origin` and `idea_origin` are review-level categories; no character/token spans. “Hybrid” text is AI-produced from human and AI critique ideas. | Strong peer-review domain match and paired paper IDs; not usable as mixed-span labels. Human text is redistributed under source venue terms, which remain relevant. Noncommercial terms fit this project's stated research use; attribution is required. |
| [FAIDSet](https://huggingface.co/datasets/ngocminhta/FAIDSet) (`e2927dd1218b32767b212f822366c01bd406f5b3`), MIT tag | Acquired merged `train.jsonl`: 60,676 rows. Actual labels: 14,176 `human-written`, 14,409 `LLM-generated`, 32,091 `human–LLM collaborative`. | Whole-document label only; collaboration subtypes (polished, continued, paraphrased) are not retained in this merged file. | Academic theses/abstracts, but no spans and no row-level language/domain metadata. Card's train count sums to 58,343 and its AI count is 12,076, inconsistent with the acquired file. Underlying thesis/abstract rights need separate review; MIT tag alone is not proof of underlying text rights. |

Only the OpAI **train** split was acquired; no OpAI dev/test files were
downloaded. The PeerPrism and FAIDSet repositories expose only files named
`train` for the acquired configurations. Nothing from these candidates has
been used to train or evaluate a model.

## OpAI normalization and integrity

Raw and normalized data are outside Git under
`/mnt/f/pangram-at-home/data/candidate_span_sources/real_mixed/`. The helper
[normalize_opai_candidate_v1.py](../scripts/normalize_opai_candidate_v1.py)
maps the dataset's zero-based, end-exclusive character spans to
`id,text,spans[{start,end,label}],kind,source,group_id`, retaining
`document_hash_id` as `group_id` and the original document ID, domain, generator,
version, and split. Label `1` means AI-generated characters; unmarked
characters are human-seed characters. These are synthetic edit labels, not
keystroke-verified author provenance. Source token labels were not copied
because their tokenizer is not documented as matching this project's Qwen
tokenizer; derive model token labels from the preserved character offsets.

The first intake pass found sampled 24-word matches in 8 rows against current
training and in 6 rows against locked LLMTrace, including one exact-text hit.
Those matching rows map to 4 document groups; all rows from those groups are
excluded. The re-audit of the resulting file reports 11,123/11,123 span-valid
rows and zero matches to current training or protected evaluations. It also
reports 79 normalized-text duplicates. Before sampling, deduplicate identical
normalized text where appropriate and cap per document group; preserve the
group split so versions of one source cannot cross train/eval boundaries.

For context, a separate exact-text check after whitespace collapse and
case-folding found 39 duplicate clusters: 38 are within one group and mainly
look like unchanged adjacent trajectory stages, while one collision is between
two different v0 groups. The audit's normalized-text rule is broader and its
79 count should guide intake deduplication.

The raw file has SHA-256
`14e1c8a6ca00d6f5572e28c18f4011f35f5b45d24b0a86bf11f5ac5200039ee6`.
Normalized spans form a full-coverage, ordered, non-overlapping partition
within each Python text. One source row had span `[1724,1854]` for a text of
length 1,853 and was excluded rather than clamped. Four groups (36 rows) with
protected-source overlap matches were excluded. The result has 11,123 rows
across 1,236 groups; the bad-offset row means 1,235 groups have all nine
versions. The source's `ai_char_ratio` differs from span-derived coverage in
three raw rows by at most 0.000537 (including the excluded row). Preserve
offset spans as labels and investigate if selected for intake. The normalized
file has zero invalid boundaries and SHA-256
`5ca59bb1d298065fa850f588c709ccbb3ac7d95b4cbfda015864ace0aa35d13f`.

Downloaded file hashes:

- FAIDSet `train.jsonl`: `04d0d0328bdafacfdd30ff4ccdaec5e7acee2ca395ee7fa9edfcea05b4ea9d9a`
- PeerPrism human: `522b0f3f35aa9aae165d3c14b26894dc5db0ce3e13344182672ca223d65aba37`
- PeerPrism hybrid: `708a341250c4ea2dc744e2b128c16d344927ea2c89ff836649211455ec36ef40`

## Source references

- OpAI-Bench [dataset card](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench), [paper](https://arxiv.org/abs/2606.06481), and Croissant `prov:wasDerivedFrom` references to [Kaggle student essays](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data), [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset), [GovReport](https://gov-report-data.github.io/), and [Kaggle arXiv abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts).
- PeerPrism [dataset card](https://huggingface.co/datasets/Reviewerly/PeerPrism) and [paper](https://arxiv.org/abs/2604.14513).
- FAIDSet [dataset card](https://huggingface.co/datasets/ngocminhta/FAIDSet) and [paper](https://arxiv.org/abs/2505.14271).
