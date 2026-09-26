# External span data audit (26 September 2026)

The normalized files and source manifests are on the external disk under
`/mnt/f/pangram-at-home/data/span_sources_v5/`. Source revisions and file hashes
are recorded in each manifest. None of these data have entered the current
training mix. The AITDNA set is reserved for evaluation.

| Source | English documents available | Role | Main limitation |
| --- | ---: | --- | --- |
| LLMTrace Detection | 27,754 train; 5,534 validation; 7,012 test | Candidate span training and external evaluation | Constructed mixtures; 17 English rows had invalid intervals and were excluded |
| AITDNA | 362 documents from 99 writers | Locked real collaboration evaluation | Small author sample; surviving-character attribution differs from a simple AI-assisted label |
| DAMASHA clean | 96,119 parsed, unique candidate documents | Candidate augmentation only | Published aggregate has no upstream corpus or prompt IDs, and overlaps protected evaluations |

AITDNA's span, token, and other views are projections of the **same 362
documents**, not 3,258 independent documents. Its span and token views
reconstruct different text, so their offsets must not be mixed. In the span
view, 103 final documents contain only human text, 258 contain both, and one
contains only surviving model text. The source's `human_only` metadata counts
95; it describes the writing condition rather than surviving authorship.

LLMTrace has nine English domains: stories, reviews, news, questions,
articles, factual text, poetry, short-form text, and paper abstracts. In its
training split, 10,258 documents are human, 10,625 mixed, and 6,871 AI.
There are 4,754 documents with multiple AI intervals; 9,763 documents have
fewer than 80 whitespace-separated words. Topic IDs and exact texts do not
cross its supplied train, validation, and test splits. A future training mix
should stratify by domain, authorship and length and remove matched topic
groups before sampling.

The pinned DAMASHA clean CSV has 96,692 rows although the paper describes
95,859 clean examples. Parsing the published tags excludes 14 malformed rows
and 559 exact duplicate texts. The resulting candidate set includes 14,137
documents with multiple AI regions. The aggregate includes TriBERT and M4GT,
so those cannot be counted as independent additional corpora. It has no
recoverable work or prompt IDs for a defensible new split.

## Overlap audit

The audit compares normalized exact text and a deterministic sample of
24-word phrases against existing training, validation and frozen test files.
This detects direct reuse, but absence of a match does not prove independent
source or prompt lineage.

| Candidate | Protected-set matches | Decision |
| --- | ---: | --- |
| LLMTrace train | 7 rows, 5 topic groups touch frozen diverse test | Exclude those topic groups before any training |
| LLMTrace test | 3 rows touch frozen diverse test, including 1 exact text | Exclude matched records from a future independent benchmark |
| AITDNA | No sampled or exact matches found | Retain as locked external test |
| DAMASHA clean | 735 rows touch locked human test, including 536 with at least 3 sampled phrase matches; 41 touch diverse test | Do not merge wholesale; require source reconstruction and decontamination |

Full reference counts, source hashes, and matched row IDs are in
`/mnt/f/pangram-at-home/data/span_sources_v5/overlap_audit.json`.

## AITDNA result at frozen 5% document-calibrated thresholds

The thresholds were set on the separate pure-human calibration set before
AITDNA was scored. These results use the same 362 documents and all labeled
source tokens (97.8% of source tokens). They are insertion/surviving-character
provenance scores, not a general verdict on whether a person was assisted.

| Metric | Repeat2 v3 | Repeat2 v4 |
| --- | ---: | ---: |
| AI token recall | 39.1% | 88.4% |
| Human token false positive rate, all documents | 1.5% | 9.3% |
| Human token false positive rate, mixed documents | 2.5% | 18.0% |
| Mixed AI spans with any highlight | 29.0% | 56.6% |
| Mixed AI spans at least half highlighted | 14.1% | 53.4% |
| Pure-human documents with any false highlight | 1/103 | 1/103 |
| Token AUROC | 0.958 | 0.956 |

The almost unchanged AUROC alongside large recall and false-positive changes
means the v4 result is a different operating tradeoff on this benchmark, not
clearer ranking of human and AI tokens. The high mixed-document false positive
rate deserves investigation before selecting v4 for deployment. CoAuthor is
also a real collaboration set but shows only 3.3% AI-token recall for v4 at
this threshold; its labels mask much more source text and derive from a
different editing process. Both benchmarks should remain visible.

Primary sources: [LLMTrace dataset](https://huggingface.co/datasets/iitolstykh/LLMTrace_detection),
[AITDNA dataset](https://huggingface.co/datasets/UKPLab/AITDNA),
[DAMASHA paper](https://aclanthology.org/2026.findings-eacl.326.pdf).
