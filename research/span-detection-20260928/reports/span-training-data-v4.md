# Span training data v4

Built from the frozen `diverse_pyramid_v1` train and validation parents with
`python scripts/build_span_training_v4.py`. Files live on the external disk in
`/mnt/f/pangram-at-home/data/span_training_v4/`. The builder refuses to
overwrite an existing output. The test split is never an input.

This dataset exercises token labeling on more and longer windows while
preserving the original text and its provenance. The parent pool still limits
realism: its median source row is 112 words in training, so the longer
documents are **synthetic joins of unrelated source excerpts**, usually from
the same source collection. They are not continuous documents or verified
human edits of AI writing. The label of each component is inherited from its
parent row, and the separators between components have no label.

| Split | Documents | 512-token windows, stride 256 | Documents >512 tokens | Documents >1024 tokens | Human labeled tokens | AI labeled tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 5,000 | 7,600 | 1,443 | 280 | 1,029,320 | 1,010,126 |
| Validation | 600 | 1,051 | 220 | 67 | 148,233 | 137,275 |

Training token lengths range from 44 to 1,665; median 302, 75th percentile
582, and 90th percentile 899. Validation median is 321; 75th percentile 751
and 90th percentile 1,041. The train set uses all 10,000 parent rows, 18,200
times in total, with a maximum of three uses per row. Validation uses all 800
parent rows, 2,199 times, with a maximum of five. These counts show the limit
of apparent sample size: 5,000 composites do not represent 5,000 independent
original works.

The construction schedule is 20% intact source rows, 10% short joins of the
same class, 20% long joins of the same class, 25% short human/AI mixtures, and
25% longer human/AI mixtures. The short mixtures use a matched source pair
when one exists. MAGE has no matched pair in the frozen parent, so its short
mixtures join unmatched rows from the same source. The long joins draw five
to seven distinct rows from one source; some have a second class change.
Same-class joins expose the same separators, including paragraph breaks, as
mixed joins. This reduces the value of detecting a join alone.

The requested domain mix by document is paper 25%, creative 20%, reference
and educational 20%, reviews 15%, social/Q&A 15%, and news 5%. All six
domains and all 16 frozen parent sources contribute. This remains the parent
corpus's research mix, including its source-label and rights limitations;
it does not add new human provenance or new AI generators. The per-source
and per-domain labeled token counts, construction counts, hashes, and parent
references are in `manifest.json`.

Every span has a `components` record with its parent text ID, source/group
ID, label, generator, license, parent text hash, and source/output character
offsets. Each component's output text was checked byte-for-byte against the
corresponding substring of its parent. The audit also checked that all
component labels match parent labels, all document hashes match their text,
and no source text IDs, group IDs, or text hashes overlap the frozen test
parent. Train and validation parent IDs, groups, and text hashes are disjoint.
The span pipeline's 512/256 window tokenization produced the counts above.

The useful next experiment is a Repeat2 run with the already selected LoRA
and batch settings, using this dataset and enough steps to cover its 7,600
training windows. Evaluate against the prior synthetic confirmation set and
separately against independent long human documents. Keep the confirmation
set for measurement rather than selecting examples into training.
