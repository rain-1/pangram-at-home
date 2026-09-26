# Attribution data: first collection

This collection supports separate exploratory author and AI-generator probes.
Full text is stored under `/mnt/f/pangram-at-home/data/`; it has not been added
to the detector's training or evaluation data. No attribution head has been
trained yet.

## Named writers

| Author | Train | Validation | Test | Total essays |
| --- | ---: | ---: | ---: | ---: |
| Gwern | 53 | 11 | 11 | 75 |
| Paul Graham | 56 | 12 | 12 | 80 |
| Scott Alexander | 69 | 15 | 15 | 99 |
| Eliezer Yudkowsky | 31 | 7 | 7 | 45 |
| **Total** | **209** | **45** | **45** | **299** |

We collected first-party essays by Gwern, Paul Graham, Scott Alexander, and
Eliezer Yudkowsky, prioritizing pre-2023 work by these contemporary writers.
Older work fills gaps. Known AI-writing demonstrations, guest posts and
comment compilations were excluded where identified; residual quotations and
unidentified third-party passages remain a limitation.

The labels mean **published-author attribution**, not guaranteed unassisted
human authorship. In particular, the Scott and Eliezer copies have uncertain
revision histories. Gwern's explicit version dates and some archived Graham
pages provide stronger evidence, but that evidence does not cover all four
authors. We therefore use the explicitly named `attributed` pool for the
four-author experiment.

The frozen split files are
`/mnt/f/pangram-at-home/data/author_attribution_probe_v1/{train,val,test}.jsonl`.
Their manifest records source and output hashes. The two acquisition packages
contain 304 records; five Gwern pages were quarantined before splitting.

Whole essays are grouped by source work, canonical URL, normalized duplicates
and substantial shared passages before splitting. Each author's oldest essays
enter training and newest essays enter testing. Future windows must inherit
the essay's split. The small holdouts will make per-author estimates uncertain;
they are suitable for a first probe, not a definitive attribution benchmark.

Acquisition details: [Gwern and Graham](attribution-gwern-graham.md),
[Scott and Eliezer](attribution-scott-eliezer.md).

## AI generators

| Source | Train | Calibration | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Arena Prose | 3,438 | 715 | 745 | 4,898 |
| LMArena preference sample | 4,873 | 1,052 | 1,041 | 6,966 |
| **Total** | **8,311** | **1,767** | **1,786** | **11,864** |

Answers to the same or near-duplicate prompt stay together. Arena Prose has
50 published model labels answering a common set of 100 prompts. The filtered
LMArena sample has 53 published labels; 37 meet the provisional minimum counts
for a separate probe. Exact labels, source revisions and available generation
metadata are retained.

These are two source-specific classification tasks. The label sets use
different naming conventions, and aliases have not been verified. A combined
103-label score could reward learning the source rather than the generator.
See the [generator report](attribution-generators-v1.md) for filters, grouping,
self-identification flags, overlap checks and source limitations.

The linked [SynthPrompts collection](https://huggingface.co/datasets/lyraaaa/synthprompts_v2_250k)
contains prompts written by Gemma, not answers from multiple labeled models.
We saved 2,000 English prompt candidates separately for possible controlled
generation later. No generation API spending was needed for this collection.

## Proposed frozen-head experiment

Freeze the detector and its adapter, cache document representations, and train
separate regularized linear heads. Compare with TF-IDF and frozen base-Qwen
representations. Report macro F1, balanced accuracy and confusion matrices,
including a diagnostic with explicit author/model names masked.

Four-author classification only chooses among those four people. Unknown
authors, AI imitations and unseen generators need separate rejection tests
before treating the output as an attribution claim. The
[experiment plan](attribution-probe-plan-v1.md) describes these checks.

The active Vast Repeat2 experiment remains separate. Code and reports are
versioned in Git; collected essay/response text remains external.
