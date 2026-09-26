# Author and generator attribution probes

These are separate research datasets and tasks. They do not alter the binary
human/AI detector or enter its training, calibration, or evaluation splits.
Named-author supervision means attribution to a published author; it does not
certify that every sentence was written without assistance. Each record keeps
that distinction and its version/date evidence.

## First experiment

Freeze both the Qwen backbone and the detector adapter, set inference mode, and
cache representations with gradients disabled. Use the same source windows and
second-copy positions as Repeat2 inference, averaging only source-token hidden
states. Store the adapter, tokenizer, and data hashes alongside the cache.
Train separate regularized linear classifiers on those cached vectors: one for
the four named writers, and separate generator classifiers for each response
corpus. Save the probe weights separately from the detector adapter. This setup
cannot change the detector's parameters or predictions.

Compare the frozen detector representations against the unchanged base Qwen
representations and inexpensive word/character TF-IDF classifiers. Tune only
the probe's regularization on validation data. Report document-level macro F1,
balanced accuracy, per-class recall, and a confusion matrix. Aggregate window
predictions per document; windows from one essay are not independent test cases.
Cap training windows per essay or weight essays equally so long articles do not
dominate.

## Human authors

Requested authors: Gwern, Paul Graham, Scott Alexander, and Eliezer Yudkowsky.
The collection prioritizes dated pre-2023 material by contemporary writers;
older fallback articles have explicit dates. Current copies with uncertain
revision histories remain distinguishable from archived or explicitly dated
versions. Known generated-text demonstrations and comment compilations are
excluded from the preferred essay pool where identified.

`scripts/build_author_attribution_v1.py` groups canonical URLs, source work IDs,
normalized duplicates, and substantial shared passages before splitting.
Cross-author duplicate groups are excluded. The oldest groups per author enter
training, followed by validation, with the newest reserved for testing. This
provides a modest temporal check; it is not a deliberate topic-disjoint test.
The `attributed` pool measures published-author attribution. The stricter
`pre2023_candidates` pool requires stronger version evidence and may not contain
enough examples of every author. Neither pool guarantees unaided human authorship.

Run a diagnostic with author names, site names, and signatures masked, and keep
metadata out of the model input. Otherwise a classifier can learn explicit
self-identification or website boilerplate. Topic and publication format remain
possible confounds even after those strings are removed.

A four-author classifier answers “which of these four is the closest match?”
It does not establish that an arbitrary text came from one of them. Before any
user-facing attribution, add other contemporary authors and AI imitations as
unknown-author checks, calibrate abstention separately, and report those error
rates. Do not present a closed-set softmax score as proof of authorship.

## AI generators

[The generator data report](attribution-generators-v1.md) describes the downloaded
responses, source-specific task manifests, model names, and split counts.
All answers to the same normalized or near-duplicate prompt belong to the same
split. Shared-prompt Arena Prose responses give a useful control for topic.
Broader Arena conversations add coverage, but question selection, generation
settings, refusals, and response length can correlate with the model label.

Keep exact published model IDs distinct unless a verified mapping exists.
Provider names, aliases, and model release names do not establish an exact
weight revision. The two corpora currently have no identical published model-ID
strings, so a pooled exact-model score could exploit source differences. Prefer
within-source model classification and a separate cross-source family check
over shared, explicitly mapped families. Family mappings are hypotheses recorded
in the manifest, not additional ground truth.

Mask explicit model self-identification for a second evaluation, retain the raw
result for comparison, and hold out entire model variants to test rejection of
unseen generators. Exact-version attribution is a more ambitious target than
family classification.

[SynthPrompts](https://huggingface.co/datasets/lyraaaa/synthprompts_v2_250k) is a
separate pool of synthetic user prompts. It has no multi-model answer labels.
Its prompt writer must not be mistaken for the generator of a response that
does not exist yet. A later controlled collection could send the same eligible
open-ended prompts to several models and record exact request/response metadata;
no such generation or API spending has been performed for this collection.

No attribution probe has been trained yet. The active Vast run is solely the
Repeat2 span-data experiment.
