# Span detection and long-document roadmap

Status update (2026-09-26): a **binary human/AI token-labeling pilot** is implemented and evaluated on known-origin synthetic spans. See [the pilot report](token_span_pilot_v3.md). It uses Repeat2 first-copy masking, 512-token source windows with stride 256, and a character-offset span predictor. The AI-assisted class and realistic reviewed mixed-document data remain future work. The audit below describes the earlier passage-only state and the longer-term design.

## Verified current implementation

- `scripts/train_segment_lora.py` uses Qwen sequence classification, two output classes, and one last-position classification head.
- LoRA targets attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj` and feed-forward projections `gate_proj`, `up_proj`, `down_proj`.
- Training tokenization truncates each input to 512 tokens. `scripts/evaluate_diverse_lora.py` likewise scores a truncated input, including its paper-body audit inputs.
- There is no Repeat2 construction, token loss, assisted label, mixed-authorship head, sliding-window assembly, or span decoder in the current training/evaluation scripts.
- Existing ROC results establish passage discrimination under that pipeline. They do not establish full-paper coverage or authorship localization.

## Intended task and label policy

Detect substantial original prose from open-ended generation, and eventually return character-aligned human, AI-assisted, and AI-generated spans. A model's confidence is distinct from authorship class: uncertainty is not an AI-assisted label.

Write a labeling guide with examples for unchanged human writing, light proofreading, substantial rewriting, open-ended generation, quotations, and factual/formulaic content. Avoid treating document-level binary labels as precise token provenance.

Audit existing task provenance against this scope, particularly MAGE's `squad`, `tldr`, and `xsum` sources. A source name alone does not establish whether a row represents open-ended original writing, extractive answers, summaries, or copied material. Freeze a new version if scope changes; do not silently relabel the current experiment.

## Proposed next experiment

Retain the current checkpoint as a candidate initialization and binary baseline. Add a shared linear projection at every token position. First validate human/AI localization using known provenance; enable the assisted class once adequately labeled editing data exists. An untrained third output is not a supported capability.

For a source window `x` of length S, construct `[x, x]` and token targets `[-100] * S + labels`. Mask padding and any special tokens. Compute token loss only for the second copy and use its predictions at inference. The first copy remains in the attention computation and receives gradients through its contribution to supervised positions; loss masking is not attention masking or detachment. Keep token labels aligned to the same positions, without the one-token shift used for language modeling.

Use a binary head as an initial auxiliary objective only where its target is meaningful. A mixed-authorship head can be derived from known span labels. Test additional heads against a token-only baseline before keeping them. Do not transfer the current HPO winner unquestioningly: doubled context, a new objective, and new labels change memory needs and optimization.

## Data and document processing

- Build mixed documents from human paragraphs and recorded AI continuations, insertions, or replacements, retaining exact character offsets. Include fully human and fully AI controls.
- Vary authorship order, span length, AI fraction, and insertion position. Keep both classes on similar topics and formats so joins do not reveal the answer. Use varied boundaries instead of only fixed alternating patterns.
- For AI assistance, retain the human draft, edit instruction, generated revision, and reviewed alignment. Distinguish preserved wording, rewritten human content, and novel additions. Automatically inferred edit labels are weak supervision and need independent review.
- Assign entire source documents and all related mirrors/revisions to a single split before making windows or synthetic mixtures. Keep source and generator holdouts.
- Begin with 512 source-token windows, stride 256, and an end-anchored final window. Repeat2 makes each full model input approximately 1,024 tokens. Maintain offsets into the extracted original text.
- Merge overlapping token logits before calibration and decoding. Evaluate both raw token predictions and any sentence/span smoothing; smoothing can hide short AI insertions.
- For PDFs, preserve a mapping from extracted prose to pages/spans. Track headers, references, equations, tables, and extraction artifacts explicitly. State which text was scored.
- Sample training windows throughout documents, including boundaries and endings, rather than only prefixes. Balance by document so long papers do not dominate simply by producing more windows. This remains a model for multiple writing domains.
- Report spans, class fractions, and calibrated confidence. Do not declare an entire long document AI solely because one overlapping window crosses a threshold. Calibrate the complete aggregation pipeline on whole documents.

## Evaluation and error analysis

Keep development/calibration material distinct from a blind final test. A new protocol and repeatedly inspected tests require fresh confirmation data.

Measure recall at fixed human FPR, FNR (= 1 - recall for the same binary operating point), precision, and uncertainty intervals. Break results down by source, category, length, generator family, and paper section. For long mixed documents, also measure human characters falsely highlighted, AI span recall by span length, boundary error, class-fraction error, and the fraction of entirely human documents receiving any false highlight. Bootstrap by original document, not correlated windows.

Prioritize independent human writing, non-native English, light proofreading, complete scientific papers, creative writing, short passages, style imitation, and known-provenance mixed documents. Maintain an error-review table with text, provenance, length, score, generator, and suspected failure mode. Review development errors for changes; reserve the final test for measurement.

Treat published generator rankings as hypotheses to investigate. Direct performance comparisons require shared samples, compatible task definitions, and matching operating points. Multiplying published error rates by a constant does not supply those controls.

## Explicitly deferred

- Hard-negative mining: later, from a reserved development pool, with verified provenance and related AI mirrors. Never turn final test errors into training rows while continuing to call that test unseen. Estimate the benefit experimentally rather than assuming it is small.
- Humanizer head: outside the requested core architecture.
- Generator and named-author attribution: later exploratory probes on frozen detector features. Keep probe gradients and outputs isolated from the detector, allow unknown authors/models, and control topic, work, and prompt leakage. Joint training is a separate ablation with possible positive or negative effects.

Reference: [Pangram 4 technical report](https://pangram-public.s3.us-east-1.amazonaws.com/pdf/pangram_4_technical_report.pdf), particularly the task formulation, training, and evaluation sections. This document records our proposed design and code audit, not reproduced Pangram results.
