# Frozen-detector attribution heads on Qwen3.5-4B (October 6, 2026)

This ports rain1's attribution-head experiment to the workbench detector (`research/span-detection-20260928/reports/attribution_comparison_v1.md`). On Qwen3-1.7B, linear heads on the frozen v10 span detector beat the same heads on the raw backbone:

| Backbone | Arena 50-model top-1 | Four-writer accuracy |
|---|---:|---:|
| Frozen v10 span detector | 48.7% | 12/12 |
| Raw Qwen3-1.7B | 42.7% | 11/12 |

Fine-tuning the whole backbone added nothing over freezing it (48.0% vs 48.7%). The Arena test has only 150 responses, and they come from just three held-out prompts, so the intervals understate uncertainty across new prompts.

## What runs

`attribution_heads.py --init {fast10, base, <sweep tag>}` loads the Detector three ways:
- `fast10`: the Atlas fast10 adapter (`backbone-fast10-20261003/runs/qwen35-4b/run/stage2-epoch0-adapters.safetensors`).
- `base`: the raw base model.
- `<sweep tag>`: a finished `hparam-sweep-20261006` run.

For each document it uses up to 8 windows of 510 tokens, with stride 256. Each window is laid out as Repeat2 with `data.layout` (stage 2), and the feature is the mean final hidden state over the second copy, averaged across windows. Each task then gets a class-weighted linear probe with early stopping on validation macro-F1. Features, heads, normalisation and `report.json` go under `attribution/<init>/<task>/`, and metrics go to the sweep's Trackio project (group `attribution-heads`).

## Data

The data is in `/data/workspace/attribution-heads-20261006/data` on the training Space, copied unchanged from rain1's `attribution_heads_v1`. Hashes are in its `manifest.json`.

- **arena:** `woog/arena-prose-100-49-models` at revision `66298b5…`. 4,998 non-empty responses from 50 models, split by prompt into 4,698 / 150 / 150.
- **authors:** four writers, split by work into 275 / 12 / 12. These essays have no redistribution licence, so they are research-only and are not in the public dataset (`open-text-detector/span-detection-rain1-v14` holds ids only).
