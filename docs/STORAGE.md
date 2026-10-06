# Training storage index (`open-text-detector/training-storage`, mounted at `/data` on the Space)

Updated 2026-10-06 after a cleanup (1,074 GB → 652 GB). Sizes are approximate. Statuses: canonical data, current experiments, data pipelines, archived (weights removed), caches.

Rules: ask the owner before deleting anything here; keep the selected checkpoint of any run in use (see `stage2-selection.json`), not only the last epoch; never delete the MoE base weights; bundle bulk transfers (tar shards) instead of per-file uploads.

## Canonical paper datasets (`datasets/`)

| Folder | Size | Status / purpose |
|---|---|---|
| `datasets/iclr2027-complete-20261004` | 238.7 GB | Source of truth for ICLR 2027: 42,422 PDFs, extracted text and position sidecars (41,004 with extraction). |
| `datasets/iclr2027-extraction-backfill-20261005` | 491 MB | Extraction for the 1,415 complete-mirror papers that had none. |
| `datasets/iclr2027-clean-text-v2-20261005` | 9.8 GB | positioned-clean-v2 text for 41,004 papers (cleanup code commit c775a07). Input to the Atlas classifications. |
| `datasets/iclr2027-backfill-clean-text-v2-20261005` | 337 MB | positioned-clean-v2 text for the 1,415 backfilled papers. |
| `datasets/iclr2027-clean-text-v3-figures-20261005` | 0.7 GB | Experimental: v3 cleanup with figure text removed (2,953 papers with MinerU layout data). |
| `datasets/iclr2027-mineru-markdown-20261005` | 0.6 GB | MinerU2.5-Pro Markdown and layout JSON (~2,950 papers). Model-generated; reference only, never training text. |
| `datasets/iclr2027-mineru-9page-300-20261005` | 20 MB | MinerU subset: 300 papers, pages 1–9. Reference only. |
| `datasets/paper-text-positions-batch001-20261005` | 4.4 GB | Position sidecars for the 14,561-paper archive (2020–2026, several venues). |
| `datasets/paper-text-clean-v2-batch001-20261005` | 3.2 GB | positioned-clean-v2 text for the 14,561-paper archive. |
| `datasets/iclr-2024-2025-openreview-sample-20261005-source` | 19.1 GB | 3,477 ICLR 2024/2025 PDFs from the OpenReview batch API (tar shards + table), papers already in the archive excluded. |
| `datasets/iclr-2024-2025-openreview-sample-20261005-extraction` | 1.0 GB | Baseline extraction of the OpenReview sample. |
| `datasets/iclr-2024-2025-openreview-sample-20261005-clean-text-v2` | 0.7 GB | positioned-clean-v2 text of the OpenReview sample. |
| `datasets/pangram-paper-text` | 481 MB | Canonical extracted text, one row per PDF (see its README). |
| `datasets/synthetic-mirrors-luna-28120` | 202 MB | 28,120 GPT-6-Luna synthetic mirror passages. |
| `datasets/iclr2027-round2-6949-20261003` | 23 MB | SUPERSEDED. PDFs and extractions removed on 2026-10-06 (all in the complete mirror). Root files kept: `result_codec.py` is imported by `scripts/omission_*.py`. |
| `datasets/iclr2027-round3-7000-20261003` | 24 MB | SUPERSEDED. Same as round 2; `result_codec.py` kept for the omission scripts. |

## Current experiments and results

| Folder | Size | Status / purpose |
|---|---|---|
| `overnight-sweep-20261004` | 94.7 GB | Oct 4–6 LoRA sweep and MoE runs (`h200/`), including moe-A-s1, moe-A-full, moe-A-full-lr1e4. All checkpoints kept. |
| `splice-sweep-20261006` | 3.3 GB | Splice-edit training waves (SPH, SPG; wave 2 LLE/MIX/Arep): final checkpoints and evals. |
| `backbone-fast10-20261003` | 18.7 GB | Oct 3 fast10 backbone runs. The Atlas classifier is `runs/qwen35-4b/run/stage2-epoch0-adapters.safetensors` (the selected checkpoint, not the last epoch). |
| `backbone-launch-20261003` | 197.8 GB | Base model assets for the backbones (`assets/`, includes the Qwen3.6-35B-A3B MoE base: never delete), Oct 3 H200 checkpoints, prepared data. |
| `evaluations-20261003` | 0.6 GB | Six-backbone evaluation outputs on the frozen suite. |
| `evaluation-suite-v1-20261003` | 74 MB | Frozen evaluation suite and runner. Its USAGE.txt example points at archived runs whose weights were removed; use current runs (e.g. backbone-fast10) and `current-data-v1/run/tokenizer` as the reference tokenizer. |
| `classifications` | 4.8 GB | ICLR 2027 Qwen3.5-4B scores (tokens, sentences, flag rates), calibration inputs and runs, eval-suite scores, year diagnostic, scoring scripts and vendored packages. |
| `current-data-v1` | 6.0 GB | Oct 3 current-data ModernBERT run; `run/tokenizer` is the reference tokenizer used by evaluations. |
| `baseline-omissions-20261005` | 1.7 GB | Omission-step prototype runs (v1–v5+), review kits and held-out page sets. |
| `mineru-markdown-20261005` | 3.3 GB | MinerU run code and logs. |
| `iclr2027-extraction-backfill-20261005` | 0 MB | Backfill run logs (dataset itself is under datasets/). |
| `artifact-shortcut-20261005` | 24 MB | Artifact-shortcut test outputs. |
| `ngram-dashboard-20261005` | 17 MB | N-gram statistics and dashboard. |
| `clause-labeling-20261006` | 463 MB | Clause-labeling (soft n-gram label) experiment units and logs. |

## Training-data pipelines (Oct 1–3)

| Folder | Size | Status / purpose |
|---|---|---|
| `human-source-mix-v1` | 0.8 GB | Human source mix, first intake (candidates, not admitted; see README). |
| `human-source-mix-v2` | 7.7 GB | Human source mix v2 (see its pipeline README). |
| `human-source-mix-v2-recovered-20261002` | 3.8 GB | Recovered v2 pipeline: stages and scaling data kept; pipeline checkpoint databases removed on 2026-10-06. |
| `synthetic-mirrors-v1` | 0 MB | Early synthetic mirror run. |
| `synthetic-mirrors-luna-dollar-v1` | 131 MB | Luna mirror generation run. |
| `synthetic-mirrors-luna-dollar-v1-recovered-20261002` | 1.4 GB | Recovered Luna mirror generation outputs. |
| `open-pangram-private-copy` | 1.7 GB | Private copy of the open Pangram materials (purpose not re-verified). |
| `baseline-mix-v1` | 265 MB | Baseline training mix (Oct 3). |
| `dataset-publication-20261003` | 164 MB | Dataset publication staging (Oct 3). |
| `stage4-labels-v1` | 0 MB | Stage-4 label files (purpose not re-verified). |
| `score-audit-v1` | 0 MB | Score audit outputs (Oct 2). |

## Archived experiments: weights removed 2026-10-06, results/configs/logs kept

| Folder | Size | Status / purpose |
|---|---|---|
| `paper-lora-comparison-v1` | 197 MB | Matched LoRA comparison (ModernBERT, Qwen3-0.6B, Qwen3.5-4B), Oct 1. |
| `paper-backbone-comparison-v1` | 72 MB | Encoder vs causal backbone comparison, Oct 1. |
| `paper-diversity-v1` | 2.2 GB | No-generation diversity pilot (RAID/MAGE/boundary arms), Oct 1–2. Note: evaluation-suite USAGE example references `control/run`. |
| `paper-batch-sweep-v1` | 106 MB | Batch size / LR sweep and throughput follow-up, Oct 1–2. |
| `paper-mix-sweep-v1` | 47 MB | Data mix sweep, Oct 1. |
| `paper-v3-modernbert-20260930` | 480 MB | First ModernBERT baseline and its wide evaluation. Tokenizer files kept; weights removed. |
| `gpu-perf-l40s-a100-20261001` | 25 MB | GPU performance comparison (assets.tar removed). |

## Caches and small files

| Folder | Size | Status / purpose |
|---|---|---|
| `model-cache` | 21.7 GB | Hugging Face model cache used on the Space (embeddings, Qwen, ModernBERT). |
| `hf-home` | 15 MB | HF_HOME for Space jobs. |
| `wandb-tracking` | 0 MB | W&B tracking helper. |
| `smoke-tests` | 0 MB | Smoke-test records. |
| `_cleanup` | — | Deletion manifests from storage cleanups. |

## Other files

- `upload-verify-batch001.log` (0 MB)
- `upload-verify-batch001.py` (6 MB)

## Outside `workspace/`

- `magpie-sweep-20261003`, `-round2`, `-round3-pilot` (~0.2 GB total): early Magpie generation sweeps.

## Space-local `/tmp` (not in this bucket; wiped on every Space restart)

Working copies only. Anything needed later must be copied into this bucket.

| Path | Purpose |
|---|---|
| `/tmp/pangram-space-fast10` | Sweep code, assets, vendor packages, run folders (rebuilt after restarts) |
| `/tmp/pangram-splice-20261006` | Splice-sweep working directory (in use by running jobs) |
| `/tmp/pangram-classify-data`, `/tmp/pangram-classify-claims` | ICLR scoring inputs, scripts and the multi-GPU claim queue |
| `/tmp/pangram-eval-20261003` | Fast10 checkpoint/code bundles used for scoring |
| `/tmp/pangram-openreview-sample` | OpenReview sample working copy (backed up under `datasets/iclr-2024-2025-openreview-sample-*`) |
| `/tmp/pangram-atlas-details-qwen35-4b` | Atlas detail files (published to Cloudflare R2) |
| `/tmp/pangram-moebench-20261006` | A100 MoE benchmark: code, vendored libraries, 68 GB weight copy |
| `/tmp/pangram-tools` | poppler/tesseract (micromamba) and pypdf/Pillow for extraction |
| `/tmp/clause-venv`, `/tmp/clause-venv2` | Python environments for the clause-labeling job |
