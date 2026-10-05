# Inference preference

For project continuity after an account or chat change, first read `docs/HANDOFF.md`. It points to current research documentation, remote job recovery locations, and a dated automation snapshot. Refresh live state before acting on historical job status.

Use reduced-precision inference for every model run. Prefer BF16 on supported GPUs; do not run full-precision (FP32) inference or FP32 reference-inference comparisons. This is an explicit user preference. Standard mixed-precision accumulation and full-precision metric calculations are fine; model storage dtype does not determine inference precision.

# Model storage preference

Download new model weights, tokenizers, and model configuration assets only on the existing Hugging Face training Space, using its persistent `/data/workspace` storage. Do not download new model assets to the local computer. Copying existing files from the Space to other machines is allowed; those files do not need to stay exclusively on the Space. Ask for permission before permanently removing anything from the Space. Local source code, datasets, revision manifests, and small verification records are fine. Keep original trained checkpoints on the Space. This is an explicit user preference.

# Training Space data transfer approval

User exception on October 3, 2026: The three larger backbones in the initial parallel training batch (Gemma 4 12B, Qwen3.6-27B, Qwen3.6-35B-A3B) may be downloaded directly from Hugging Face to the H200 node. Use `/workspace/woog` for the user's work on that node. This does not authorize local-computer model downloads or removal of Space files; trained checkpoints should still be preserved on the Space.

User exception on October 4, 2026: Qwen3.5-4B and Qwen3.5-9B may also be downloaded from the public Hugging Face Hub directly to the H200 node (pinned revisions 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and c202236235762e1c871ad0ccb60c8ee5ba337b9a) for overnight training sweeps; the H200 has no Hugging Face login, so bucket copies are not reachable there.

User approval on October 3, 2026: All data transfers to the existing Hugging Face training Space `open-text-detector/training` are authorized on an ongoing basis, including local datasets, source code, tokenizer/configuration assets, and existing model files. Do not ask again merely to transfer data to that Space. This approval does not change the requirements to download new model assets only on the Space, keep original trained checkpoints there, or ask before permanently removing anything from the Space. Other destinations are outside this standing approval.

# Flex service tier preference

User approval on October 2, 2026: Flex-tier use is approved on an ongoing basis for otherwise authorized model runs; do not ask again merely to use Flex. Existing model choices, cumulative budgets, attempt caps and task scope still apply. Preserve provider-reported tier metadata honestly; a missing response tier is unverified, not proof of Flex execution.

# Classifier training dashboard

User preference October 3, 2026: Put every classifier training run in the same W&B project `rigg-alice0/pangram-text-classifiers`, run group `text-classifiers`, with a distinct descriptive run name. User explicitly approved disabling force-projects-private on October 3, 2026. Workspace rigg-alice0 now has privateOnly=false and defaultAccess=PRIVATE; classifier project is public (USER_READ), verified anonymously. Keep future-project defaults private; put all classifier training runs in this public project. User is the sole member and admin of rigg-alice0. The current run was moved from narmal; do not log new classifier runs there. Use `benchmarks/pangram4/training/wandb-tracking/attach.py` for the current-data trainer, or integrate equivalent tracking when introducing another trainer. Track training/validation losses, progress, timing and completion/failure. The current tracker deliberately excludes datasets, source files, checkpoints, configuration and machine metadata. Do not silently expand that payload. A successful training launch should also attach tracking and record the run URL; tracking must not restart or interrupt training. Never put credentials in repository files or logs.

# Training weight and checkpoint precision

User direction October 3, 2026: Storage is a hard cap. Use BF16 for trainable adapter/classifier weights and saved model checkpoints rather than defaulting to FP32 or keeping duplicate FP32 copies. Address any numerical issues through backend optimization instead of silently increasing weight/checkpoint precision. Full-precision metrics and standard mixed-precision accumulation remain allowed. Preserve existing trained Space checkpoints unless deletion is explicitly authorized; do not interrupt active runs merely to change this preference.

# Git repository and publication standard

User direction October 3, 2026: This folder is a git repository pushed to the private GitHub repo `woog97/paingram`. It stays private until the user announces the project, then becomes public with its full history. Treat every commit as eventually public.

- Commit code, docs, configs, prompts, reports and small records. Datasets, model weights, classification outputs, extractions and receipts larger than a few MB stay on Hugging Face/R2; extend `.gitignore` rather than committing them.
- Never commit credentials, `docs/HANDOFF.md` or `docs/handoff/`, infrastructure addresses, LLM call records or datasets containing third-party paper text, per-paper detector scores (`app/public/catalogue/`), or Pangram brand assets.
- Reach the H200 node with `ssh pangram-h200` (an alias in the local `~/.ssh/config`); never hardcode its address or port in repository files.
- A local pre-commit hook rejects files over 10 MB, credential-like strings and the H200 address. Fix the cause instead of bypassing it with `--no-verify`.
