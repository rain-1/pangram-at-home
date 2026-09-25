# Local 4080 Repeat2 and token pilot

The queue is `scripts/run_local_overnight.py`. Its total wall-clock budget is ten hours, including evaluation. It logs to W&B project `pangram-at-home`, group `local_repeat2_span_v1`, and writes live status to `F:/pangram-at-home/overnight_repeat2_span_v1/status.json`.

## Matched passage comparison

| Setting | Single copy | Repeat2 |
| --- | --- | --- |
| Base | Qwen3-1.7B, BF16 LoRA | Same |
| Training / validation | Frozen diverse v1: 10,000 / 800 passages | Same |
| Seed | 42 | 42 |
| Source-token cap | 512 | 512 |
| Maximum model input | 512 | 1,024 |
| Effective batch | 8 (microbatch 1 × accumulation 8) | Same |
| Learning rate | 7.607757094e-5 | Same |
| Rank / alpha / dropout | 32 / 64 / 0.0688373663 | Same |
| Schedule | Cosine, 5% warmup, weight decay 0.01 | Same |
| Budget | 3,200 optimizer steps = 25,600 source examples | Same |
| Validation / checkpointing | Every 400 steps | Same |
| Early stopping | Disabled, except the overall time limit | Same |

Both start from the same base, not from a previously trained adapter. This isolates repetition within the local comparison. Microbatch differs from the remote sweep but is the same in both local runs. The same source tokens are retained before repetition; padding is added afterward. Validation follows the corresponding input construction. This is a sequence-level experiment, with one label at the final position, so first-copy token-loss masking does not apply to this pair.

The selected checkpoint for each run maximizes validation partial AUROC at ≤5% FPR. Report the complete learning curves and equal-budget endpoints too. Keep additional test suites untouched during this pilot. Repetition costs more computation at equal examples and is not assumed to improve passage classification.

## Token pilot

After the pair, train `qwen3_token_repeat2_pilot_v1` for 800 steps (6,400 windows), evaluating every 200. Transfer the selected remote winner's backbone LoRA weights and initialize a new token head. Transfer checks require identical LoRA module keys, rank, and alpha; the sequence head is discarded. The initial adapter SHA256 is `2af818a57d10139ad8f8e6889a560b870278a82c05099a216d5b998be6ee86c7`.

The pilot uses 1,200 constructed training documents and 240 development documents. Each split contains 25% pure human, 25% pure AI, and 50% alternating mixtures, sampled from its respective original split. No original text hash or group ID crosses train/validation. Mixtures combine excerpts from the same source category with exact offsets; labels inherit the parent rows' binary labels. The new construction does not independently verify every parent's provenance, and joins can create topic artifacts. It is an engineering and learning pilot, not a realistic mixed-authorship benchmark.

Use a binary token head, two copies of each source window, first-copy targets set to -100, and padding excluded from loss. Both copies remain visible to causal attention. No language-model label shift applies. There is no AI-assisted output or humanizer objective.

Windows cover each document with size 512, stride 256, and an end-anchored final window. Validation training metrics operate on supervised window tokens and therefore include repeated coverage. The subsequent document evaluation averages overlapping logits back onto source tokens, calibrates on pure-human development controls, and reports token recall/FPR, false human character highlights, any-highlight FPR on human documents, and error in AI character fraction. Results are development diagnostics; final test calibration and reviewed mixed data are separate work.

Evaluate both passage models on the same constructed documents using constant per-window scores as coarse localization baselines. Compare their assembled predictions with the token model. Do not compare token AUROC directly with passage AUROC: they measure different targets.

## Outputs and next decisions

Each run saves config, dataset hashes, metrics, best adapter, runtime, and peak allocated GPU memory on F:. At completion, `scripts/summarize_overnight.py` writes a local Markdown and JSON comparison. If the time cap interrupts the pair, mark it incomplete rather than interpreting unequal budgets as a controlled result.

Remaining prerequisites for a stronger model: reviewed span provenance, realistic coherent continuations/replacements, independent human and source holdouts, whole-paper extraction and offsets, a held-out calibration set, checks across seeds, and labels that distinguish substantial AI assistance from proofreading. Current tuning provides useful starting settings and a reusable adapter; it does not establish token-task optimality.
