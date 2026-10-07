# Backbone comparison on T2.1 + A2: Qwen3.6-35B-A3B vs Nemotron 3.5 Lightning 30B-A3B (October 7, 2026)

Both MoEs train on woog's T2.1 `prepared-v2` plus the layout-neutral heterogeneous-ai-spans additions (arm A2 of
`../data-ablation-20261007`), at 20% length, with the recipe of `moe-base-lr1e4-w10-s1` (`../hparam-sweep-20261006/moe.json`).
Each run uses 4 A100s. Code is the hparam-sweep trainer with these changes:

- `adapters_short.py`: expert LoRA also targets `up_proj` (Nemotron-H experts are relu2 without a gate); MoE check reads `n_routed_experts`.
- `nemotron_fast.py`: Triton Mamba-2 chunk scan from a `--no-deps` mamba_ssm copy (`vendor-mamba`, installed with
  `MAMBA_SKIP_CUDA_BUILD`). 20–30x faster on Mamba layers than the PyTorch reference; agrees to 1.1% without MoE layers
  (with MoE layers, rare routing flips give larger per-token differences).
- `train_sweep.py`: overlong rows (Nemotron's tokenizer puts 6–16% of the Qwen-cut windows over 510 tokens) are re-cut to 510 tokens
  centred on the target span instead of dropped. `--typo-aug` added (`typo_aug.py`), off for these runs.
- `setup_nemotron.py`, `prepare_model.py`: Space setup; a native-tokenizer `prepared-v2` (base data) was also built and is unused.
- `run_arm.sh` (train, sweep eval, benchmark-v3 scoring over 4 GPUs, woog's held-out battery, copy to `/data`), `gate_launch.sh` (start when GPUs idle).
- The held-out battery uses woog's `cross_model_eval.py` and inputs, copied on the Space from `/tmp/pangram-ablation-20261007`.

Weights were downloaded on the Space (pinned revisions in `models.json` on the Space). Final adapters and records:
`/data/workspace/pangram-nemotron-20261007`.

## Results (seed 1)

Woog's evaluations, recall at 1% FPR:

| | All edits | Small | paper_v3 | Standalone | Strict held-out | Held-out writers | Step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 0.803 | 0.493 | 0.770 | 0.847 | 0.770 | 0.643 | 6.3 s |
| Nemotron 3.5 Lightning | 0.762 | 0.453 | 0.560 | 0.622 | 0.764 | 0.614 | 3.6 s |

Benchmark-v3 (native protocol, mean token score, 1% calibration FPR) against the `prepared-v2` base MoE (50% length):

| Panel | Base MoE | Qwen T2.1+A2 | Nemotron T2.1+A2 |
|---|---:|---:|---:|
| Fully AI core recall | 88.6 | 82.6 | 80.6 |
| Controlled replacement recall | 69.9 | 32.5 | 39.6 |
| Beemo AI / human-edited AI recall | 95.5 / 77.5 | 60.0 / 27.5 | 75.0 / 29.5 |
| Hetero human controls flagged | 31.3% | 7.7% | 8.7% |
| AITDNA human documents flagged | 11.6% | 0% | 1.1% |
| Source-matched human references flagged | 11.2% | 2.8% | 2.5% |

Qwen stays the better backbone. T2.1 trades general AI recall for precision; the base MoE also trained at 50% length, a confound.
