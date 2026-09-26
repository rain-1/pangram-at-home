# Hyperparameter tuning results — diverse v3

24 trials finished: 7 completed 25,600 source examples and 17 were stopped by ASHA. Ray reported no trial failures. W&B marked some intentionally terminated trials as crashed because the old shutdown path did not finalize their runs; the shutdown code has since been corrected for future runs.

The selected configuration is learning rate **7.6077571e-05**, effective batch **8**, LoRA rank **32**, alpha **64**, dropout **0.068837**. Other settings: Qwen3-1.7B, BF16 LoRA without quantization, 512 source tokens, attention and feed-forward LoRA, weight decay 0.01, cosine schedule, 5% warmup, seed 42. There was no Repeat2. Batch 8 was implemented as microbatch 2 × accumulation 4.

Selection used the final reported score among full-budget trials: 70% standardized partial AUROC at ≤5% FPR plus 30% full AUROC. Each training run saved its checkpoint with best partial AUROC. The selected trial's saved checkpoint is also its final checkpoint, so its reported selected metrics match that checkpoint.

## Full-budget results

| Trial | LR | Batch | Rank | Dropout | AUROC | Partial AUROC | AI recall at val ≤2% FPR | Worst-domain recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00019 | 7.61e-05 | 8 | 32 | 0.069 | 0.99459 | 0.96808 | 93.50% | 85.00% |
| 00007 | 0.000101 | 8 | 16 | 0.114 | 0.99291 | 0.96651 | 93.50% | 90.00% |
| 00003 | 8.65e-05 | 32 | 32 | 0.019 | 0.99180 | 0.96647 | 95.00% | 91.25% |
| 00001 | 8.91e-05 | 16 | 16 | 0.129 | 0.99240 | 0.96577 | 93.50% | 87.50% |
| 00015 | 7.52e-05 | 8 | 8 | 0.125 | 0.99176 | 0.96449 | 92.50% | 87.50% |
| 00005 | 6.61e-05 | 16 | 32 | 0.067 | 0.99270 | 0.96375 | 92.75% | 86.25% |
| 00000 | 5e-05 | 16 | 16 | 0.100 | 0.98932 | 0.96282 | 92.75% | 87.50% |

## Interpretation

The reference trial (#00) used LR 5e-5, effective batch 16, rank 16, and dropout 0.1. The selected trial gained 0.527 percentage points of full AUROC and 0.526 points of standardized partial AUROC. Recall rose from 92.75% to 93.50%: three additional AI passages out of 400. Its worst-domain recall fell from 87.5% to 85.0%. Trial #03 instead achieved 95.0% overall recall and 91.25% worst-domain recall, so retain it as a confirmation candidate.

These are selection results on the same 800 validation passages (400 human / 400 AI), not independent evidence of improvement. Each parameter combination was run once. Jointly varied parameters and ASHA pruning do not establish that an individual rank, batch, or learning rate causes better performance. A threshold permitting up to eight human errors on this validation set does not guarantee 2% FPR on new writing.

The data mixture experiment automatically uses the selected configuration for all nine 4,000-row mixes, with 12,800 source examples seen per run. Results are still pending when this report was generated.

## Next controlled experiment

Run a matched single-copy / Repeat2 pair from the same base checkpoint, seed, data order, source-token cap, hyperparameters, and optimizer-step budget. Use BF16 for both to match tuning. If memory requires microbatch 1 and accumulation 8, apply it to both; this is a controlled local pair rather than an exact numerical reproduction of the remote microbatch configuration. Keep all 512 source tokens before duplicating to up to 1,024 model tokens. Repetition also applies during evaluation. Record wall time and peak GPU memory: equal examples imply more compute for Repeat2.

Repeat2 may help passage classification, but the passage head already sees the whole window. Its stronger motivation is token-level prediction with a causal backbone. A token pilot is a new objective and needs known span provenance; this sweep supplies starting settings, not proven token-task optima.

## Follow-up correction

The later local pair used microbatch 1 × accumulation 8 for both single-copy and Repeat2 runs. This retained the selected effective batch of 8 but changed the sweep's microbatch 2 × accumulation 4. No memory probe established that microbatch 2 would fail on the local 4080 before this change. The local pair is internally matched, while neither run is an exact reproduction of the Vast sweep winner. The old local `qwen3_17b_diverse_v1` run used the trainer's NF4 default; the two local HPO follow-up runs and the selected Vast run used unquantized BF16.

A later one-step memory probe at microbatch 2 succeeded for both 512-token single-copy and 1,024-token Repeat2 inputs, peaking at 4.23 GiB and 4.56 GiB reserved by PyTorch respectively on the 16 GiB RTX 4080. This does not prove a sustained run would have identical memory use, but the original reduction was not evidence-based. See [the probe record](metrics/local_microbatch2_probe.json) and [the wider evaluation](comparative_models_v1.md).

Artifacts: [charts](hpo_results_v3.pdf), [all-trial CSV](hpo_results_v3.csv), [machine-readable results](metrics/hpo_results_v3.json).
