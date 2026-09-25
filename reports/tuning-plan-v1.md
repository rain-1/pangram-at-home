# Qwen tuning and data study, version 1

## Objective and guardrails

Improve AI recall while keeping human false positives low across writing categories. The sweep trains on `diverse_pyramid_v1/train_full.parquet` and evaluates on its fixed validation split only. RAID, Enron, and the diverse test are absent from the Vast upload and are reserved for the selected final models. The first model's best adapter had validation AUROC 0.9905; its cached validation scores give standardized partial AUROC at ≤5% FPR of 0.953 and AI recall 91.3% at a validation-selected ≤2% human FPR.

Because validation has only 400 human passages, a 2% FPR target permits eight errors. Individual category estimates are noisier. Trial ranking uses 70% standardized partial AUROC at ≤5% FPR plus 30% full AUROC. We also log AI recall at ≤2% FPR and worst-category recall for inspection. We will not optimize against test scores.

## GPU pilot sweep

Use [Ray Tune's ASHA scheduler](https://docs.ray.io/en/latest/tune/getting-started.html) on one Vast.ai instance with four 24 GB GPUs. ASHA prunes weak trials after comparable amounts of training data. Each trial gets one GPU and the same Qwen3-1.7B revision, 512-token input, frozen data, and seed. Search 24 trials over:

| Parameter | Search |
| --- | --- |
| Learning rate | log-uniform 1.5e-5 to 1.5e-4 |
| Effective batch size | 8, 16, or 32 via gradient accumulation; microbatch fixed at 2 |
| LoRA rank | 8, 16, or 32; alpha = 2 × rank |
| LoRA dropout | uniform 0 to 0.15 |

Evaluation is every 3,200 examples. ASHA's first pruning point is 6,400 examples, and the maximum budget is 25,600 examples, roughly 2.56 passes over the 10,000-row training set. Comparing examples seen rather than optimization steps makes batch-size comparisons interpretable. All trials log to Weights & Biases and save local checkpoints. A short local smoke run checked the new training metrics and checkpoint reports.

The first rental is a four-GPU RTX 4090 host at $1.674/hour. The monitor limits the active sweep to 15 hours or $26 from monitor start, before storage and transfer; setup time incurred under $2. The first phase remains under $30, leaving over $120 of the stated $150 for longer confirmatory runs, dataset experiments, and scale studies. Use on-demand instances and destroy them when outputs are collected; stopped instances can still incur storage charges.

The host's CUDA stack failed on 4-bit bitsandbytes training. The remote sweep instead runs full BF16 weights with LoRA adapters; one-step training and validation completed on the host before the sweep started. This fits the 24 GB GPUs and does not change the trial search dimensions. The run configuration records the quantization mode.

HEBO is an available second-stage option through [Ray's HEBOSearch integration](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.hebo.HEBOSearch.html). The first phase uses random search plus ASHA, which is simple to parallelize and gives a diverse initial set of observations. We can let HEBO refine around the promising region after that data exists.

## Dataset ablations

The script `build_diverse_ablation_pilots.py` produced nine private 4,000-row mixes from existing training rows: the 35%-paper control, six leave-one-category-out mixes, and 20%/50% paper mixes. Every mix has 2,000 human and 2,000 AI passages, a fixed total size, and the same validation file. Excluded categories are replaced by examples from the other training categories, not by copies of test data. After selecting training parameters, run these nine with the same seed, number of examples seen, and model configuration. This estimates the benefit of each category without confusing it with total training size. It is an ablation of the current sources, not proof that all data in a category is equally valuable.

## Longer runs and scale directions

Take the strongest two or three pilot settings to longer runs on the full 10,000-row mix; compare them with the original model on validation. Select one based on partial AUROC, category recall, and stability across seeds, then evaluate once on the held-out test suites. Before declaring a final winner, collect a fresh blind test because earlier results have already been inspected.

After tuning, test scale in one dimension at a time: (1) more source-diverse training examples, (2) a larger Qwen backbone, (3) longer input windows for full paper prose, and (4) source-balanced sampling that reduces reliance on the MAGE family. Data breadth and label provenance remain likely constraints; hyperparameters alone cannot fix missing real-world domains.

## Reproduction

`train_segment_lora.py` exposes the pilot parameters and saves a frozen run config. `tune_diverse_ray.py` manages Ray trials, reports validation metrics after checkpoints are saved, and can run the ablation grid with a chosen HPO config. `package_vast_tuning.py` builds a private upload with code and train/validation rows; `bootstrap_vast_tuning.sh` verifies hashes and downloads the pinned public Qwen model. Raw text and tokens stay outside Git. Vast deployment follows the [official CLI workflow](https://github.com/vast-ai/vast-cli).
