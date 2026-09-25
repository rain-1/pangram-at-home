"""Run GPU-parallel Qwen pilot sweeps with Ray Tune and ASHA.

Run on one multi-GPU Vast instance. Only training and validation data are
needed on the instance; all test sets remain off the tuning machine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ABLATIONS = ["control_35paper", "without_paper", "without_reference_education",
             "without_creative", "without_social_qa", "without_reviews", "without_news",
             "paper_20pct", "paper_50pct"]


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def train_trial(config: dict, *, root: str, sweep: str, max_examples: int,
                eval_examples: int, report_to: str) -> None:
    from ray import tune

    trial_id = tune.get_context().get_trial_id()
    run_name = f"{sweep}_{trial_id}_{int(time.time())}"
    data_root = Path(root)
    run_dir = data_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "sweep_metrics.jsonl"
    effective_batch = 2 * int(config["gradient_accumulation_steps"])
    max_steps = math.ceil(max_examples / effective_batch)
    eval_steps = max(1, math.ceil(eval_examples / effective_batch))
    dataset_folder = ("diverse_pyramid_v1" if sweep.startswith("hpo")
                      else f"diverse_ablation_v1_{config['dataset']}")
    command = [sys.executable, "-u", str(REPO / "scripts/train_segment_lora.py"),
               "--root", root, "--model", str(data_root / "models/Qwen3-1.7B"),
               "--dataset-folder", dataset_folder, "--train-tier", "full",
               "--run-name", run_name, "--max-steps", str(max_steps),
               "--epochs", "8", "--hours", "5", "--eval-steps", str(eval_steps),
               "--train-batch-size", "2",
               "--gradient-accumulation-steps", str(config["gradient_accumulation_steps"]),
               "--learning-rate", str(config["learning_rate"]),
               "--lora-rank", str(config["lora_rank"]),
               "--lora-alpha", str(2 * int(config["lora_rank"])),
               "--lora-dropout", str(config["lora_dropout"]),
               "--metrics-jsonl", str(metrics_path), "--report-to", report_to]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = "pangram-at-home"
    env["WANDB_RUN_GROUP"] = sweep
    env["WANDB_DIR"] = str(data_root / "wandb")
    env["WANDB_CONSOLE"] = "off"
    env["TOKENIZERS_PARALLELISM"] = "false"
    print("trial", run_name, "config", config, "max_steps", max_steps, flush=True)
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        offset = 0
        reported = 0
        final_report = None
        try:
            while True:
                if metrics_path.exists():
                    with metrics_path.open(encoding="utf-8") as file:
                        file.seek(offset)
                        lines = file.readlines()
                        offset = file.tell()
                    for line in lines:
                        entry = json.loads(line)
                        examples_seen = min(int(entry["step"]) * effective_batch, max_examples)
                        score = (.7 * entry["eval_partial_auc_fpr_5pct"]
                                 + .3 * entry["eval_roc_auc"])
                        payload = {"examples_seen": examples_seen, "score": score,
                                   "val_roc_auc": entry["eval_roc_auc"],
                                   "val_partial_auc_fpr_5pct": entry["eval_partial_auc_fpr_5pct"],
                                   "val_ai_recall_at_fpr_2pct": entry["eval_ai_recall_at_fpr_2pct"],
                                   "val_worst_domain_ai_recall_at_fpr_2pct":
                                   entry.get("eval_worst_domain_ai_recall_at_fpr_2pct"),
                                   "run_name": run_name}
                        if examples_seen >= max_examples:
                            # Let Trainer finish writing best_adapter before ASHA closes the trial.
                            final_report = payload
                        else:
                            tune.report(payload)
                            reported += 1
                if child.poll() is not None:
                    if child.returncode != 0:
                        raise RuntimeError(f"Training failed (exit {child.returncode}); see {run_dir / 'train.log'}")
                    if final_report is not None:
                        if not (run_dir / "best_adapter/adapter_model.safetensors").exists():
                            raise RuntimeError(f"Full-budget adapter missing: {run_dir}")
                        tune.report(final_report)
                        reported += 1
                    if reported == 0:
                        raise RuntimeError(f"No evaluation reported; see {run_dir / 'train.log'}")
                    break
                time.sleep(2)
        finally:
            stop_child(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/workspace/pangram-data")))
    parser.add_argument("--mode", choices=["hpo", "ablation"], default="hpo")
    parser.add_argument("--search", choices=["random", "hebo"], default="random")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=25600)
    parser.add_argument("--eval-examples", type=int, default=3200)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--best-config", type=Path, help="JSON with selected HPO parameters for ablation")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    if args.mode == "ablation":
        if args.best_config is None:
            parser.error("--best-config is required for dataset ablations")
        chosen = json.loads(args.best_config.read_text())
        parameters = {key: chosen[key] for key in
                      ("learning_rate", "gradient_accumulation_steps", "lora_rank", "lora_dropout")}
        search_space = {**parameters, "dataset": tune.grid_search(ABLATIONS)}
        trials = 1
        scheduler = None
        max_examples = 12800
        eval_examples = 3200
    else:
        search_space = {"learning_rate": tune.loguniform(1.5e-5, 1.5e-4),
                        "gradient_accumulation_steps": tune.choice([4, 8, 16]),
                        "lora_rank": tune.choice([8, 16, 32]),
                        "lora_dropout": tune.uniform(0, .15)}
        trials = args.trials
        max_examples = args.max_examples
        eval_examples = args.eval_examples
        scheduler = ASHAScheduler(time_attr="examples_seen",
                                  max_t=max_examples, grace_period=2 * eval_examples,
                                  reduction_factor=2)
    sweep = args.name or f"{args.mode}_diverse_v1"
    ray.init()
    trainable = tune.with_resources(
        tune.with_parameters(train_trial, root=str(args.root.resolve()), sweep=sweep,
                             max_examples=max_examples, eval_examples=eval_examples,
                             report_to=args.report_to),
        resources={"cpu": 4, "gpu": 1},
    )
    search_alg = None
    if args.mode == "hpo" and args.search == "hebo":
        from ray.tune.search.hebo import HEBOSearch
        search_alg = HEBOSearch(metric="score", mode="max", random_state_seed=42)
    results = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=trials, metric="score", mode="max",
                                    scheduler=scheduler, search_alg=search_alg),
        run_config=tune.RunConfig(name=sweep,
                                  storage_path=str((args.root / "ray_results").resolve())),
    ).fit()
    summary = []
    for result in results:
        summary.append({"config": result.config, "metrics": result.metrics,
                        "error": str(result.error) if result.error else None,
                        "path": result.path})
    out = args.root / "runs" / f"{sweep}_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(out)


if __name__ == "__main__":
    main()
