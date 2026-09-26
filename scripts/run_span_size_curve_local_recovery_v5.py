"""Finish the interrupted 20k and compute-matched 5k trials on one local GPU."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from dotenv import load_dotenv


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home")
RUNS = (
    ("qwen3_token_repeat2_v5_20k_e1_local", 20000, 3269),
    ("qwen3_token_repeat2_v5_5k_e4_local", 5000, 3269),
)
EVALS = (
    ("span_size_curve_v5/size_20000", "test_llmtrace.jsonl", "v5_llmtrace_heldout"),
    ("span_training_v4", "val.jsonl", "v5_prior_synthetic_val"),
    ("span_human_eval_v2", "test.jsonl", "v5_human_locked_test"),
    ("span_sources_v5/normalized_aitdna_real", "locked_test.jsonl", "v5_aitdna"),
    ("span_realistic_eval_v1", "test.jsonl", "v5_coauthor"),
)


def save(status: dict) -> None:
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = ROOT / "span_size_curve_v5_local_status.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(temp, path)


def command(args: list[str], log: Path) -> None:
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false",
               WANDB_PROJECT="pangram-at-home", WANDB_ENTITY="eac-adsf",
               WANDB_DIR=str(ROOT / "wandb"), WANDB_LOG_MODEL="false")
    (ROOT / "wandb").mkdir(exist_ok=True)
    with log.open("w") as file:
        subprocess.run(args, cwd=REPO, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)


def run(name: str, size: int, steps: int, status: dict) -> None:
    folder = f"span_size_curve_v5/size_{size}"
    status["runs"][name] = "training"
    save(status)
    train = [sys.executable, "-u", str(REPO / "scripts/train_token_lora.py"),
             "--root", str(ROOT), "--model", str(ROOT / "models/Qwen3-1.7B"),
             "--init-adapter", str(ROOT / "runs/vast_hpo_selected_v3/best_adapter"),
             "--run-name", name, "--dataset-folder", folder,
             "--max-steps", str(steps), "--eval-steps", str(steps // 2),
             "--learning-rate", "7.607757094022466e-5", "--lora-rank", "32",
             "--lora-alpha", "64", "--lora-dropout", ".068837366330751",
             "--batch-size", "2", "--accumulation", "4", "--hours", "5",
             "--report-to", "wandb", "--seed", "42"]
    command(train, ROOT / f"{name}.train.log")
    output = ROOT / "runs" / name
    summary = json.loads((output / "train_summary.json").read_text())
    if summary["global_step"] != steps:
        raise RuntimeError(f"{name} ended at {summary['global_step']}/{steps}")
    status["runs"][name] = "evaluating"
    save(status)
    threshold = None
    for dataset, filename, report_name in (
        ("span_human_eval_v2", "calibration.jsonl", "v5_human_calibration"), *EVALS
    ):
        args = [sys.executable, "-u", str(REPO / "scripts/evaluate_span_pilot.py"),
                "--root", str(ROOT), "--run-name", name, "--task", "token",
                "--dataset-folder", dataset, "--validation-file", filename,
                "--output-name", report_name, "--report-to", "wandb"]
        if threshold is None:
            args += ["--calibration-unit", "document", "--target-fpr", ".05"]
        else:
            args += ["--threshold", repr(threshold)]
        command(args, output / f"{report_name}.log")
        report = json.loads((output / f"{report_name}.json").read_text())
        if threshold is None:
            threshold = report["threshold"]
        elif report["threshold"] != threshold:
            raise RuntimeError("Frozen threshold changed")
        status["last_evaluation"] = {"run": name, "set": report_name}
        save(status)
    status["runs"][name] = "complete"
    save(status)


def main() -> None:
    load_dotenv(REPO / ".env", override=False)
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY missing")
    status = {"phase": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs": {}}
    save(status)
    try:
        for name, size, steps in RUNS:
            run(name, size, steps, status)
        status["phase"] = "complete"
    except Exception:
        status["phase"] = "failed"
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        save(status)
    if status["phase"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
