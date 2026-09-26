"""Durable four-GPU size-curve runner with frozen evaluations and export."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import traceback


ROOT = Path("/workspace/pangram-data")
REPO = Path("/workspace/pangram-at-home")
RUNS = (
    ("qwen3_token_repeat2_v5_5k_e1", 5000, 816, 0),
    ("qwen3_token_repeat2_v5_10k_e1", 10000, 1634, 1),
    ("qwen3_token_repeat2_v5_20k_e1", 20000, 3269, 2),
    ("qwen3_token_repeat2_v5_5k_e4", 5000, 3269, 3),
)
EVALS = (
    ("span_size_curve_v5/size_20000", "test_llmtrace.jsonl", "v5_llmtrace_heldout"),
    ("span_training_v4", "val.jsonl", "v5_prior_synthetic_val"),
    ("span_human_eval_v2", "test.jsonl", "v5_human_locked_test"),
    ("span_sources_v5/normalized_aitdna_real", "locked_test.jsonl", "v5_aitdna"),
    ("span_realistic_eval_v1", "test.jsonl", "v5_coauthor"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2)
        file.write("\n")
        temp = Path(file.name)
    os.replace(temp, path)


def command(args: list[str], log: Path, gpu: int) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
    with log.open("w") as file:
        subprocess.run(args, cwd=REPO, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)


def worker(run_name: str, size: int, steps: int, gpu: int) -> None:
    folder = f"span_size_curve_v5/size_{size}"
    train = [sys.executable, "-u", str(REPO / "scripts/train_token_lora.py"),
             "--root", str(ROOT), "--model", str(ROOT / "models/Qwen3-1.7B"),
             "--init-adapter", str(ROOT / "runs/vast_hpo_selected_v3/best_adapter"),
             "--run-name", run_name, "--dataset-folder", folder,
             "--max-steps", str(steps), "--eval-steps", str(steps // 2),
             "--learning-rate", "7.607757094022466e-5", "--lora-rank", "32",
             "--lora-alpha", "64", "--lora-dropout", ".068837366330751",
             "--batch-size", "2", "--accumulation", "4", "--hours", "5",
             "--report-to", "wandb", "--seed", "42"]
    log = ROOT / f"{run_name}.train.log"
    command(train, log, gpu)
    run = ROOT / "runs" / run_name
    summary = json.loads((run / "train_summary.json").read_text())
    if summary["global_step"] != steps:
        raise RuntimeError(f"{run_name} stopped at {summary['global_step']}/{steps}")
    threshold = None
    for dataset, filename, output in (
        ("span_human_eval_v2", "calibration.jsonl", "v5_human_calibration"), *EVALS
    ):
        args = [sys.executable, "-u", str(REPO / "scripts/evaluate_span_pilot.py"),
                "--root", str(ROOT), "--run-name", run_name, "--task", "token",
                "--dataset-folder", dataset, "--validation-file", filename,
                "--output-name", output, "--report-to", "wandb"]
        if threshold is None:
            args += ["--calibration-unit", "document", "--target-fpr", ".05"]
        else:
            args += ["--threshold", repr(threshold)]
        command(args, run / f"{output}.log", gpu)
        report = json.loads((run / f"{output}.json").read_text())
        if threshold is None:
            threshold = report["threshold"]
        elif report["threshold"] != threshold:
            raise RuntimeError(f"{run_name} evaluation threshold changed")
    print(run_name, "complete", flush=True)


def export(status: dict) -> dict:
    included = []
    for run_name, _, _, _ in RUNS:
        run = ROOT / "runs" / run_name
        if not run.exists():
            continue
        included.extend(path for path in run.iterdir() if path.is_file() and
                        path.suffix in (".json", ".jsonl", ".npz", ".log"))
        adapter = run / "best_adapter"
        if adapter.exists():
            included.extend(path for path in adapter.rglob("*") if path.is_file())
        top_log = ROOT / f"{run_name}.train.log"
        if top_log.exists():
            included.append(top_log)
    snapshot = ROOT / "span_size_curve_v5_status_snapshot.json"
    atomic_json(snapshot, status)
    for name in ("span_size_curve_v5_bootstrap.log", "span_size_curve_v5_status_snapshot.json"):
        path = ROOT / name
        if path.exists():
            included.append(path)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {str(path.relative_to(ROOT)): {"sha256": digest(path),
                                                       "bytes": path.stat().st_size}
                          for path in included}}
    manifest_path = ROOT / "span_size_curve_v5_export_manifest.json"
    atomic_json(manifest_path, manifest)
    archive = ROOT / "span_size_curve_v5_export.tar.gz"
    temp = Path(str(archive) + ".tmp")
    with tarfile.open(temp, "w:gz") as tar:
        tar.add(manifest_path, arcname=manifest_path.name)
        for path in included:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    os.replace(temp, archive)
    return {"archive": str(archive), "sha256": digest(archive),
            "bytes": archive.stat().st_size, "files": len(included),
            "complete_runs": sum(status["runs"].get(name) == "complete" for name, _, _, _ in RUNS)}


def main():
    status_path = ROOT / "span_size_curve_v5_status.json"
    status = {"phase": "preflight", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs": {}, "gpu_count": 4}
    atomic_json(status_path, status)
    processes = []
    try:
        required = [ROOT / "models/Qwen3-1.7B/config.json",
                    ROOT / "runs/vast_hpo_selected_v3/best_adapter/adapter_model.safetensors",
                    ROOT / "data/span_human_eval_v2/calibration.jsonl",
                    ROOT / "data/span_human_eval_v2/test.jsonl"]
        required += [ROOT / f"data/span_size_curve_v5/size_{size}/train.jsonl"
                     for _, size, _, _ in RUNS]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError([str(path) for path in required if not path.is_file()])
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY missing")
        for run_name, size, steps, gpu in RUNS:
            log = ROOT / f"{run_name}.worker.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
            with log.open("w") as file:
                process = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts/run_span_size_curve_v5.py"),
                                            "--worker", run_name], cwd=REPO, env=env,
                                           stdout=file, stderr=subprocess.STDOUT)
            processes.append((run_name, process, log))
            status["runs"][run_name] = "running"
        status["phase"] = "training_and_evaluation"
        atomic_json(status_path, status)
        for run_name, process, log in processes:
            code = process.wait()
            status["runs"][run_name] = "complete" if code == 0 else f"failed_exit_{code}"
            status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(status_path, status)
            print(run_name, status["runs"][run_name], "log", log, flush=True)
        status["phase"] = "complete" if all(value == "complete" for value in status["runs"].values()) else "partial_failure"
    except Exception:
        status["phase"] = "failed"
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        atomic_json(status_path, status)
        try:
            status["export"] = export(status)
        except Exception:
            status["export_error"] = traceback.format_exc()
        atomic_json(status_path, status)
    if status["phase"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_name = sys.argv[sys.argv.index("--worker") + 1]
        details = next((item for item in RUNS if item[0] == run_name), None)
        if details is None:
            raise SystemExit(f"Unknown worker {run_name}")
        worker(*details)
    else:
        main()
