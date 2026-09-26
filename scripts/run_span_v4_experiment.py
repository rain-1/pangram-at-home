"""Durable remote Repeat2 v4 train, calibration, and frozen evaluation queue."""
from __future__ import annotations

import argparse
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


OLD = "qwen3_token_repeat2_v3_pilot1"
NEW = "qwen3_token_repeat2_v4_pilot1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as file:
        json.dump(value, file, indent=2)
        file.write("\n")
        temp = Path(file.name)
    os.replace(temp, path)


def run(command: list[str], log: Path, repo: Path, status: Path, phase: str) -> None:
    current = json.loads(status.read_text())
    current.update({"phase": phase, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "active_log": str(log)})
    atomic_json(status, current)
    print("phase", phase, "log", log, flush=True)
    with log.open("w") as file:
        subprocess.run(command, cwd=repo, stdout=file, stderr=subprocess.STDOUT, check=True)


def evaluate(repo: Path, root: Path, status: Path, model: str, dataset: str,
             filename: str, output: str, threshold: float | None = None,
             *, calibration: bool = False) -> dict:
    command = [sys.executable, "-u", str(repo / "scripts/evaluate_span_pilot.py"),
               "--root", str(root), "--run-name", model, "--task", "token",
               "--dataset-folder", dataset, "--validation-file", filename,
               "--output-name", output, "--report-to", "wandb"]
    if calibration:
        command += ["--calibration-unit", "document", "--target-fpr", ".05"]
    elif threshold is not None:
        command += ["--threshold", repr(threshold)]
    else:
        raise ValueError("A frozen evaluation requires a supplied threshold")
    run(command, root / "runs" / model / f"{output}.log", repo, status,
        f"evaluating:{model}:{output}")
    result = json.loads((root / "runs" / model / f"{output}.json").read_text())
    if not calibration and result["threshold"] != threshold:
        raise RuntimeError(f"Frozen threshold mismatch for {model} {output}")
    return result


def relocate_old_config(root: Path) -> None:
    """Point the transferred local run at the same base model on the remote host."""
    path = root / "runs" / OLD / "run_config.json"
    config = json.loads(path.read_text())
    remote_model = str(root / "models/Qwen3-1.7B")
    existing = config.get("base_model", config.get("model"))
    if existing != remote_model:
        config["relocated_base_model_from"] = existing
        config["base_model"] = remote_model
        atomic_json(path, config)


def export(root: Path, status: Path) -> dict:
    """Package artifacts needed for local comparison; omit datasets and optimizer states."""
    paths: list[Path] = []
    names = ("run_config.json", "train_summary.json", "sweep_metrics.jsonl",
             "trainer_state.json", "train.log")
    suffixes = (".json", ".jsonl", ".npz", ".log")
    for model in (OLD, NEW):
        run_dir = root / "runs" / model
        if not run_dir.exists():
            continue
        for path in run_dir.iterdir():
            if path.is_file() and (path.name in names or
                                   (path.name.startswith(("span_", "v4_")) and
                                    path.suffix in suffixes)):
                paths.append(path)
        adapter = run_dir / "best_adapter"
        if adapter.exists():
            paths.extend(path for path in adapter.rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError("No run artifacts to export")
    if (root / "span_v4_train.log").exists():
        paths.append(root / "span_v4_train.log")
    if (root / "span_v4_bootstrap.log").exists():
        paths.append(root / "span_v4_bootstrap.log")
    snapshot = root / "span_v4_status_snapshot.json"
    atomic_json(snapshot, json.loads(status.read_text()))
    paths.append(snapshot)
    report_folder = root / "reports/span_v4"
    if report_folder.exists():
        paths.extend(path for path in report_folder.iterdir() if path.is_file()
                     and path.suffix in (".md", ".json", ".pdf", ".log"))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
                          for path in paths}}
    manifest_path = root / "span_v4_export_manifest.json"
    atomic_json(manifest_path, manifest)
    archive = root / "span_v4_export.tar.gz"
    temp_archive = root / "span_v4_export.tar.gz.tmp"
    with tarfile.open(temp_archive, "w:gz") as tar:
        tar.add(manifest_path, arcname=manifest_path.name)
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(root)))
    os.replace(temp_archive, archive)
    return {"archive": str(archive), "archive_sha256": digest(archive),
            "files": len(paths), "bytes": archive.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/pangram-data"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/pangram-at-home"))
    parser.add_argument("--max-steps", type=int, default=950)
    parser.add_argument("--eval-steps", type=int, default=190)
    parser.add_argument("--skip-train", action="store_true", help="Reuse an already completed v4 adapter after an evaluation failure")
    args = parser.parse_args()
    root, repo = args.root, args.repo
    status = root / "span_v4_status.json"
    atomic_json(status, {"phase": "preflight", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                         "old_run": OLD, "new_run": NEW})
    success = False
    try:
        required = [repo / "scripts/train_token_lora.py", repo / "scripts/evaluate_span_pilot.py",
                    root / "data/span_training_v4/train.jsonl",
                    root / "data/span_training_v4/val.jsonl",
                    root / "data/span_human_eval_v2/calibration.jsonl",
                    root / "data/span_human_eval_v2/test.jsonl",
                    root / "data/span_pilot_v3/val.jsonl",
                    root / "data/span_realistic_eval_v1/test.jsonl",
                    root / "runs" / OLD / "best_adapter/adapter_model.safetensors",
                    root / "runs/vast_hpo_selected_v3/best_adapter/adapter_model.safetensors",
                    root / "models/Qwen3-1.7B/config.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing prerequisites: " + ", ".join(missing))
        relocate_old_config(root)
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY unavailable to remote process")
        train = [sys.executable, "-u", str(repo / "scripts/train_token_lora.py"),
                 "--root", str(root), "--model", str(root / "models/Qwen3-1.7B"),
                 "--init-adapter", str(root / "runs/vast_hpo_selected_v3/best_adapter"),
                 "--run-name", NEW, "--dataset-folder", "span_training_v4",
                 "--max-steps", str(args.max_steps), "--eval-steps", str(args.eval_steps),
                 "--learning-rate", "7.607757094022466e-5", "--lora-rank", "32",
                 "--lora-alpha", "64", "--lora-dropout", ".068837366330751",
                 "--batch-size", "2", "--accumulation", "4", "--hours", "3",
                 "--report-to", "wandb", "--seed", "42"]
        if args.skip_train:
            if not (root / "runs" / NEW / "best_adapter/adapter_model.safetensors").exists():
                raise FileNotFoundError("--skip-train requires a finished v4 adapter")
        else:
            run(train, root / "span_v4_train.log", repo, status, "training")
            summary = json.loads((root / "runs" / NEW / "train_summary.json").read_text())
            if summary["global_step"] != args.max_steps:
                raise RuntimeError(f"Training stopped at step {summary['global_step']} of {args.max_steps}")
        for model in (OLD, NEW):
            calibration = evaluate(repo, root, status, model, "span_human_eval_v2",
                                   "calibration.jsonl", "v4_human_calibration", calibration=True)
            threshold = calibration["threshold"]
            current = json.loads(status.read_text())
            current.setdefault("calibrated_thresholds", {})[model] = threshold
            atomic_json(status, current)
            evaluate(repo, root, status, model, "span_training_v4", "val.jsonl",
                     "v4_synthetic_val", threshold)
            evaluate(repo, root, status, model, "span_pilot_v3", "val.jsonl",
                     "v4_prior_synthetic_val", threshold)
            evaluate(repo, root, status, model, "span_human_eval_v2", "test.jsonl",
                     "v4_human_locked_test", threshold)
            evaluate(repo, root, status, model, "span_realistic_eval_v1", "test.jsonl",
                     "v4_realistic_locked_test", threshold)
        success = True
    except Exception as error:
        current = json.loads(status.read_text())
        current.update({"phase": "failing", "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(status, current)
        print(current["traceback"], file=sys.stderr, flush=True)
    finally:
        try:
            current = json.loads(status.read_text())
            current["phase"] = "exporting" if success else "failed_exporting"
            atomic_json(status, current)
            export_info = export(root, status)
            current = json.loads(status.read_text())
            current["export"] = export_info
            current["phase"] = "complete" if success else "failed"
            current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(status, current)
        except Exception:
            traceback.print_exc()
            success = False
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
