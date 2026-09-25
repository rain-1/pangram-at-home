"""Finish the local comparison, inspect each validation, then spend spare GPU time.

This controller may adopt a trainer that was already launched by the earlier queue.
It uses validation only for experiment choices; broad held-out evaluation runs last.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home")
FIRST = "qwen3_hpo_single_local_v1"
SECOND = "qwen3_hpo_repeat2_local_v1"


def read(path: Path):
    return json.loads(path.read_text())


def atomic_write(path: Path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def latest_validation(name: str):
    path = ROOT / "runs" / name / "sweep_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return max(rows, key=lambda row: row.get("eval_partial_auc_fpr_5pct", -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-pid", type=int, required=True)
    parser.add_argument("--total-hours", type=float, default=11.5)
    args = parser.parse_args()
    original = read(ROOT / "overnight_repeat2_span_v1/status.json")
    deadline = original["started"] + args.total_hours * 3600
    work = ROOT / "adaptive_local_v1"
    work.mkdir(exist_ok=True)
    status_path = work / "status.json"
    status = read(status_path) if status_path.exists() else {"started": original["started"],
        "deadline": deadline, "steps": [], "state": "running"}
    status["deadline"] = deadline

    def save():
        status["updated"] = time.time()
        atomic_write(status_path, status)

    def completed(step):
        return any(entry["step"] == step and entry["state"] == "complete" for entry in status["steps"])

    def run(step, command, reserve_seconds=0):
        if completed(step):
            return
        remaining = deadline - time.time() - reserve_seconds
        if remaining < 300:
            status["steps"].append({"step": step, "state": "skipped", "reason": "time budget"})
            save()
            return
        status["current_step"] = step
        save()
        print("Starting", step, flush=True)
        with (work / f"{step}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                exit_code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                exit_code = -1
        entry = {"step": step, "state": "complete" if exit_code == 0 else "failed",
                 "exit_code": exit_code, "finished": time.time()}
        status["steps"].append(entry)
        save()
        if exit_code:
            print("Stage failed:", step, "see", work / f"{step}.log", flush=True)

    os.environ.update(WANDB_PROJECT="pangram-at-home", WANDB_RUN_GROUP="local_adaptive_v1",
                      WANDB_DIR=str(ROOT / "wandb"), TOKENIZERS_PARALLELISM="false",
                      CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    save()
    try:
        if not completed("single_train"):
            pid = args.adopt_pid
            proc = Path(f"/proc/{pid}")
            if proc.exists() and FIRST not in (proc / "cmdline").read_bytes().decode().split("\0"):
                raise RuntimeError("Adopted PID is not the expected single-copy trainer")
            print("Monitoring existing single-copy trainer", pid, flush=True)
            while proc.exists():
                try:
                    if (proc / "stat").read_text().split(") ", 1)[1].startswith("Z"):
                        break
                except FileNotFoundError:
                    break
                if time.time() >= deadline:
                    raise TimeoutError("Overnight deadline reached")
                time.sleep(15)
            if not (ROOT / "runs" / FIRST / "train_summary.json").exists():
                raise RuntimeError("Single-copy trainer ended without a summary")
            status["steps"].append({"step": "single_train", "state": "complete", "finished": time.time()})
            save()

        common = ["--root", str(ROOT), "--model", str(ROOT / "models/Qwen3-1.7B"),
                  "--dataset-folder", "diverse_pyramid_v1", "--train-tier", "full",
                  "--quantization", "none", "--train-batch-size", "1",
                  "--gradient-accumulation-steps", "8", "--eval-batch-size", "1",
                  "--disable-early-stopping", "--selection-metric", "partial_auc_fpr_5pct",
                  "--report-to", "wandb"]
        best = read(ROOT / "vast_results_v1/live/hpo_diverse_v3_best_config.json")
        common += ["--learning-rate", str(best["learning_rate"]),
                   "--lora-rank", str(best["lora_rank"]),
                   "--lora-alpha", str(2 * best["lora_rank"]),
                   "--lora-dropout", str(best["lora_dropout"])]

        def diagnostics(name):
            if not (ROOT / "runs" / name / "best_adapter/adapter_config.json").exists():
                return
            run(name + "_validation", [sys.executable, "-u", "scripts/evaluate_diverse_lora.py",
                "--root", str(ROOT), "--run-name", name, "--batch-size", "2", "--validation-only"],
                reserve_seconds=3600)
            run(name + "_windows", [sys.executable, "-u", "scripts/evaluate_span_pilot.py",
                "--root", str(ROOT), "--run-name", name, "--task", "sequence", "--report-to", "wandb"],
                reserve_seconds=3600)

        diagnostics(FIRST)
        if not (ROOT / "runs" / SECOND / "train_summary.json").exists():
            remaining = deadline - time.time()
            if remaining < 3 * 3600:
                raise TimeoutError("Too little time to start the full Repeat2 comparison")
            run("repeat2_train", [sys.executable, "-u", "scripts/train_segment_lora.py", *common,
                "--run-name", SECOND, "--repeat2", "--max-steps", "3200", "--eval-steps", "400",
                "--metrics-jsonl", str(ROOT / "runs" / SECOND / "sweep_metrics.jsonl"),
                "--hours", str(max(.5, (remaining - 2 * 3600) / 3600))], reserve_seconds=2 * 3600)
        if (ROOT / "runs" / SECOND / "train_summary.json").exists():
            if not completed("repeat2_train"):
                status["steps"].append({"step": "repeat2_train", "state": "complete", "finished": time.time()})
                save()
            diagnostics(SECOND)

        # A shorter source window probes the granularity needed to mark long papers.
        # Choose its training style from the completed 512-token development results.
        first = latest_validation(FIRST)
        second = latest_validation(SECOND) if (ROOT / "runs" / SECOND / "train_summary.json").exists() else None
        use_repeat2 = bool(second and
            second["eval_partial_auc_fpr_5pct"] >= first["eval_partial_auc_fpr_5pct"] + .002 and
            second["eval_worst_domain_ai_recall_at_fpr_2pct"] >=
                first["eval_worst_domain_ai_recall_at_fpr_2pct"] - .03)
        short = "qwen3_hpo_256_" + ("repeat2" if use_repeat2 else "single") + "_pilot_v1"
        status["decision"] = {"first_best_validation": first, "second_best_validation": second,
                              "short_window_run": short, "reason": "512-token validation pAUC and worst-domain recall"}
        save()
        if deadline - time.time() >= 3 * 3600 and not (ROOT / "runs" / short / "train_summary.json").exists():
            command = [sys.executable, "-u", "scripts/train_segment_lora.py", *common,
                       "--run-name", short, "--max-length", "256", "--max-steps", "1600",
                       "--eval-steps", "400", "--metrics-jsonl",
                       str(ROOT / "runs" / short / "sweep_metrics.jsonl"),
                       "--hours", str(max(.5, (deadline - time.time() - 2 * 3600) / 3600))]
            if use_repeat2:
                command.append("--repeat2")
            run("short_window_train", command, reserve_seconds=2 * 3600)
        if (ROOT / "runs" / short / "train_summary.json").exists():
            diagnostics(short)

        # Held-out sets are touched after all experiment choices have been made.
        # Include the Vast checkpoint and the strongest completed local run.
        candidates = [name for name in (FIRST, SECOND, short)
                      if (ROOT / "runs" / name / "train_summary.json").exists()]
        selected = max(candidates, key=lambda name: latest_validation(name)["eval_partial_auc_fpr_5pct"])
        status["selected_for_holdout"] = selected
        save()
        for name in ("vast_hpo_selected_v3", selected):
            if (ROOT / "runs" / name / "best_adapter/adapter_config.json").exists():
                run(name + "_holdouts", [sys.executable, "-u", "scripts/evaluate_diverse_lora.py",
                    "--root", str(ROOT), "--run-name", name, "--batch-size", "2"])
        run("summarize", [sys.executable, "scripts/summarize_overnight.py", "--root", str(ROOT)])
        status["state"] = "complete"
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        raise
    finally:
        status["elapsed_hours"] = (time.time() - original["started"]) / 3600
        save()
        subprocess.run([sys.executable, "scripts/summarize_local_adaptive.py"], cwd=REPO,
                       stdout=(work / "summary.log").open("w"), stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
