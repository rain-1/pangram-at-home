"""Run the frozen token-pilot comparisons serially on one GPU."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]
RUNS = [("qwen3_token_single_v3_pilot1", "token"),
        ("vast_hpo_selected_v3", "sequence"),
        ("qwen3_token_repeat2_v3_pilot1", "token")]


def evaluate(name: str, task: str, dataset: str, data_file: str,
             output_name: str, threshold: float | None = None):
    run = ROOT / "runs" / name
    target = run / f"{output_name}.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if "by_construction_family" in previous:
            print("cached", name, output_name, flush=True)
            return previous
    command = [sys.executable, "-u", str(REPO / "scripts/evaluate_span_pilot.py"),
               "--run-name", name, "--task", task, "--dataset-folder", dataset,
               "--validation-file", data_file, "--output-name", output_name]
    if threshold is not None:
        command += ["--threshold", repr(threshold)]
    log = run / f"{output_name}.log"
    print("scoring", name, output_name, flush=True)
    with log.open("w") as file:
        subprocess.run(command, cwd=REPO, stdout=file, stderr=subprocess.STDOUT, check=True)
    result = json.loads(target.read_text())
    print("done", name, output_name,
          "fpr", round(result["overall"]["fpr"], 4),
          "recall", round(result["overall"]["ai_recall"], 4), flush=True)
    return result


def main():
    for name, task in RUNS:
        val = evaluate(name, task, "span_pilot_v3", "val.jsonl", "span_validation_v3")
        threshold = val["threshold"]
        evaluate(name, task, "span_test_probe_v1", "test.jsonl", "span_test_v1", threshold)
        evaluate(name, task, "span_long_probe_v1", "val.jsonl", "span_long_v1", threshold)


if __name__ == "__main__":
    main()
