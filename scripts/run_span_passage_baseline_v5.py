"""Score the frozen Vast passage model on the same span benchmark sets."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]
MODEL = "vast_hpo_selected_v3"
SETS = (
    ("span_human_eval_v2", "calibration.jsonl", "span_v5_human_calibration"),
    ("span_training_v4", "val.jsonl", "span_v5_synthetic_v4_val"),
    ("span_pilot_v3", "val.jsonl", "span_v5_prior_synthetic_val"),
    ("span_human_eval_v2", "test.jsonl", "span_v5_human_locked_test"),
    ("span_realistic_eval_v1", "test.jsonl", "span_v5_coauthor"),
    ("span_sources_v5/normalized_aitdna_real", "locked_test.jsonl", "span_v5_aitdna"),
)


def main():
    run = ROOT / "runs" / MODEL
    threshold = None
    for dataset, filename, output in SETS:
        command = [sys.executable, str(REPO / "scripts/evaluate_span_pilot.py"),
                   "--root", str(ROOT), "--run-name", MODEL, "--task", "sequence",
                   "--dataset-folder", dataset, "--validation-file", filename,
                   "--output-name", output]
        if threshold is None:
            command += ["--calibration-unit", "document", "--target-fpr", ".05"]
        else:
            command += ["--threshold", repr(threshold)]
        with (run / f"{output}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / f"{output}.json").read_text())
        if threshold is None:
            threshold = result["threshold"]
        elif result["threshold"] != threshold:
            raise RuntimeError("Frozen threshold changed")
        print(output, result["overall"]["ai_recall"], result["overall"]["fpr"], flush=True)


if __name__ == "__main__":
    main()
