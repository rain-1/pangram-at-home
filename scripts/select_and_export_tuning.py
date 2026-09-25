"""Select full-budget validation winner and export model artifacts, no raw text."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/pangram-data"))
    parser.add_argument("--hpo-name", default="hpo_diverse_v1")
    parser.add_argument("--ablation-name", default="ablation_diverse_v1")
    parser.add_argument("--full-examples", type=int, default=25600)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    summary_path = args.root / "runs" / f"{args.hpo_name}_summary.json"
    trials = json.loads(summary_path.read_text())
    successful = [t for t in trials if t["error"] is None and t["metrics"].get("score") is not None]
    full = [t for t in successful if t["metrics"].get("examples_seen", 0) >= args.full_examples]
    if not full:
        raise SystemExit("No full-budget HPO trial completed; cannot select a winner")
    ranked = sorted(full, key=lambda t: t["metrics"]["score"], reverse=True)
    best = ranked[0]
    keys = ("learning_rate", "gradient_accumulation_steps", "lora_rank", "lora_dropout")
    config = {key: best["config"][key] for key in keys}
    config.update({"validation_score": best["metrics"]["score"],
                   "validation_partial_auc_fpr_5pct": best["metrics"]["val_partial_auc_fpr_5pct"],
                   "validation_ai_recall_at_fpr_2pct": best["metrics"]["val_ai_recall_at_fpr_2pct"],
                   "source_run": best["metrics"]["run_name"]})
    config_path = args.root / "runs" / f"{args.hpo_name}_best_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("Full-budget trials:", len(full), "of", len(trials))
    print("Selected run:", config["source_run"], "score", config["validation_score"])
    if args.select_only:
        return
    export = args.root / "hpo_export_v1.tar.gz"
    with tarfile.open(export, "w:gz") as tar:
        for path in (summary_path, config_path,
                     args.root / "runs" / f"{args.ablation_name}_summary.json"):
            if path.exists():
                tar.add(path, arcname=f"results/{path.name}")
        for trial in ranked[:3]:
            run_name = trial["metrics"]["run_name"]
            run = args.root / "runs" / run_name
            for name in ("best_adapter", "run_config.json", "train_summary.json", "sweep_metrics.jsonl"):
                path = run / name
                if path.exists():
                    tar.add(path, arcname=f"runs/{run_name}/{name}")
    print("Exported", export, export.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
