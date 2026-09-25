"""Summarize the paired passage experiment and separate span pilot."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    args=p.parse_args();out=Path(__file__).resolve().parents[1]/"reports"
    rows=[]
    for name in ("qwen3_hpo_single_local_v1","qwen3_hpo_repeat2_local_v1","qwen3_token_repeat2_pilot_v1"):
        run=args.root/"runs"/name
        if not (run/"train_summary.json").exists():continue
        summary=json.loads((run/"train_summary.json").read_text());config=json.loads((run/"run_config.json").read_text())
        metrics_path=run/"sweep_metrics.jsonl"
        history=[json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        best_step=int(summary["best_checkpoint"].rsplit("-",1)[1]) if summary["best_checkpoint"] else None
        best=next((x for x in history if x["step"]==best_step),{})
        span=json.loads((run/"span_validation.json").read_text()) if (run/"span_validation.json").exists() else None
        rows.append({"run":name,"config":config,"training":summary,"best_checkpoint_validation":best,"span_validation":span})
    (out/"overnight_repeat2_span_v1.json").write_text(json.dumps(rows,indent=2)+"\n")
    hpo=json.loads((args.root/"vast_results_v1/live/hpo_diverse_v3_best_config.json").read_text())
    vast_row=f'| Vast selected model (existing) | 3200 | 3200 | {hpo["validation_partial_auc_fpr_5pct"]:.5f} | remote | remote |\n'
    token_trained=any(r["config"].get("task")=="binary_token_classification" for r in rows)
    text="# Vast winner and local Repeat2 comparison\n\nValidation-only development results. No blind test was used.\n\n"
    if not token_trained:
        text+="Token training was deferred pending realistic, reviewed span data. Highlighting uses coarse overlapping-window classification scores.\n\n"
    text+="| Run | Steps | Best step | Validation pAUC | Training hours | Peak allocated GB |\n| --- | --- | --- | --- | --- | --- |\n"
    text+=vast_row
    for r in rows:
        m=r["best_checkpoint_validation"];s=r["training"]
        text+=f'| {r["run"]} | {s["global_step"]} | {m.get("step")} | {m.get("eval_partial_auc_fpr_5pct",float("nan")):.5f} | {s.get("train_runtime_seconds",0)/3600:.2f} | {s.get("peak_allocated_gb",0):.2f} |\n'
    text+="\nThe saved Vast model is the practical benchmark for improvement. The two fresh local runs isolate Repeat2: compare their passage metrics only if both completed the requested 3,200 steps. Vast used microbatch 2 × accumulation 4; local runs use 1 × 8, with the same effective batch. The token pilot, if enabled, uses a different synthetic task, so its token pAUC is not directly comparable.\n"
    text+="\nSynthetic joins and inherited source labels limit realism. An AI-assisted class is not trained. HPO settings and the selected adapter are starting points; no claim is made that they are optimal for token training.\n"
    text+="\nWhole-document span diagnostics (including the passage classifiers as coarse sliding-window baselines) are in the JSON report. Their thresholds are calibrated on the same development controls, so they do not estimate blind-test FPR.\n"
    (out/"overnight_repeat2_span_v1.md").write_text(text)
    print(out/"overnight_repeat2_span_v1.md")


if __name__=="__main__":main()
