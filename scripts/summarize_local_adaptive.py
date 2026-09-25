"""Summarize the autonomous local queue using only completed artifacts."""
import json
from pathlib import Path


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]


def optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def main():
    status = optional(ROOT / "adaptive_local_v1/status.json") or {}
    selected = status.get("selected_for_holdout")
    names = ["vast_hpo_selected_v3", "qwen3_hpo_single_local_v1",
             "qwen3_hpo_repeat2_local_v1", status.get("decision", {}).get("short_window_run")]
    names = [name for name in names if name]
    runs = {}
    for name in names:
        directory = ROOT / "runs" / name
        runs[name] = {"train": optional(directory / "train_summary.json"),
                      "validation": optional(directory / "validation_diagnostics.json"),
                      "windows": optional(directory / "span_validation.json"),
                      "holdout": optional(REPO / "reports/metrics" / f"{name}.json")}
    result = {"status": status, "selected_local": selected, "runs": runs}
    output = REPO / "reports"
    (output / "local_adaptive_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Local 4080 adaptive run", "", f"Queue state: **{status.get('state', 'unknown')}**.",
             "Choices used development validation only. Held-out sets were scored after the choice was made.", "",
             "| Run | Steps | Validation AUROC | Validation FPR | Validation AI recall | Window AI recall | Human false-highlight rate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in runs.items():
        val = (item["validation"] or {}).get("overall", {})
        win = item["windows"] or {}
        lines.append(f"| {name} | {(item['train'] or {}).get('global_step', '—')} | "
                     f"{pct(val.get('roc_auc'))} | {pct(val.get('fpr'))} | {pct(val.get('tpr'))} | "
                     f"{pct(win.get('overall', {}).get('ai_recall'))} | "
                     f"{pct(win.get('human_character_false_highlight_rate'))} |")
    lines += ["", "The window scores use synthetic joins and a threshold calibrated on that same development set.",
              "They are localization diagnostics, not a measured real-world span error rate.", ""]
    decision = status.get("decision")
    if decision:
        lines += [f"Short-window pilot chosen: **{decision['short_window_run']}**.",
                  "Its 1,600-step result is exploratory and has a smaller training budget than the full runs.", ""]
    for name, item in runs.items():
        report = item.get("holdout")
        if not report:
            continue
        lines += [f"## Held-out evaluation: {name}", "",
                  "| Set | Rows | AUROC | FPR | AI recall |", "| --- | ---: | ---: | ---: | ---: |"]
        for split, values in report["splits"].items():
            lines.append(f"| {split} | {values.get('rows', '—')} | {pct(values.get('roc_auc'))} | "
                         f"{pct(values.get('fpr'))} | {pct(values.get('tpr'))} |")
        lines += [""]
    (output / "local_adaptive_v1.md").write_text("\n".join(lines) + "\n")
    print(output / "local_adaptive_v1.md")


if __name__ == "__main__":
    main()
