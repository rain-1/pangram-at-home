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
                      "holdout": optional(REPO / "reports/metrics" / f"{name}.json"),
                      "fullpaper": optional(REPO / "reports/metrics" / f"{name}_pmc_fullpaper.json")}
    result = {"status": status, "selected_local": selected, "runs": runs}
    output = REPO / "reports"
    (output / "local_adaptive_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Local 4080 adaptive run", "", f"Queue state: **{status.get('state', 'unknown')}**.",
             "Choices used development validation only. Held-out sets were scored after the choice was made.", "",
             "| Run | Steps | Validation pAUC (≤5% FPR) | Validation FPR | Validation AI recall | Window AI recall | Human false-highlight rate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in runs.items():
        val = ((item["holdout"] or {}).get("splits", {}).get("val") or
               (item["validation"] or {}).get("overall", {}))
        win = item["windows"] or {}
        lines.append(f"| {name} | {(item['train'] or {}).get('global_step', '—')} | "
                     f"{pct(val.get('partial_auc_fpr_5pct'))} | {pct(val.get('fpr'))} | {pct(val.get('tpr'))} | "
                     f"{pct(win.get('overall', {}).get('ai_recall'))} | "
                     f"{pct(win.get('human_character_false_highlight_rate'))} |")
    lines += ["", "The window scores use synthetic joins and a threshold calibrated on that same development set.",
              "They are localization diagnostics, not a measured real-world span error rate.", ""]
    lines += ["Validation operating points were recalibrated after correcting float32 threshold rounding.",
              "The saved training-time AI recall values used one extra human validation example at the threshold;",
              "partial AUROC and the selected checkpoints were unaffected.", ""]
    decision = status.get("decision")
    if decision:
        lines += [f"Short-window pilot chosen: **{decision['short_window_run']}**.",
                  "Its 1,600-step result is exploratory and has a smaller training budget than the full runs.", ""]
    for name, item in runs.items():
        report = item.get("holdout")
        if not report:
            continue
        lines += [f"## Held-out evaluation: {name}", "",
                  "| Set | Rows | pAUC (≤5% FPR) | AUROC | FPR | AI recall |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for split, values in report["splits"].items():
            lines.append(f"| {split} | {values.get('rows', '—')} | "
                         f"{pct(values.get('partial_auc_fpr_5pct'))} | {pct(values.get('roc_auc'))} | "
                         f"{pct(values.get('fpr'))} | {pct(values.get('tpr'))} |")
        lines += [""]
    audited = [(name, item["fullpaper"]) for name, item in runs.items() if item["fullpaper"]]
    if audited:
        lines += ["## Whole-paper human audit", "",
                  "Held-out pre-2023 PMC bodies, scored in overlapping windows with",
                  "the passage-calibrated threshold. The same documents are used for each model.", "",
                  "| Run | Papers | Tokens | False-highlight tokens (2% val) | False-highlight tokens (0.5% val) | Papers with any false highlight (2% val) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, audit in audited:
            strict = audit.get("operating_points", {}).get("val_fpr_le_0.005", {})
            lines.append(f"| {name} | {audit['documents']} | {audit['tokens']} | "
                         f"{pct(audit['token_false_highlight_rate'])} | "
                         f"{pct(strict.get('token_false_highlight_rate'))} | "
                         f"{audit['documents_with_any_false_highlight']} |")
        overlap = audited[0][1].get("chunk_audit_overlap_ids")
        lines += ["", f"The original PMC body chunk audit shares paper IDs with the diverse splits "
                  f"({overlap} of {audited[0][1].get('chunk_audit_ids')} paper IDs). "
                  "Its broad-evaluation FPR is therefore not fully document-independent. "
                  "This whole-paper audit excludes every overlapping ID.", ""]
    (output / "local_adaptive_v1.md").write_text("\n".join(lines) + "\n")
    print(output / "local_adaptive_v1.md")


if __name__ == "__main__":
    main()
