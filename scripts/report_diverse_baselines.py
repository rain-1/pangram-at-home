"""Write compact baseline tables from frozen evaluation JSON results."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home/results")
NAMES = {"char": "Character TF-IDF", "word": "Word TF-IDF", "embedding": "MiniLM + logistic",
         "load_bearing": "Load Bearing PR cluster"}
AUDITS = {"standard_ebooks_human": "Classic fiction", "persuade_essays_human": "Student essays",
          "federal_reserve_human": "Federal Reserve", "stackexchange_writers_human": "Writers Stack Exchange",
          "pmc_full_body_human": "PMC full-body prose"}


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def auc(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    reports = {name: json.loads((ROOT / f"baseline_diverse_{name}_full.json").read_text())
               for name in ("char", "word", "embedding")}
    reports["load_bearing"] = json.loads((REPO / "reports/metrics/baseline_diverse_load_bearing_full.json").read_text())
    lines = ["# Diverse baseline results", "",
             "All thresholds were chosen on the 800-row diverse validation split to allow at most 2% human false positives.",
             "The 1,000-row mixed test contains six writing categories. RAID is an independent source-family test (1,600 rows, eight domains, 11 generators).",
             "", "| Baseline | Mixed AUROC | Mixed FPR | Mixed AI recall | RAID AUROC | RAID FPR | RAID AI recall | Enron AUROC |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, report in reports.items():
        mixed, raid, enron = report["test"], report["raid_external"], report["enron_external"]
        lines.append(f"| {NAMES[name]} | {auc(mixed['roc_auc'])} | {pct(mixed['fpr'])} | {pct(mixed['tpr'])} | "
                     f"{auc(raid['roc_auc'])} | {pct(raid['fpr'])} | {pct(raid['tpr'])} | {auc(enron['roc_auc'])} |")
    lines.extend(["", "Human-only false-positive audits:", "",
                  "| Audit | Rows | Character | Word | MiniLM |", "| --- | ---: | ---: | ---: | ---: |"])
    for key, label in AUDITS.items():
        rows = reports["char"].get(key, {}).get("rows", "—")
        values = [pct(reports[name].get(key, {}).get("fpr")) for name in ("char", "word", "embedding")]
        lines.append(f"| {label} | {rows} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", "The Load Bearing score is a vocabulary-cluster transfer probe, not a purpose-trained AI-authorship detector.",
                  "EditLens reference checkpoints and Qwen are scored after the GPU training run.",
                  "See [the PDF](diverse_baselines_v1.pdf) for per-domain AUROC and AI recall.", ""])
    out = REPO / "reports/diverse-baseline-results-v1.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
