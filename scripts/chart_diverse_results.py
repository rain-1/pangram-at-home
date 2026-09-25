"""Render the completed Qwen run beside every available diverse baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home/results")
OUT = REPO / "reports/diverse_full_results_v1.pdf"
ORDER = ["qwen", "char", "word", "embedding", "load_bearing", "roberta", "llama"]
NAMES = {"qwen": "Qwen3-1.7B LoRA", "char": "Character TF-IDF", "word": "Word TF-IDF",
         "embedding": "MiniLM + logistic", "load_bearing": "Load Bearing cluster",
         "roberta": "EditLens RoBERTa", "llama": "EditLens Llama"}
COLORS = {"qwen": "#173F64", "char": "#2778A8", "word": "#E68A44",
          "embedding": "#60A680", "load_bearing": "#B772A4",
          "roberta": "#8A6FC1", "llama": "#D5B33D"}


def load() -> dict:
    result = {}
    result["qwen"] = json.loads((REPO / "reports/metrics/qwen3_17b_diverse_v1.json").read_text())["splits"]
    for name in ("char", "word", "embedding"):
        result[name] = json.loads((ROOT / f"baseline_diverse_{name}_full.json").read_text())
    result["load_bearing"] = json.loads((REPO / "reports/metrics/baseline_diverse_load_bearing_full.json").read_text())
    for name in ("roberta", "llama"):
        path = ROOT / f"reference_diverse_{name}_full.json"
        if path.exists():
            try:
                result[name] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return result


def by_domain(report: dict, split: str) -> dict:
    section = report[split]
    return section.get("by_domain", section.get("by_source", {}))


def grouped_plot(pdf: PdfPages, reports: dict, order: list[str], split: str, domains: list[str], title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), constrained_layout=True)
    x = np.arange(len(domains))
    width = .8 / len(order)
    for i, name in enumerate(order):
        cells = by_domain(reports[name], split)
        auc = [cells[d]["roc_auc"] for d in domains]
        recall = [cells[d]["tpr"] for d in domains]
        axes[0].bar(x + (i - (len(order) - 1) / 2) * width, auc, width, label=NAMES[name], color=COLORS[name])
        axes[1].bar(x + (i - (len(order) - 1) / 2) * width, recall, width, color=COLORS[name])
    for ax in axes:
        ax.set_xticks(x, [d.replace("_", " ").title() for d in domains])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("AUROC by domain", fontsize=15)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=8)
    axes[1].set_title("AI recall at each model's validation-calibrated threshold", fontsize=15)
    axes[1].set_ylabel("AI recall")
    fig.suptitle(title, fontsize=17)
    pdf.savefig(fig)
    plt.close(fig)


def value(report: dict, split: str, field: str) -> str:
    if split not in report or report[split].get(field) is None:
        return "—"
    number = report[split][field]
    return f"{number:.3f}" if field == "roc_auc" else f"{number:.1%}"


def main() -> None:
    reports = load()
    order = [name for name in ORDER if name in reports]
    with PdfPages(OUT) as pdf:
        domains = list(by_domain(reports["qwen"], "test"))
        grouped_plot(pdf, reports, order, "test", domains,
                     f"Diverse test | 1,000 balanced passages | {len(order)} baselines/models")
        domains = list(by_domain(reports["qwen"], "raid_external"))
        grouped_plot(pdf, reports, order, "raid_external", domains,
                     "RAID external | 1,600 balanced passages | eight domains")

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        columns = ["Model", "Mixed AUC", "Mixed FPR", "Mixed recall", "RAID AUC",
                   "RAID recall", "Enron AUC", "GPT-4 AUC", "Paraphrase AUC"]
        rows = []
        for name in order:
            r = reports[name]
            rows.append([NAMES[name], value(r, "test", "roc_auc"), value(r, "test", "fpr"),
                         value(r, "test", "tpr"), value(r, "raid_external", "roc_auc"),
                         value(r, "raid_external", "tpr"), value(r, "enron_external", "roc_auc"),
                         value(r, "gpt4_ood", "roc_auc"), value(r, "paraphrase", "roc_auc")])
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_facecolor("#E8F0F5" if row == 0 else ("#F6F8FA" if row % 2 else "white"))
            cell.set_edgecolor("#D8E0E5")
        ax.set_title("Model comparison across balanced tests", fontsize=18, pad=22)
        ax.text(.5, .15, "Thresholds were selected from the diverse validation split at <=2% human FPR.\n"
                "RAID is a separate source family; MAGE GPT-4/paraphrase shares a source family with training.\n"
                "Load Bearing is a PR vocabulary-cluster model, included as a transfer baseline.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        qwen_path = REPO / "reports/metrics/qwen3_17b_diverse_v1.json"
        points = json.loads(qwen_path.read_text()).get("operating_points", {}) if qwen_path.exists() else {}
        if points:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ordered = sorted(points.items(), key=lambda item: float(item[0].rsplit("_", 1)[-1]))
            for split, label, color in (("test", "Diverse test", COLORS["qwen"]),
                                        ("raid_external", "RAID external", "#C96A45")):
                xs = [point[split]["fpr"] * 100 for _, point in ordered]
                ys = [point[split]["tpr"] * 100 for _, point in ordered]
                ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
                for (name, _), x_value, y_value in zip(ordered, xs, ys):
                    ax.annotate(f"val ≤{float(name.rsplit('_', 1)[-1]):.1%}",
                                (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax.set_xlabel("Observed human false-positive rate (%)")
            ax.set_ylabel("AI recall (%)")
            ax.set_title("Qwen decision threshold trade-off", fontsize=17)
            ax.grid(alpha=.25)
            ax.legend()
            fig.text(.5, .06, "Thresholds set on validation only; points show transfer to each test set.",
                     ha="center", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

        audits = ["standard_ebooks_human", "persuade_essays_human", "federal_reserve_human",
                  "stackexchange_writers_human", "pmc_full_body_human"]
        labels = ["Classic fiction", "Student essays", "Federal Reserve", "Writers SE", "PMC paper bodies"]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        x = np.arange(len(audits))
        width = .8 / len(order)
        for i, name in enumerate(order):
            vals = [reports[name].get(audit, {}).get("fpr") for audit in audits]
            if all(v is None for v in vals):
                continue
            ax.bar(x + (i - (len(order) - 1) / 2) * width, [v * 100 if v is not None else np.nan for v in vals],
                   width, label=NAMES[name], color=COLORS[name])
        ax.set_xticks(x, labels)
        ax.set_ylabel("Human text falsely flagged as AI (%)")
        ax.set_title("Human-only false-positive audits", fontsize=17)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=3)
        fig.subplots_adjust(bottom=.22, top=.88)
        fig.text(.5, .07, "Writers Stack Exchange is pre-2023 but individual human authorship is not verified.\n"
                 "PERSUADE is restricted student writing; all raw text remains outside Git.",
                 ha="center", va="center", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
