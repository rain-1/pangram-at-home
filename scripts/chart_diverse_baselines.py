"""Plot baseline strengths by domain on the new source-aware test split."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/mnt/f/pangram-at-home/results")
OUT = Path(__file__).resolve().parents[1] / "reports/diverse_baselines_v1.pdf"
MODELS = ["char", "word", "embedding"]
NAMES = {"char": "Character TF-IDF", "word": "Word TF-IDF", "embedding": "MiniLM + logistic"}
COLORS = {"char": "#1E5B8F", "word": "#E98943", "embedding": "#57A681"}


def main() -> None:
    reports = {name: json.loads((ROOT / f"baseline_diverse_{name}_full.json").read_text()) for name in MODELS}
    domains = list(reports["char"]["test"]["by_domain"])
    with PdfPages(OUT) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), constrained_layout=True)
        x = np.arange(len(domains))
        width = .24
        for i, name in enumerate(MODELS):
            values = [reports[name]["test"]["by_domain"][d]["roc_auc"] for d in domains]
            axes[0].bar(x + (i - 1) * width, values, width, label=NAMES[name], color=COLORS[name])
            recall = [reports[name]["test"]["by_domain"][d]["tpr"] for d in domains]
            axes[1].bar(x + (i - 1) * width, recall, width, color=COLORS[name])
        for ax in axes:
            ax.set_xticks(x, [d.replace("_", " ").title() for d in domains])
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[0].set_title("Diverse test: AUROC by writing domain", fontsize=15)
        axes[0].set_ylabel("AUROC")
        axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0))
        axes[1].set_title("AI recall at each model's 2% validation human-FPR threshold", fontsize=15)
        axes[1].set_ylabel("AI recall")
        fig.suptitle("Baseline performance | 1,000 balanced test passages | source-aware pyramid v1", fontsize=17)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        columns = ["Model", "Mixed AUROC", "Mixed FPR", "Mixed AI recall", "GPT-4 OOD AUROC", "Paraphrase AUROC", "Books human FPR"]
        rows = []
        for name in MODELS:
            r = reports[name]
            rows.append([NAMES[name], f"{r['test']['roc_auc']:.3f}", f"{r['test']['fpr']:.1%}",
                         f"{r['test']['tpr']:.1%}", f"{r['gpt4_ood']['roc_auc']:.3f}",
                         f"{r['paraphrase']['roc_auc']:.3f}", f"{r['standard_ebooks_human']['fpr']:.1%}"])
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_facecolor("#E8F0F5" if row == 0 else ("#F6F8FA" if row % 2 else "white"))
            cell.set_edgecolor("#D8E0E5")
        ax.set_title("Baseline summary", fontsize=18, pad=25)
        ax.text(.5, .2, "All thresholds selected on the diverse validation split only.\n"
                "MAGE GPT-4 and paraphrase sets share the MAGE source family with training.\n"
                "Standard Ebooks is human-only, so AUROC is undefined.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
