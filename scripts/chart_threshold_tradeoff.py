"""Draw a compact scorecard of validation-chosen threshold policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "reports/threshold-tradeoff.json").read_text())
OUT = ROOT / "reports/charts"


def format_cell(metric: dict, kind: str) -> tuple[str, str]:
    if kind == "recall":
        rate = metric["recall"]
        text = f"{rate:.1%}\n{metric['tp']}/{metric['ai']}"
        color = "#e4f4ed" if rate >= 0.99 else "#fff1dc"
    else:
        rate = metric["fpr"]
        text = f"{rate:.2%}\n{metric['fp']}/{metric['human']}"
        color = "#e4f4ed" if rate < 0.02 else "#fae5e1"
    return text, color


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(17.2, 7.7), dpi=170)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.025, 0.955, "Can we lower false positives by moving the threshold?", fontsize=20, weight="bold", color="#172638", va="center")
    ax.text(0.025, 0.902, "One saved model · six cutoff rules chosen from mixed validation labels or fixed in advance · no retraining", fontsize=10, color="#53657a", va="center")

    cols = [
        ("RULE", 0.028, 0.25, None, None),
        ("AI SCORE\nCUTOFF", 0.287, 0.084, None, None),
        ("MIXED\nAI RECALL", 0.377, 0.095, "mixed_test", "recall"),
        ("MIXED\nHUMAN FPR", 0.478, 0.095, "mixed_test", "fpr"),
        ("PAPER\nAI RECALL", 0.579, 0.095, "paper_subset", "recall"),
        ("PAPER\nHUMAN FPR", 0.680, 0.095, "paper_subset", "fpr"),
        ("ACL ABSTRACTS\nHUMAN FPR", 0.781, 0.095, "acl_human_audit", "fpr"),
        ("PMC BODY\nHUMAN FPR", 0.882, 0.095, "pmc_body_audit", "fpr"),
    ]
    for header, x, width, _, _ in cols:
        ax.text(x + width / 2, 0.825, header, ha="center", va="center", fontsize=8.3, weight="bold", color="#53657a")

    for index, policy in enumerate(data["policies"]):
        y = 0.748 - index * 0.099
        if policy["policy"] == "val_midpoint":
            ax.add_patch(Rectangle((0.021, y - 0.042), 0.958, 0.087, facecolor="#eeeafd", edgecolor="none"))
            ax.add_patch(Rectangle((0.021, y - 0.042), 0.004, 0.087, facecolor="#6849bd", edgecolor="none"))
        elif index % 2 == 0:
            ax.add_patch(Rectangle((0.021, y - 0.042), 0.958, 0.087, facecolor="#ffffff", edgecolor="none"))
        ax.text(0.03, y, policy["label"], fontsize=9.3, color="#172638", va="center", weight="bold" if policy["policy"] == "val_midpoint" else "normal")
        cutoff = policy["ai_probability_cutoff"]
        cutoff_text = f"{cutoff:.3%}" if cutoff > 0.99 else f"{cutoff:.1%}"
        ax.text(0.329, y, cutoff_text, fontsize=9, ha="center", va="center", color="#384b60")
        for _, x, width, key, kind in cols[2:]:
            label, color = format_cell(policy[key], kind)
            ax.add_patch(Rectangle((x, y - 0.038), width, 0.078, facecolor=color, edgecolor="none"))
            ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=8.25, linespacing=1.35, color="#183b37" if kind == "recall" else "#64312d")

    ax.text(0.025, 0.108, "Paper = 221 human + 221 AI abstracts within the mixed test. ACL and PMC body are human-only audits (17,560 and 261 texts).", fontsize=9.1, color="#53657a", va="center")
    ax.text(0.025, 0.073, "Cutoffs use the 622-human/622-AI mixed validation set; the 50% rule is fixed. Holdouts were used only to measure outcomes.", fontsize=9.1, color="#53657a", va="center")
    ax.text(0.025, 0.038, "Rates are observed on these sets, not a guarantee. Green FPR cells are below 2%. The reported AI score is an uncalibrated softmax value.", fontsize=9.1, color="#53657a", va="center")
    fig.savefig(OUT / "threshold_tradeoff.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / "threshold_tradeoff.png", bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
