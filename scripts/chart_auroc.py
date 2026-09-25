"""Plot comparable held-out AUROC scores and ROC curves from frozen splits."""

from pathlib import Path
import json
import os
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home"))
OUT = ROOT / "reports" / "charts"
OUT.mkdir(exist_ok=True, parents=True)
COLORS = {"Qwen3 1.7B": "#b03a48", "Llama 3B": "#735fb1", "RoBERTa": "#28799a", "Char TF-IDF full": "#4b8f55", "Char TF-IDF medium": "#8bad6f", "Word TF-IDF": "#c48a32", "MiniLM embedding": "#858585"}


def qwen(dataset):
    p = DATA / "runs/qwen3_17b_mixed_stage1_v1/score_cache_v1/mixed_test.npz"
    z = np.load(p)
    mask = np.ones(len(z["label"]), dtype=bool) if dataset == "mixed" else np.isin(z["source"], ["acl_anthology", "pmc_oa"])
    return z["label"][mask], z["margin"][mask]


def baseline(dataset, model):
    z = np.load(DATA / "runs/roc_cache_v1" / f"{dataset}_{model}.npz")
    return z["label"], z["score"]


def plot_curves():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    summaries = {}
    for row, dataset in enumerate(("mixed", "paper")):
        models = [("Qwen3 1.7B", None), ("Llama 3B", "llama"), ("RoBERTa", "roberta"), ("Char TF-IDF full", "char_full" if dataset == "mixed" else "char"), ("Word TF-IDF", "word"), ("MiniLM embedding", "embedding")]
        if dataset == "mixed":
            models.insert(4, ("Char TF-IDF medium", "char"))
        summaries[dataset] = {}
        for name, model in models:
            labels, scores = qwen(dataset) if model is None else baseline(dataset, model)
            auc = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            summaries[dataset][name] = {"auroc": float(auc), "human": int((labels == 0).sum()), "ai": int((labels == 1).sum())}
            for ax in axes[row]:
                ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.3 if model is None else 1.5, alpha=0.95, label=f"{name}  {auc:.5f}")
        labels, scores = qwen(dataset)
        original = math.log(.0373971275985241 / (1 - .0373971275985241))
        midpoint = json.loads((ROOT / "configs/qwen3_stage1_operating_point.json").read_text())["threshold_margin"]
        for threshold, marker, label in [(original, "o", "Original cutoff"), (midpoint, "*", "Validation midpoint")]:
            pred = scores >= threshold
            point = (float(pred[labels == 0].mean()), float(pred[labels == 1].mean()))
            axes[row, 1].scatter(*point, s=105 if marker == "*" else 65, marker=marker, facecolor="#b03a48" if marker == "*" else "white", edgecolor="#b03a48", zorder=10, label=label)
        for ax in axes[row]:
            ax.grid(alpha=.17)
            ax.set_xlabel("False-positive rate (human flagged as AI)")
            ax.set_ylabel("AI recall")
        axes[row, 0].plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=.8)
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_ylim(0, 1.01)
        axes[row, 0].set_title(f"{dataset.title()} test · full ROC")
        axes[row, 1].set_xlim(0, .05)
        axes[row, 1].set_ylim(0, 1.01)
        axes[row, 1].set_title(f"{dataset.title()} test · 0–5% human FPR")
        axes[row, 1].legend(loc="lower right", fontsize=7.7, framealpha=.95)
    fig.suptitle("AI detector ranking on shared, balanced holdouts", fontsize=16, fontweight="bold")
    fig.text(.5, .005, "AUROC measures ranking across thresholds; use the right panels to inspect the low false-positive region.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .025, 1, .965])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"roc_curves.{ext}", dpi=200)
    plt.close(fig)
    (ROOT / "reports/auroc_scores.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def plot_overview():
    specs = [
        ("Mixed test", [("Qwen3 1.7B", "segment_qwen3_17b_mixed_stage1_v1", "test"), ("Llama 3B", "reference_mixed_llama_full", "test"), ("RoBERTa", "reference_mixed_roberta_full", "test"), ("Char TF-IDF full", "baseline_mixed_char_full", "test"), ("Char TF-IDF medium", "baseline_mixed_char_medium", "test"), ("Char TF-IDF small", "baseline_mixed_char_small", "test"), ("Char TF-IDF tiny", "baseline_mixed_char_tiny", "test"), ("Word TF-IDF", "baseline_mixed_word_medium", "test"), ("MiniLM embedding", "baseline_mixed_embedding_medium", "test")]),
        ("Paper abstracts", [("Qwen3 1.7B", "segment_qwen3_17b_mixed_stage1_v1", "test_paper"), ("Llama 3B", "reference_paper_llama_full", "test"), ("RoBERTa", "reference_paper_roberta_full", "test"), ("Char TF-IDF full", "baseline_paper_char_full", "test"), ("Char TF-IDF medium", "baseline_paper_char_medium", "test"), ("Char TF-IDF small", "baseline_paper_char_small", "test"), ("Char TF-IDF tiny", "baseline_paper_char_tiny", "test"), ("Word TF-IDF", "baseline_paper_word_full", "test"), ("MiniLM embedding", "baseline_paper_embedding_full", "test")]),
        ("General EditLens", [("Qwen3 1.7B", "segment_qwen3_17b_mixed_stage1_v1", "editlens_test"), ("Llama 3B*", "reference_editlens_llama_small", "test"), ("RoBERTa", "reference_editlens_roberta_full", "test"), ("Char TF-IDF", "baseline_editlens_char_medium", "test"), ("Word TF-IDF", "baseline_editlens_word_medium", "test"), ("MiniLM embedding", "baseline_editlens_embedding_medium", "test")]),
        ("PMC-only test†", [("Llama 3B", "reference_pmc_llama_full", "test"), ("RoBERTa", "reference_pmc_roberta_full", "test"), ("Char TF-IDF", "baseline_pmc_char_full", "test"), ("Word TF-IDF", "baseline_pmc_word_full", "test"), ("MiniLM embedding", "baseline_pmc_embedding_full", "test")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, entries) in zip(axes.flat, specs):
        names, values = [], []
        for name, stem, key in entries:
            data = json.loads((ROOT / "reports/metrics" / f"{stem}.json").read_text())
            if key == "test_paper":
                labels, scores = qwen("paper")
                auc = roc_auc_score(labels, scores)
            else:
                auc = data[key]["roc_auc"]
            names.append(name)
            values.append(auc)
        y = np.arange(len(names))[::-1]
        ax.hlines(y, .8, values, color="#d3d9dd", linewidth=2)
        ax.scatter(values, y, s=54, c=[COLORS.get(n, "#888888") for n in names], zorder=3)
        for yi, val in zip(y, values):
            ax.text(min(val + .003, 1.004), yi, f"{val:.5f}", va="center", fontsize=8)
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xlim(.8, 1.065)
        ax.set_xticks([.8, .85, .9, .95, 1.0])
        ax.set_xlabel("AUROC (axis starts at 0.8)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("AUROC across every evaluated baseline", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "* Llama used a smaller 500+500 EditLens test.  † Qwen omitted: this PMC-only split overlaps its training data.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"auroc_overview.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_curves()
    plot_overview()
