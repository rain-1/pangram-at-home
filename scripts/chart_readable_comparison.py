"""Create plain-language model comparison charts from frozen evaluation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home")
METRICS = REPO / "reports/metrics"
OUT = REPO / "reports/readable_model_comparison_v1.pdf"
ORDER = ["qwen", "char", "word", "embedding", "roberta", "llama", "load_bearing"]
NAMES = {"qwen": "Our Qwen model", "char": "Character TF-IDF", "word": "Word TF-IDF",
         "embedding": "MiniLM + logistic", "roberta": "EditLens RoBERTa",
         "llama": "EditLens Llama", "load_bearing": "Load Bearing"}
COLORS = {"qwen": "#104A70", "char": "#3294A4", "word": "#D6843B",
          "embedding": "#77A16A", "roberta": "#876CB0", "llama": "#B5A034",
          "load_bearing": "#AD637D"}
SPLITS = {"test": "Diverse test", "raid_external": "RAID independent test"}
DOMAIN_LABELS = {"creative": "Creative writing", "news": "News", "paper": "Papers",
                 "reference_education": "Reference / education", "reviews": "Reviews",
                 "social_qa": "Social / Q&A", "abstracts": "Abstracts", "books": "Books",
                 "poetry": "Poetry", "recipes": "Recipes", "reddit": "Reddit", "wiki": "Wiki"}


def load_reports() -> dict:
    files = {"qwen": "qwen3_17b_diverse_v1.json",
             "char": "baseline_diverse_char_full.json",
             "word": "baseline_diverse_word_full.json",
             "embedding": "baseline_diverse_embedding_full.json",
             "roberta": "reference_diverse_roberta_full.json",
             "llama": "reference_diverse_llama_full.json",
             "load_bearing": "baseline_diverse_load_bearing_full.json"}
    reports = {}
    for name, filename in files.items():
        raw = json.loads((METRICS / filename).read_text())
        reports[name] = raw["splits"] if name == "qwen" else raw
    return reports


def group(report: dict, split: str) -> dict:
    section = report[split]
    return section.get("by_domain", section.get("by_source", {}))


def source_file(split: str) -> Path:
    return (ROOT / "data/diverse_pyramid_v1/test_full.parquet" if split == "test"
            else ROOT / "data/raid_external_v1/frozen.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_scores(name: str, split: str, published_auc: float) -> tuple[np.ndarray, np.ndarray]:
    if name == "qwen":
        path = ROOT / f"runs/qwen3_17b_diverse_v1/final_eval/{split}.npz"
    elif name == "load_bearing":
        path = ROOT / f"runs/load_bearing_v1/diverse_{split}.npz"
    else:
        path = ROOT / f"runs/diverse_roc_scores_v1/{name}_{split}.npz"
    with np.load(path) as data:
        score = np.asarray(data["margin"] if name == "qwen" else data["score"])
        label = np.asarray(data["label"])
        if "input_sha256" in data and str(data["input_sha256"]) != sha256(source_file(split)):
            raise RuntimeError(f"Frozen input changed for {name}/{split}")
    auc = roc_auc_score(label, score)
    if abs(auc - published_auc) > .001:
        raise RuntimeError(f"AUROC mismatch for {name}/{split}: {auc:.6f} vs {published_auc:.6f}")
    return score, label


def heading(fig, title: str, subtitle: str) -> None:
    fig.text(.07, .955, title, fontsize=19, weight="bold", color="#17354A")
    fig.text(.07, .92, subtitle, fontsize=10, color="#42596A")


def overview(pdf: PdfPages, reports: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.subplots_adjust(left=.22, right=.97, top=.79, bottom=.16, wspace=.30)
    heading(fig, "Overall comparison", "Equal-weight mean of the 1,000-row diverse test and 1,600-row RAID test. Both are 50% human / 50% AI.")
    y = np.arange(len(ORDER))
    values = {}
    for name in ORDER:
        recall = np.mean([reports[name][s]["tpr"] for s in SPLITS]) * 100
        fpr = np.mean([reports[name][s]["fpr"] for s in SPLITS]) * 100
        accuracy = (recall + 100 - fpr) / 2
        values[name] = {"recall": recall, "accuracy": accuracy, "fpr": fpr}
    for ax, field, title, xlim in zip(axes,
                                       ("recall", "accuracy", "fpr"),
                                       ("AI recall ↑", "Balanced accuracy ↑", "Human false positives ↓"),
                                       ((0, 100), (0, 100), (0, 6.5))):
        vals = [values[n][field] for n in ORDER]
        ax.barh(y, vals, color=[COLORS[n] for n in ORDER], height=.64)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_yticks(y, [NAMES[n] for n in ORDER] if ax is axes[0] else [""] * len(ORDER))
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_xlabel("Percent")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(vals):
            ax.text(min(value + (1.4 if field != "fpr" else .10), xlim[1] - .05), i,
                    f"{value:.1f}%", va="center", fontsize=10, weight="bold" if i == 0 else "normal")
    fig.text(.07, .085, "Recall: share of AI passages caught. Balanced accuracy: mean of AI recall and human specificity."
             " False positives: share of human passages wrongly flagged.", fontsize=9, color="#42596A")
    fig.text(.07, .055, "Each model uses its own threshold chosen on the same validation set for ≤2% human false positives."
             " Observed test rates can differ.", fontsize=9, color="#42596A")
    pdf.savefig(fig)
    fig.savefig(REPO / "reports/readable_overview_v1.png", dpi=180)
    plt.close(fig)


def heatmap(pdf: PdfPages, reports: dict, split: str, field: str) -> None:
    domains = list(group(reports["qwen"], split))
    matrix = np.asarray([[group(reports[n], split)[d][field] * 100 for d in domains] for n in ORDER])
    is_recall = field == "tpr"
    fig, ax = plt.subplots(figsize=(15, 8.4))
    fig.subplots_adjust(left=.20, right=.91, top=.78, bottom=.27)
    title = f"{SPLITS[split]}: {'AI recall' if is_recall else 'human false positives'} by category"
    subtitle = ("Higher is better. Each cell is the percent of AI passages correctly flagged."
                if is_recall else "Lower is better. Each cell is the percent of human passages wrongly flagged.")
    heading(fig, title, subtitle)
    max_value = 100 if is_recall else max(10, np.ceil(matrix.max() / 5) * 5)
    im = ax.imshow(matrix, cmap="YlGn" if is_recall else "YlOrRd", vmin=0, vmax=max_value, aspect="auto")
    ax.set_xticks(range(len(domains)), [DOMAIN_LABELS.get(d, d.title()) for d in domains], rotation=25, ha="right")
    ax.set_yticks(range(len(ORDER)), [NAMES[n] for n in ORDER])
    ax.axhline(.5, color="#17354A", linewidth=2)
    for i in range(len(ORDER)):
        for j, domain in enumerate(domains):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=10,
                    weight="bold" if i == 0 else "normal", color="#132632")
    fig.colorbar(im, ax=ax, fraction=.024, pad=.025, label="Percent")
    counts = [group(reports["qwen"], split)[d]["ai" if is_recall else "human"] for d in domains]
    fig.text(.20, .09, "Passages per category (" + ("AI" if is_recall else "human") + "): " +
             ", ".join(f"{DOMAIN_LABELS.get(d, d)} {n}" for d, n in zip(domains, counts)),
             fontsize=8.5, color="#42596A")
    fig.text(.20, .055, "Small categories have noisy percentages; one error changes a 50-passage category by 2 points.",
             fontsize=8.5, color="#42596A")
    pdf.savefig(fig)
    plt.close(fig)


def roc_page(pdf: PdfPages, reports: dict, split: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.4))
    fig.subplots_adjust(left=.07, right=.97, top=.78, bottom=.18, wspace=.25)
    heading(fig, f"{SPLITS[split]}: ROC curves", "Each curve shows AI recall as the false-positive rate changes across all possible thresholds.")
    for name in ORDER:
        auc = reports[name][split]["roc_auc"]
        score, label = get_scores(name, split, auc)
        fpr, tpr, _ = roc_curve(label, score)
        for ax in axes:
            ax.plot(fpr * 100, tpr * 100, color=COLORS[name],
                    linewidth=3 if name == "qwen" else 1.65, alpha=1 if name == "qwen" else .8,
                    label=f"{NAMES[name]}  ·  AUC {auc:.3f}" if ax is axes[0] else None)
    for ax, right, title in zip(axes, (100, 10), ("Full range", "Focus: 0–10% false positives")):
        ax.plot([0, right], [0, right], linestyle="--", color="#778590", linewidth=1, alpha=.7)
        ax.set_xlim(0, right)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Human false-positive rate (%)")
        ax.set_ylabel("AI recall (%)")
        ax.set_title(title, fontsize=13)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=.95)
    fig.text(.07, .075, "AUC summarizes ranking across all thresholds. It does not say which false-positive rate a deployed threshold will have.",
             fontsize=9, color="#42596A")
    pdf.savefig(fig)
    if split == "raid_external":
        fig.savefig(REPO / "reports/readable_roc_raid_v1.png", dpi=180)
    else:
        fig.savefig(REPO / "reports/readable_roc_diverse_v1.png", dpi=180)
    plt.close(fig)


def human_audits(pdf: PdfPages, reports: dict) -> None:
    audits = [("standard_ebooks_human", "Classic fiction"),
              ("persuade_essays_human", "Student essays"),
              ("federal_reserve_human", "Federal Reserve"),
              ("stackexchange_writers_human", "Writers Stack Exchange"),
              ("pmc_full_body_human", "Paper body text")]
    fig, ax = plt.subplots(figsize=(15, 8.4))
    fig.subplots_adjust(left=.21, right=.96, top=.79, bottom=.23)
    heading(fig, "Additional human-only checks", "False positives on human text from sources outside the two balanced test sets. Lower is better.")
    y = np.arange(len(audits))
    width = .10
    audit_models = [name for name in ORDER if all(key in reports[name] for key, _ in audits)]
    for i, name in enumerate(audit_models):
        vals = [reports[name][key]["fpr"] * 100 for key, _ in audits]
        ax.barh(y + (i - (len(audit_models) - 1) / 2) * width, vals,
                height=width, color=COLORS[name], label=NAMES[name])
    ax.set_yticks(y, [f"{label}\n(n={reports['qwen'][key]['rows']:,})" for key, label in audits])
    ax.invert_yaxis()
    ax.set_xlabel("Human passages wrongly flagged as AI (%)")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(.59, .015), fontsize=9)
    fig.text(.21, .14, "These sets contain no AI examples, so they measure false positives only."
             " Writers Stack Exchange authorship is inferred from pre-2023 posts.", fontsize=9, color="#42596A")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    reports = load_reports()
    with PdfPages(OUT) as pdf:
        overview(pdf, reports)
        for split in SPLITS:
            heatmap(pdf, reports, split, "tpr")
            heatmap(pdf, reports, split, "fpr")
        for split in SPLITS:
            roc_page(pdf, reports, split)
        human_audits(pdf, reports)
    print(OUT)


if __name__ == "__main__":
    main()
