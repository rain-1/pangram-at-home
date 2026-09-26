"""Make the fixed-evaluation span data-size curve report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


ROOT = Path("/mnt/f/pangram-at-home")
REPORTS = Path(__file__).resolve().parents[1] / "reports"
RUNS = (
    ("5k · 1 epoch", "qwen3_token_repeat2_v5_5k_e1", 5000, 816),
    ("10k · 1 epoch", "qwen3_token_repeat2_v5_10k_e1", 10000, 1634),
    ("20k · 1 epoch", "qwen3_token_repeat2_v5_20k_e1_local", 20000, 3269),
    ("5k · 4 epochs", "qwen3_token_repeat2_v5_5k_e4_local", 5000, 3269),
)
SETS = {
    "LLMTrace heldout": "v5_llmtrace_heldout",
    "Prior synthetic": "v5_prior_synthetic_val",
    "Locked human": "v5_human_locked_test",
    "AITDNA": "v5_aitdna",
    "CoAuthor": "v5_coauthor",
}


def percent(report: dict, key: str, kind: str = "overall") -> float:
    x = report["overall"] if kind == "overall" else report["by_kind"][kind]
    return 100 * (x.get(key) or 0)


def recall_at_fpr(path: Path, target_fpr: float) -> float:
    scores = np.load(path)
    fpr, tpr, _ = roc_curve(scores["label"], scores["score"])
    return 100 * float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    data = {}
    for label, run, docs, steps in RUNS:
        folder = ROOT / "runs" / run
        summary = json.loads((folder / "train_summary.json").read_text())
        if summary["global_step"] != steps:
            raise ValueError(f"Incomplete {run}")
        data[label] = {"docs": docs, "steps": steps, "summary": summary,
                       **{name: json.loads((folder / f"{file}.json").read_text())
                          for name, file in SETS.items()}}
    old_path = ROOT / "runs/qwen3_token_repeat2_v4_pilot1/v5_llmtrace_heldout.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    target_fpr = old["overall"]["fpr"] if old is not None else .0067
    labels = list(data)
    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("LLMTrace heldout: AI token recall", "LLMTrace heldout", "ai_recall", "overall"),
        ("AITDNA mixed: AI token recall", "AITDNA", "ai_recall", "mixed"),
        ("AITDNA mixed: human token FPR", "AITDNA", "fpr", "mixed"),
        ("Locked human: docs falsely highlighted", "Locked human", "pure_human_document_any_false_highlight_rate", "overall"),
    )
    for ax, (title, dataset, metric, kind) in zip(axes.flat, panels):
        values = [percent(data[name][dataset], metric, kind) for name in labels]
        colors = ["#4b8fc4", "#2876b6", "#07538b", "#b3843d"]
        bars = ax.bar(x, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(values + [1]) * .02,
                    f"{val:.2f}%" if metric == "pure_human_document_any_false_highlight_rate" else f"{val:.1f}%",
                    ha="center", fontsize=9)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, max(values + [1]) * 1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if old is not None and dataset == "LLMTrace heldout":
            reference = percent(old, "ai_recall")
            ax.axhline(reference, color="#ae4835", linestyle="--", linewidth=1.3,
                       label=f"previous v4: {reference:.1f}%")
            ax.legend(fontsize=8)
    fig.suptitle("Span model data-size curve: fixed architecture and evaluations", fontsize=15)
    fig.text(.5, .015, "One epoch per distinct size; 5k × 4 matches the 20k optimizer steps. All thresholds are independently\n"
             "calibrated at 5% document-any false highlight on the same pure-human calibration set.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95), h_pad=2)
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "span_size_curve_v5.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    curves = [(label, ROOT / "runs" / run / "v5_llmtrace_heldout_scores.npz")
              for label, run, _, _ in RUNS]
    if old is not None:
        curves.insert(0, ("Previous v4", ROOT / "runs/qwen3_token_repeat2_v4_pilot1/v5_llmtrace_heldout_scores.npz"))
    colors = ["#a84432", "#4b8fc4", "#2876b6", "#07538b", "#b3843d"]
    for (label, path), color in zip(curves, colors):
        scores = np.load(path)
        fpr, tpr, _ = roc_curve(scores["label"], scores["score"])
        for ax in axes:
            ax.plot(fpr * 100, tpr * 100, label=label, color=color, linewidth=1.8)
    axes[0].plot([0, 100], [0, 100], linestyle=":", color="#777", label="Chance")
    axes[0].set(xlim=(0, 100), ylim=(0, 100), title="Full ROC")
    axes[1].set(xlim=(0, 5), ylim=(0, 100), title="Low false-positive region")
    for ax in axes:
        ax.set_xlabel("Human-token false-positive rate (%)")
        ax.set_ylabel("AI-token recall (%)")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Held-out LLMTrace token ROC: 2,000 documents", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORTS / "span_size_curve_v5_roc.pdf")
    plt.close(fig)

    if old is not None:
        domains = sorted(old["by_domain"])
        domain_models = [("Previous v4", old)] + [(label, data[label]["LLMTrace heldout"]) for label in labels]
        recall_grid = np.array([[100 * (report["by_domain"][domain].get("ai_recall") or 0)
                                 for label, report in domain_models] for domain in domains])
        fpr_grid = np.array([[100 * (report["by_domain"][domain].get("fpr") or 0)
                              for label, report in domain_models] for domain in domains])
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, grid, title, cmap, fmt in ((axes[0], recall_grid, "AI token recall", "Blues", ".1f"),
                                            (axes[1], fpr_grid, "Human token FPR", "Oranges", ".2f")):
            img = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0)
            ax.set_xticks(range(len(domain_models)), [label for label, _ in domain_models],
                          rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(domains)), domains)
            ax.set_title(title)
            for i in range(len(domains)):
                for j in range(len(domain_models)):
                    val = grid[i, j]
                    ax.text(j, i, f"{val:{fmt}}%", ha="center", va="center",
                            color="white" if val > grid.max() * .55 else "black", fontsize=8)
            fig.colorbar(img, ax=ax, shrink=.75)
        fig.suptitle("Held-out LLMTrace performance by domain", fontsize=14)
        fig.tight_layout()
        fig.savefig(REPORTS / "span_size_curve_v5_domains.pdf")
        plt.close(fig)
    lines = ["# Span data-size curve", "",
             "Nested 5k, 10k, and 20k training documents use the same Qwen3-1.7B Repeat2 token architecture, Vast-selected LoRA settings, initialization adapter, fixed validation, and frozen test sets. The 5k × 4 run matches the 20k run's 3,269 optimizer steps to separate exposure to new data from additional updates. The 5k and 10k runs used Vast RTX 4090s; the 20k and matched-compute 5k runs were completed on the local RTX 4080 after Vast credit ran out. All training hyperparameters and BF16 precision were held fixed, but GPU hardware is a minor remaining experimental difference. Thresholds are calibrated separately on the same pure-human calibration set at 5% document-any false highlight.", "",
             "| Run | Documents | Optimizer steps | LLMTrace heldout AUROC | LLMTrace AI recall | LLMTrace human FPR | LLMTrace recall at v4 FPR* | AITDNA mixed AI recall | AITDNA mixed human FPR | Locked human any-highlight | CoAuthor AI recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, row in data.items():
        run = next(run for entry, run, _, _ in RUNS if entry == label)
        auc = row["LLMTrace heldout"]["overall"]["roc_auc"]
        values = (percent(row["LLMTrace heldout"], "ai_recall"),
                  percent(row["LLMTrace heldout"], "fpr"),
                  recall_at_fpr(ROOT / "runs" / run / "v5_llmtrace_heldout_scores.npz", target_fpr),
                  percent(row["AITDNA"], "ai_recall", "mixed"),
                  percent(row["AITDNA"], "fpr", "mixed"),
                  percent(row["Locked human"], "pure_human_document_any_false_highlight_rate"),
                  percent(row["CoAuthor"], "ai_recall"))
        formatted = (f"{values[0]:.1f}%", f"{values[1]:.3f}%", f"{values[2]:.1f}%",
                     f"{values[3]:.1f}%", f"{values[4]:.1f}%", f"{values[5]:.2f}%",
                     f"{values[6]:.1f}%")
        lines.append(f"| {label} | {row['docs']:,} | {row['steps']:,} | {auc:.3f} | " +
                     " | ".join(formatted) + " |")
    if old is not None:
        lines += ["", f"The previous v4 checkpoint has {old['overall']['roc_auc']:.3f} AUROC and recalls {percent(old, 'ai_recall'):.1f}% of AI tokens at {percent(old, 'fpr'):.2f}% human-token FPR on the same held-out LLMTrace test. It trained on the older 5k synthetic mix and was not part of this controlled nested-mixture sweep."]
        lines += ["", "## LLMTrace held-out domains", "",
                  "| Domain | Previous v4 recall | 5k recall | 10k recall | 20k recall | 5k × 4 recall | 20k human FPR |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for domain in domains:
            recall = [100 * (report["by_domain"][domain].get("ai_recall") or 0)
                      for _, report in domain_models]
            human_fpr = 100 * (data["20k · 1 epoch"]["LLMTrace heldout"]["by_domain"][domain].get("fpr") or 0)
            lines.append(f"| {domain} | " + " | ".join(f"{v:.1f}%" for v in recall) +
                         f" | {human_fpr:.2f}% |")
    lines += ["", f"*ROC-interpolated recall at the previous v4 checkpoint's {100*target_fpr:.2f}% LLMTrace human-token FPR. This is a retrospective test-set tradeoff, not a deployable threshold. AUROC and recall at the frozen threshold answer different questions. The ROC chart shows the available recall/FPR tradeoff, while the other table columns show the prespecified calibration rule.", "",
              "The frozen pure-human calibration set contains social Q&A, professional finance, and creative writing. It does not cover all nine LLMTrace domain labels. Large differences between AUROC and recall at its calibrated threshold may therefore reflect score calibration across domains; a broader independent human calibration set is the next threshold study.", "",
              "The 20k tier consists of 4,964 unique earlier synthetic composites and 15,036 substantial English LLMTrace documents. The 5k and 10k tiers are subsets of it. Labeled AI characters comprise 48.3–48.6% across tiers. Train, validation, and test texts are exact-hash disjoint; LLMTrace topic groups overlapping the earlier validation and frozen diverse test were excluded. These experiments do not establish a 50k-data result or guarantee generalization beyond the tested generators and domains.", ""]
    (REPORTS / "span_size_curve_v5.md").write_text("\n".join(lines))
    print(REPORTS / "span_size_curve_v5.pdf")


if __name__ == "__main__":
    main()
