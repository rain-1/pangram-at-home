"""Make the fixed-evaluation span data-size curve report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/mnt/f/pangram-at-home")
REPORTS = Path(__file__).resolve().parents[1] / "reports"
RUNS = (
    ("5k · 1 epoch", "qwen3_token_repeat2_v5_5k_e1", 5000, 816),
    ("10k · 1 epoch", "qwen3_token_repeat2_v5_10k_e1", 10000, 1634),
    ("20k · 1 epoch", "qwen3_token_repeat2_v5_20k_e1", 20000, 3269),
    ("5k · 4 epochs", "qwen3_token_repeat2_v5_5k_e4", 5000, 3269),
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
                    f"{val:.1f}%", ha="center", fontsize=9)
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
    lines = ["# Span data-size curve", "",
             "Nested 5k, 10k, and 20k training documents use the same Qwen3-1.7B Repeat2 token architecture, Vast-selected LoRA settings, initialization adapter, fixed validation, and frozen test sets. The 5k × 4 run matches the 20k run's 3,269 optimizer steps to separate exposure to new data from additional updates. Thresholds are calibrated separately on the same pure-human calibration set at 5% document-any false highlight.", "",
             "| Run | Documents | Optimizer steps | LLMTrace heldout AUROC | LLMTrace AI recall | LLMTrace human FPR | AITDNA mixed AI recall | AITDNA mixed human FPR | Locked human any-highlight | CoAuthor AI recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, row in data.items():
        values = (percent(row["LLMTrace heldout"], "roc_auc"),
                  percent(row["LLMTrace heldout"], "ai_recall"),
                  percent(row["LLMTrace heldout"], "fpr"),
                  percent(row["AITDNA"], "ai_recall", "mixed"),
                  percent(row["AITDNA"], "fpr", "mixed"),
                  percent(row["Locked human"], "pure_human_document_any_false_highlight_rate"),
                  percent(row["CoAuthor"], "ai_recall"))
        lines.append(f"| {label} | {row['docs']:,} | {row['steps']:,} | " +
                     " | ".join(f"{v:.1f}%" for v in values) + " |")
    if old is not None:
        lines += ["", f"The previous v4 checkpoint has {percent(old, 'roc_auc'):.1f}% AUROC and recalls {percent(old, 'ai_recall'):.1f}% of AI tokens at {percent(old, 'fpr'):.2f}% human-token FPR on the same held-out LLMTrace test. It trained on the older 5k synthetic mix and was not part of this controlled nested-mixture sweep."]
    lines += ["", "The 20k tier consists of 4,964 unique earlier synthetic composites and 15,036 substantial English LLMTrace documents. The 5k and 10k tiers are subsets of it. Labeled AI characters comprise 48.3–48.6% across tiers. Train, validation, and test texts are exact-hash disjoint; LLMTrace topic groups overlapping the earlier validation and frozen diverse test were excluded. These experiments do not establish a 50k-data result or guarantee generalization beyond the tested generators and domains.", ""]
    (REPORTS / "span_size_curve_v5.md").write_text("\n".join(lines))
    print(REPORTS / "span_size_curve_v5.pdf")


if __name__ == "__main__":
    main()
