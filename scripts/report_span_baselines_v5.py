"""Compare v4 span localization with calibrated window baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/f/pangram-at-home/runs")
REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"

MODELS = {
    "Token v4": ("qwen3_token_repeat2_v4_pilot1", {
        "synthetic": "v4_synthetic_val.json", "human": "v4_human_locked_test.json",
        "aitdna": "aitdna_locked_v1.json", "coauthor": "v4_realistic_locked_test.json"}),
    "Token v3": ("qwen3_token_repeat2_v3_pilot1", {
        "synthetic": "v4_synthetic_val.json", "human": "v4_human_locked_test.json",
        "aitdna": "aitdna_locked_v1.json", "coauthor": "v4_realistic_locked_test.json"}),
    "Qwen passage": ("vast_hpo_selected_v3", {
        "synthetic": "span_v5_synthetic_v4_val.json", "human": "span_v5_human_locked_test.json",
        "aitdna": "span_v5_aitdna.json", "coauthor": "span_v5_coauthor.json"}),
    "Char TF-IDF": ("span_char_window_baseline_v5", None),
    "Word TF-IDF": ("span_word_window_baseline_v5", None),
}
KEYS = {
    "synthetic": "synthetic_v4_val", "human": "human_locked_test",
    "aitdna": "aitdna", "coauthor": "coauthor",
}


def load() -> dict:
    data = {}
    for model, (folder, filenames) in MODELS.items():
        base = ROOT / folder
        if filenames is None:
            report = json.loads((base / "report.json").read_text())
            data[model] = {key: report["sets"][name] for key, name in KEYS.items()}
        else:
            data[model] = {key: json.loads((base / name).read_text()) for key, name in filenames.items()}
    return data


def value(data: dict, model: str, set_name: str, metric: str, kind: str = "overall") -> float:
    report = data[model][set_name]
    x = report["overall"] if kind == "overall" else report["by_kind"][kind]
    return 100 * (x.get(metric) or 0)


def main() -> None:
    data = load()
    REPORTS.mkdir(exist_ok=True)
    models = list(MODELS)
    colors = ["#116ab4", "#7ba7c7", "#e39636", "#6aa96e", "#9274b5"]
    panels = [
        ("V4 synthetic: AI token recall", "synthetic", "ai_recall", "overall"),
        ("V4 synthetic: human token FPR", "synthetic", "fpr", "overall"),
        ("AITDNA mixed: AI token recall", "aitdna", "ai_recall", "by_kind.mixed"),
        ("AITDNA mixed: human token FPR", "aitdna", "fpr", "by_kind.mixed"),
        ("Locked human: docs falsely highlighted", "human", "pure_human_document_any_false_highlight_rate", "overall"),
        ("CoAuthor: AI token recall", "coauthor", "ai_recall", "overall"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, (title, set_name, metric, kind) in zip(axes.flat, panels):
        values = [value(data, model, set_name, metric, kind.split(".")[-1]) for model in models]
        ax.barh(np.arange(len(models)), values, color=colors)
        ax.set_yticks(np.arange(len(models)), models)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("Percent")
        ax.set_xlim(0, max(values) * 1.2 + (1 if max(values) < 10 else 0))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for i, x in enumerate(values):
            ax.text(x + ax.get_xlim()[1] * .015, i, f"{x:.1f}%", va="center", fontsize=9)
    fig.suptitle("Span model v4 vs calibrated window baselines", fontsize=16)
    fig.text(.5, .02, "All thresholds set independently at 5% document-any false highlight on the same pure-human calibration set.\n"
             "AITDNA and CoAuthor are real collaboration sets; CoAuthor includes many very short model insertions.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95), h_pad=2.1)
    fig.savefig(REPORTS / "span_v5_baselines.pdf")
    plt.close(fig)

    lines = ["# Span localization: v4 and baselines", "",
             "All methods use a separately calibrated threshold allowing 5% of documents in the same 1,120-document pure-human calibration set to receive any false highlight. All methods are then evaluated without threshold adjustment. Load Bearing is omitted from this new comparison as requested. The char and word TF-IDF baselines train on 10,000 labeled passage documents and broadcast sliding-window predictions across tokens; the Qwen passage baseline uses the selected Vast checkpoint the same way.", "",
             "| Model | V4 synthetic AI token recall | V4 synthetic human token FPR | AITDNA mixed AI token recall | AITDNA mixed human token FPR | Locked human docs with any false highlight | CoAuthor AI token recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        cells = [value(data, model, s, m, k) for s, m, k in (
            ("synthetic", "ai_recall", "overall"), ("synthetic", "fpr", "overall"),
            ("aitdna", "ai_recall", "mixed"), ("aitdna", "fpr", "mixed"),
            ("human", "pure_human_document_any_false_highlight_rate", "overall"),
            ("coauthor", "ai_recall", "overall"))]
        lines.append("| " + model + " | " + " | ".join(f"{x:.1f}%" for x in cells) + " |")
    lines += ["", "The synthetic set contains 600 documents; the locked human set contains 3,579. AITDNA has 258 mixed documents; its mixed human-token FPR is the most direct indicator of false span marking within actual collaborative writing. CoAuthor has short accepted suggestions and substantial provenance masking, so its token recall tests a harder and narrower use case than paragraph localization.", "",
              "The passage Qwen baseline has strong AI recall but marks substantially more human material inside mixed documents. V4 is the clearest overall tradeoff at this fixed calibration rule. These are descriptive rates on fixed datasets, not a guarantee for unseen authors, prompts, or generator models.", ""]
    (REPORTS / "span_v5_baselines.md").write_text("\n".join(lines))
    print(REPORTS / "span_v5_baselines.pdf")


if __name__ == "__main__":
    main()
