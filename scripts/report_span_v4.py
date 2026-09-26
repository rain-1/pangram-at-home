"""Summarize the frozen-threshold Repeat2 v3/v4 comparison after collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OLD = "qwen3_token_repeat2_v3_pilot1"
NEW = "qwen3_token_repeat2_v4_pilot1"
SETS = ("v4_human_calibration", "v4_synthetic_val", "v4_prior_synthetic_val",
        "v4_human_locked_test", "v4_realistic_locked_test")


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def load(root: Path, name: str):
    result = {}
    for dataset in SETS:
        path = root / "runs" / name / f"{dataset}.json"
        if path.exists():
            result[dataset] = json.loads(path.read_text())
    return result


def table_line(label: str, old, new, fmt=pct):
    return f"| {label} | {fmt(old)} | {fmt(new)} |"


def alternate_token_point(root: Path, name: str):
    calibration = root / "runs" / name / "v4_human_calibration_scores.npz"
    if not calibration.exists():
        return None
    with np.load(calibration) as cache:
        human_scores = np.asarray(cache["score"][cache["label"] == 0], dtype=np.float64)
    ordered = np.sort(human_scores)[::-1]
    threshold = float(np.nextafter(ordered[int(.02 * len(ordered))], np.inf))
    results = {}
    for dataset in SETS:
        path = root / "runs" / name / f"{dataset}_scores.npz"
        if not path.exists():
            continue
        with np.load(path) as cache:
            score = np.asarray(cache["score"], dtype=np.float64)
            label = np.asarray(cache["label"], dtype=np.int8)
            offsets = np.asarray(cache["document_offsets"])
        pred = score >= threshold
        h, a = label == 0, label == 1
        pure = [pred[start:end].any() for start, end in zip(offsets[:-1], offsets[1:])
                if end > start and np.all(label[start:end] == 0)]
        results[dataset] = {"fpr": float(pred[h].mean()) if h.any() else None,
                            "ai_recall": float(pred[a].mean()) if a.any() else None,
                            "pure_human_doc_any": float(np.mean(pure)) if pure else None,
                            "pure_human_documents": len(pure)}
    return {"threshold": threshold, "sets": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args()
    reports = {OLD: load(args.root, OLD), NEW: load(args.root, NEW)}
    if any("v4_human_locked_test" not in reports[name] for name in (OLD, NEW)):
        raise SystemExit("Both frozen human test reports are required")
    args.out.mkdir(parents=True, exist_ok=True)
    old, new = reports[OLD], reports[NEW]
    alternate = {OLD: alternate_token_point(args.root, OLD), NEW: alternate_token_point(args.root, NEW)}
    result = {"old": old, "new": new, "alternate_token_calibration": alternate}
    (args.out / "span_v4_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Repeat2 span model: v4 data experiment", "",
             "The v3 and v4 token models were calibrated separately on the same independent,",
             "pure-human calibration set. Each threshold allows at most 5% of calibration",
             "documents to receive any false highlight. The thresholds were frozen before",
             "scoring the held-out human set. Both models use the same Qwen3-1.7B base,",
             "Vast passage-adapter initialization, Repeat2, LoRA modules, learning rate,",
             "and effective batch size. V4 trains on the expanded synthetic span mix.", "",
             "| Metric | v3 Repeat2 | v4 Repeat2 |", "| --- | ---: | ---: |"]
    locked_old, locked_new = old["v4_human_locked_test"], new["v4_human_locked_test"]
    cal_old, cal_new = old["v4_human_calibration"], new["v4_human_calibration"]
    lines.extend([
        table_line("Calibration threshold (logit margin)", cal_old["threshold"],
                   cal_new["threshold"], lambda x: f"{x:.3f}"),
        table_line("Held-out human token FPR", locked_old["overall"]["fpr"],
                   locked_new["overall"]["fpr"]),
        table_line("Held-out human documents with any false highlight",
                   locked_old["pure_human_document_any_false_highlight_rate"],
                   locked_new["pure_human_document_any_false_highlight_rate"]),
    ])
    for key, title in (("v4_synthetic_val", "V4 synthetic validation"),
                       ("v4_prior_synthetic_val", "Prior synthetic validation"),
                       ("v4_realistic_locked_test", "Realistic mixed-document evaluation")):
        if key in old and key in new:
            a, b = old[key], new[key]
            lines.extend([table_line(f"{title}: AI token recall", a["overall"]["ai_recall"],
                                     b["overall"]["ai_recall"]),
                          table_line(f"{title}: human token FPR", a["overall"]["fpr"],
                                     b["overall"]["fpr"])])
    lines.extend(["", "## Held-out human breakdown", "",
                  "| Domain | v3 token FPR | v4 token FPR | v3 any-highlight | v4 any-highlight | Human documents |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for domain in sorted(set(locked_old["by_domain"]) | set(locked_new["by_domain"])):
        a, b = locked_old["by_domain"].get(domain, {}), locked_new["by_domain"].get(domain, {})
        lines.append(f"| {domain} | {pct(a.get('fpr'))} | {pct(b.get('fpr'))} | "
                     f"{pct(a.get('pure_human_document_any_false_highlight_rate'))} | "
                     f"{pct(b.get('pure_human_document_any_false_highlight_rate'))} | "
                     f"{b.get('pure_human_documents', 0)} |")
    lines.extend(["", "## Human-document length", "",
                  "| Source-token length | v3 any-highlight | v4 any-highlight | Human documents |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for band in sorted(set(locked_old["by_length"]) | set(locked_new["by_length"])):
        a, b = locked_old["by_length"].get(band, {}), locked_new["by_length"].get(band, {})
        lines.append(f"| {band} | {pct(a.get('pure_human_document_any_false_highlight_rate'))} | "
                     f"{pct(b.get('pure_human_document_any_false_highlight_rate'))} | "
                     f"{b.get('pure_human_documents', 0)} |")
    if all(alternate.values()):
        a, b = alternate[OLD], alternate[NEW]
        lines.extend(["", "## Alternate operating point: 2% calibration token FPR", "",
                      "This is a retrospective threshold calculation from cached scores, not",
                      "a new training run. The threshold is set only on pure-human calibration",
                      "tokens and then applied unchanged to the other sets.", "",
                      "| Metric | v3 Repeat2 | v4 Repeat2 |", "| --- | ---: | ---: |",
                      table_line("Threshold (logit margin)", a["threshold"], b["threshold"],
                                 lambda x: f"{x:.3f}")])
        for dataset, title in (("v4_human_locked_test", "Held-out human"),
                               ("v4_synthetic_val", "V4 synthetic"),
                               ("v4_realistic_locked_test", "CoAuthor")):
            if dataset in a["sets"] and dataset in b["sets"]:
                lines.append(table_line(f"{title}: human token FPR",
                                        a["sets"][dataset]["fpr"], b["sets"][dataset]["fpr"]))
                if dataset == "v4_human_locked_test":
                    lines.append(table_line(f"{title}: any false highlight per document",
                                            a["sets"][dataset]["pure_human_doc_any"],
                                            b["sets"][dataset]["pure_human_doc_any"]))
                else:
                    lines.append(table_line(f"{title}: AI token recall",
                                            a["sets"][dataset]["ai_recall"],
                                            b["sets"][dataset]["ai_recall"]))
    n = locked_new["overall"]["pure_human_documents"]
    lines.extend(["", f"The held-out human set has {n:,} documents, with possible clustering",
                  "by author or writing prompt. The table gives descriptive rates rather than",
                  "uncertainty estimates across independent authors.",
                  "", "## Interpretation limits", "",
                  "The v4 training data is built from exact known-origin excerpts, but its long",
                  "mixed documents are synthetic joins. This experiment tests whether that",
                  "training improves localization and reduces false highlights. It does not",
                  "establish performance on reviewed real human edits of AI documents. The",
                  "human test data is independent by record from the span-training parent,",
                  "with provenance and rights limitations described in its manifest. No",
                  "threshold or checkpoint was selected using the locked human set.", ""])
    (args.out / "span_v4_comparison.md").write_text("\n".join(lines))

    with PdfPages(args.out / "span_v4_comparison.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = range(2)
        labels = ["v3", "v4"]
        colors = ["#4C78A8", "#E08E45"]
        for ax, metric, title in ((axes[0], "fpr", "Held-out human token FPR"),
                                  (axes[1], "pure_human_document_any_false_highlight_rate",
                                   "Human documents with any false highlight")):
            values = [locked_old["overall"].get(metric) if metric == "fpr" else locked_old.get(metric),
                      locked_new["overall"].get(metric) if metric == "fpr" else locked_new.get(metric)]
            ax.bar(x, [100 * v for v in values], color=colors)
            ax.set_xticks(list(x), labels)
            ax.set_ylabel("Percent")
            ax.set_title(title)
            for i, v in enumerate(values):
                ax.text(i, 100*v, pct(v), ha="center", va="bottom")
            ax.set_ylim(0, max(1, max(values) * 115))
        fig.suptitle("Repeat2 span pilot: frozen human evaluation")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        if "v4_synthetic_val" in old and "v4_synthetic_val" in new:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for ax, key, title in ((axes[0], "ai_recall", "AI token recall"),
                                   (axes[1], "fpr", "Human token FPR")):
                values = [old["v4_synthetic_val"]["overall"][key],
                          new["v4_synthetic_val"]["overall"][key]]
                ax.bar(x, [100*v for v in values], color=colors)
                ax.set_xticks(list(x), labels)
                ax.set_ylabel("Percent")
                ax.set_title(title)
                for i, v in enumerate(values):
                    ax.text(i, 100*v, pct(v), ha="center", va="bottom")
                ax.set_ylim(0, max(1, max(values)*115))
            fig.suptitle("V4 synthetic validation at separately calibrated thresholds")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if "v4_realistic_locked_test" in old and "v4_realistic_locked_test" in new:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for ax, key, title in ((axes[0], "ai_recall", "AI token recall"),
                                   (axes[1], "fpr", "Human token FPR")):
                values = [old["v4_realistic_locked_test"]["overall"][key],
                          new["v4_realistic_locked_test"]["overall"][key]]
                ax.bar(x, [100*v for v in values], color=colors)
                ax.set_xticks(list(x), labels)
                ax.set_ylabel("Percent")
                ax.set_title(title)
                for i, v in enumerate(values):
                    ax.text(i, 100*v, pct(v), ha="center", va="bottom")
                ax.set_ylim(0, max(1, max(values)*115))
            fig.suptitle("CoAuthor human–AI writing: frozen evaluation")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(args.out / "span_v4_comparison.md")


if __name__ == "__main__":
    main()
