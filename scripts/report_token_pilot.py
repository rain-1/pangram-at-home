"""Summarize the three-model binary token localization pilot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home")
MODELS = [
    ("Vast window", "vast_hpo_selected_v3", "#146C94"),
    ("Token single", "qwen3_token_single_v3_pilot1", "#54A24B"),
    ("Token Repeat2", "qwen3_token_repeat2_v3_pilot1", "#E08E45"),
]
SETS = [("Validation", "span_validation_v3"),
        ("Confirmation", "span_test_v1"),
        ("Long composite", "span_long_v1")]


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def main():
    reports = {}
    for label, name, _ in MODELS:
        run = ROOT / "runs" / name
        reports[label] = {set_label: json.loads((run / f"{stem}.json").read_text())
                          for set_label, stem in SETS}
    validation_hashes = {reports[label]["Validation"]["validation_sha256"]
                         for label, _, _ in MODELS}
    assert len(validation_hashes) == 1
    for set_label, _ in SETS:
        assert len({reports[label][set_label]["validation_sha256"]
                    for label, _, _ in MODELS}) == 1
    (REPO / "reports/metrics/token_span_pilot_v3_summary.json").write_text(
        json.dumps({"role": "aggregate synthetic token-pilot diagnostics; no raw text",
                    "models": reports}, indent=2) + "\n")
    data_manifest = json.loads((ROOT / "data/span_pilot_v3/manifest.json").read_text())
    test_manifest = json.loads((ROOT / "data/span_test_probe_v1/manifest.json").read_text())
    long_manifest = json.loads((ROOT / "data/span_long_probe_v1/manifest.json").read_text())
    lines = [
        "# Binary token-localization pilot",
        "",
        "The models predict **human** or **AI-generated** at each token. They do not predict AI-assisted text. The token models use a Qwen3-1.7B backbone with the selected Vast learning rate 7.607757e-5, LoRA rank 32/alpha 64/dropout 0.068837, attention and feed-forward targets, BF16, microbatch 2 × accumulation 4, 512 source tokens, 256 stride, and 300 steps. Both start with the same selected Vast backbone adapter; their token heads are newly initialized. Repeat2 masks all first-copy token labels and supervises the second copy. W&B logged both runs.",
        "",
        f"Data: {data_manifest['splits']['train']['documents']} synthetic training documents and {data_manifest['splits']['val']['documents']} validation documents, with {test_manifest['documents']} source-group-disjoint confirmation documents and {long_manifest['documents']} long composites. Training and validation cover paper, reviews, creative writing, news, reference/education, and social Q&A. Joins use exact character offsets and varied separators; the underlying source text stays on the external drive.",
        "",
        "| Model | Input / output | W&B |",
        "|---|---|---|",
        "| Vast window | Single 512-token input; one score per window broadcast over its tokens | Prior HPO run |",
        "| Token single | One input copy; token scores | [run](https://wandb.ai/eac-adsf/pangram-at-home/runs/2jrklgmg) |",
        "| Token Repeat2 | Two input copies; second-copy token scores | [run](https://wandb.ai/eac-adsf/pangram-at-home/runs/053oc7dr) |",
        "",
        "## Whole-document scores at frozen thresholds",
        "",
        "Each model's threshold is set to 2% false positives on pure-human *validation tokens*, then held fixed for the confirmation and long-document diagnostics. FPR and recall below include all labeled tokens; AI-span recall asks whether at least half the characters in a known AI span are highlighted.",
        "",
        "| Set | Model | Human token FPR | AI token recall | Mixed AI spans ≥half covered | Fully human docs with any false highlight | Mixed AI-fraction MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for set_label, _ in SETS:
        for label, _, _ in MODELS:
            item = reports[label][set_label]
            lines.append("| " + " | ".join([
                set_label, label, percent(item["overall"]["fpr"]),
                percent(item["overall"]["ai_recall"]),
                percent(item["mixed_ai_span_recall_half_covered"]),
                percent(item["pure_human_document_any_false_highlight_rate"]),
                f'{item["mixed_document_ai_character_fraction_mae"]:.3f}',
            ]) + " |")
    audit = json.loads((REPO / "reports/metrics/token_threshold_audit_v3.json").read_text())
    lines += ["", "## Stricter document-safe operating point", "",
              "For each token model, a second threshold allows at most 3 of 67 fully human validation documents to receive any highlight (4.5%). It is frozen for the other sets. This is a development calibration, not a guarantee on new writing.",
              "", "| Set | Token model | Human token FPR | AI token recall | Fully human docs with any false highlight |",
              "|---|---|---:|---:|---:|"]
    audit_labels = {"Validation": "validation", "Confirmation": "confirmation",
                    "Long composite": "long_composite"}
    for set_label, key in audit_labels.items():
        for label, name, _ in MODELS[1:]:
            item = audit["models"][name]["doc_5pct"][key]
            lines.append("| " + " | ".join([
                set_label, label, percent(item["human_token_fpr"]),
                percent(item["ai_token_recall"]),
                percent(item["pure_human_documents_with_any_false_highlight"]),
            ]) + " |")
    lines += ["", "## Confirmation-set source construction", "",
              "Matched pairs share a source group/prompt where available. Unmatched same-source joins also contain MAGE sources absent from the matched stratum, so construction and source family are confounded.",
              "", "| Construction | Model | Human FPR | AI recall | Documents |",
              "|---|---|---:|---:|---:|"]
    for construction in ["matched_source_pair", "unaltered_source_excerpt",
                         "unmatched_same_source_join"]:
        for label, _, _ in MODELS:
            item = reports[label]["Confirmation"]["by_construction"][construction]
            lines.append("| " + " | ".join([construction, label,
                percent(item["fpr"]), percent(item["ai_recall"]), str(item["documents"])]) + " |")
    lines += ["", "### Construction × source family", "",
              "| Construction / family | Model | Human FPR | AI recall | Documents |",
              "|---|---|---:|---:|---:|"]
    strata = sorted(reports["Vast window"]["Confirmation"]["by_construction_family"])
    for stratum in strata:
        for label, _, _ in MODELS:
            item = reports[label]["Confirmation"]["by_construction_family"][stratum]
            lines.append("| " + " | ".join([stratum, label,
                percent(item["fpr"]), percent(item["ai_recall"]), str(item["documents"])]) + " |")
    lines += ["", "## Confirmation-set domains", "",
              "| Domain | Model | Human FPR | AI recall | Documents |",
              "|---|---|---:|---:|---:|"]
    domains = sorted(reports["Vast window"]["Confirmation"]["by_domain"])
    for domain in domains:
        for label, _, _ in MODELS:
            item = reports[label]["Confirmation"]["by_domain"][domain]
            lines.append("| " + " | ".join([domain, label,
                percent(item["fpr"]), percent(item["ai_recall"]), str(item["documents"])]) + " |")
    lines += ["", "## Learning curve", "",
              "| Token model | Best step | Best validation partial AUROC ≤5% FPR | Training minutes | Peak PyTorch allocation |",
              "|---|---:|---:|---:|---:|"]
    for label, name, _ in MODELS[1:]:
        summary = json.loads((ROOT / "runs" / name / "train_summary.json").read_text())
        best_step = int(Path(summary["best_checkpoint"]).name.split("-")[-1])
        lines.append(f'| {label} | {best_step} | {summary["best_metric"]:.3f} | {summary["train_runtime_seconds"]/60:.1f} | {summary["peak_allocated_gb"]:.2f} GiB |')
    repeat_threshold = audit["models"]["qwen3_token_repeat2_v3_pilot1"]["threshold_doc_5pct"]
    lines += [
        "", "## Run inference", "",
        "`scripts/predict_token_spans.py` reads a UTF-8 text file, scores overlapping 512-token windows, averages token logits, and writes character-offset human/AI spans. For this pilot's stricter document threshold:",
        "", "```bash",
        f"python scripts/predict_token_spans.py --run-name qwen3_token_repeat2_v3_pilot1 --text-file /path/to/document.txt --threshold {repeat_threshold} --output /path/to/spans.json",
        "```",
        "", "## What this pilot can establish", "",
        "All mixed spans have known origin because they were assembled from separately labeled human and AI source text with recorded character offsets. The joins are synthetic and may reveal formatting or source cues. The confirmation set is source-group-disjoint from pilot train/validation, but its parent diverse test was examined in earlier passage-model work, so it is not a blind final test. The long composite set reuses validation excerpts and tests window stitching and length effects only. The Vast hyperparameters were tuned for passage classification, not this token objective. A realistic reviewed span set remains necessary before claiming real-world localization quality.",
        "", "Raw text, token scores, and predicted spans stay on the external drive. This report contains aggregate metrics only.",
    ]
    target = REPO / "reports/token_span_pilot_v3.md"
    target.write_text("\n".join(lines) + "\n")

    with PdfPages(REPO / "reports/token_span_pilot_v3.pdf") as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), layout="constrained")
        groups = ["Confirmation", "Long composite"]
        y = np.arange(len(groups))
        width = .23
        for axis, metric, title in [(axes[0], "fpr", "Human token false-positive rate"),
                                    (axes[1], "ai_recall", "AI token recall")]:
            for i, (label, _, color) in enumerate(MODELS):
                values = [100 * reports[label][group]["overall"][metric] for group in groups]
                bars = axis.barh(y + (i - 1) * width, values, height=width,
                                 color=color, label=label)
                axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
            axis.set_yticks(y, groups)
            axis.invert_yaxis()
            axis.set_title(title)
            axis.set_xlabel("Percent")
            axis.grid(axis="x", alpha=.2)
            axis.legend(loc="lower right", ncol=3, fontsize=8)
        axes[0].set_xlim(0, max(10, max(100 * reports[label][group]["overall"]["fpr"]
                                    for label, _, _ in MODELS for group in groups) * 1.25))
        axes[1].set_xlim(0, 110)
        fig.suptitle("Token pilot versus the Vast sliding-window baseline", fontsize=15)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        for axis, set_label, stem in zip(axes, ["Confirmation", "Long composite"],
                                         ["span_test_v1", "span_long_v1"]):
            off_scale = []
            for label, name, color in MODELS:
                cache = np.load(ROOT / "runs" / name / f"{stem}_scores.npz")
                fpr, tpr, _ = roc_curve(cache["label"], cache["score"])
                axis.plot(100 * fpr, 100 * tpr, color=color, label=label)
                item = reports[label][set_label]
                point_fpr = 100 * item["overall"]["fpr"]
                if point_fpr <= 20:
                    axis.scatter(point_fpr, 100 * item["overall"]["ai_recall"],
                                 color=color, s=30)
                else:
                    off_scale.append(f"{label} point: {point_fpr:.1f}% FPR")
            axis.set(xlim=(0, 20), ylim=(0, 102), title=set_label,
                     xlabel="Human token FPR (%)", ylabel="AI token recall (%)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            if off_scale:
                axis.text(.98, .04, "; ".join(off_scale), transform=axis.transAxes,
                          ha="right", va="bottom", fontsize=8)
        fig.suptitle("Token-level ROC; dots are frozen validation-threshold points", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        for label, name, color in MODELS[1:]:
            path = ROOT / "runs" / name / "sweep_metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            axis.plot([row["step"] for row in rows],
                      [row["eval_partial_auc_fpr_5pct"] for row in rows],
                      marker="o", color=color, label=label)
        axis.set(xlabel="Optimizer step", ylabel="Validation partial AUROC at ≤5% FPR",
                 xlim=(0, 310), ylim=(0.5, 1.0))
        axis.grid(alpha=.2)
        axis.legend()
        fig.suptitle("Token pilot learning curves", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
