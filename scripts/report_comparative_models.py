"""Summarize the frozen, validation-threshold comparison of three passage models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
NAMES = ["vast_hpo_selected_v3", "qwen3_hpo_single_local_v1", "qwen3_hpo_repeat2_local_v1"]
LABELS = ["Vast single", "Local single", "Local Repeat2"]
COLORS = ["#146C94", "#54A24B", "#E08E45"]
BASE_SPLITS = ["test", "raid_external", "enron_external", "gpt4_ood", "paraphrase",
               "standard_ebooks_human", "persuade_essays_human", "federal_reserve_human",
               "stackexchange_writers_human", "pmc_full_body_human"]
NEW_SPLITS = ["editlens_original", "editlens_llama", "mage_main", "paper_generator_swap"]
DISPLAY = {
    "test": "Diverse held-out", "raid_external": "RAID", "enron_external": "Enron",
    "gpt4_ood": "GPT-4", "paraphrase": "Paraphrased AI",
    "standard_ebooks_human": "Standard Ebooks", "persuade_essays_human": "Persuade essays",
    "federal_reserve_human": "Federal Reserve", "stackexchange_writers_human": "Stack Exchange",
    "pmc_full_body_human": "PMC full body", "editlens_original": "EditLens original",
    "editlens_llama": "EditLens Llama", "mage_main": "MAGE holdout",
    "paper_generator_swap": "Paper generator swap",
}


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def main():
    reports = []
    thresholds = []
    for name in NAMES:
        base = json.loads((ROOT / "reports/metrics" / f"{name}.json").read_text())
        extra = json.loads((ROOT / "reports/metrics" / f"{name}_comparative_ood_v1.json").read_text())
        assert base["adapter_sha256"] == extra["adapter_sha256"]
        assert base["threshold"] == extra["threshold"]
        thresholds.append(base["threshold"])
        reports.append({**base["splits"], **extra["splits"]})
    splits = BASE_SPLITS + NEW_SPLITS
    lines = [
        "# Wider comparison of the three Qwen passage classifiers",
        "",
        "All figures use each model's frozen threshold chosen on the same diverse validation split for nominal 2% human false-positive rate. No threshold was tuned on these evaluation sets. Scores refer to the first 512 source tokens (Repeat2 sees two copies of those tokens).",
        "",
        "| Model | Quantization | Microbatch × accumulation | Effective batch | Validation logit-margin threshold |",
        "|---|---|---:|---:|---:|",
        f"| Vast single | BF16, unquantized | 2 × 4 | 8 | {thresholds[0]:.3f} |",
        f"| Local single | BF16, unquantized | 1 × 8 | 8 | {thresholds[1]:.3f} |",
        f"| Local Repeat2 | BF16, unquantized | 1 × 8 | 8 | {thresholds[2]:.3f} |",
        "",
        "All three runs share the same training and validation file hashes, seed 42, learning rate, LoRA rank/dropout, 512 source-token cap, and 3,200 optimizer-step budget. The local microbatch change was made without a memory probe. It preserves effective batch but means Vast versus local single is not an exact training replication. The binary sequence-classification Repeat2 run also does not implement [Pangram 4's tokenwise Repeat2 objective](https://pangram-public.s3.us-east-1.amazonaws.com/pdf/pangram_4_technical_report.pdf).",
        "",
        "## Local microbatch-2 memory check",
        "",
    ]
    memory = json.loads((ROOT / "reports/metrics/local_microbatch2_probe.json").read_text())
    lines += [
        "A subsequent one-step BF16 LoRA backward and AdamW optimizer probe on the 16 GiB RTX 4080 succeeded with the sweep's microbatch 2 for both input lengths. The probe used synthetic maximum-length inputs; it does not establish sustained-run memory behavior, but the original microbatch reduction had no measured need.",
        "",
        "| Input mode | Model tokens/example | PyTorch peak reserved |",
        "|---|---:|---:|",
        f"| Single copy | 512 | {memory['single']['peak_reserved_gib']:.2f} GiB |",
        f"| Repeat2 | 1024 | {memory['repeat2']['peak_reserved_gib']:.2f} GiB |",
        "",
        "## Results at the fixed operating point",
        "",
        "False-positive rate (FPR) measures human text incorrectly called AI; AI recall measures AI text correctly called AI. A dash means the set has only human examples.",
        "",
        "| Dataset | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        first = reports[0][split]
        assert all((r[split]["human"], r[split]["ai"]) == (first["human"], first["ai"]) for r in reports)
        parts = [DISPLAY[split], f'{first["human"]} / {first["ai"]}']
        parts.extend(pct(r[split]["fpr"]) for r in reports)
        parts.extend(pct(r[split]["tpr"]) for r in reports)
        lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "## Ranking quality (AUROC)", "",
              "AUROC uses every possible threshold; partial AUROC focuses on the 0–5% false-positive region. Neither replaces the fixed-threshold error rates above.",
              "", "| Dataset | Vast AUROC / partial | Local single AUROC / partial | Repeat2 AUROC / partial |",
              "|---|---:|---:|---:|"]
    for split in splits:
        if reports[0][split]["ai"] == 0:
            continue
        parts = [DISPLAY[split]]
        for report in reports:
            item = report[split]
            parts.append(f'{item["roc_auc"]:.3f} / {item["partial_auc_fpr_5pct"]:.3f}')
        lines.append("| " + " | ".join(parts) + " |")
    lines += [
        "",
        "The two EditLens sets reuse source prompts and must not be pooled as independent evidence. For the four new probes, source IDs and normalized exact texts occurring in any diverse train, validation, or test split were removed. MAGE source IDs are hashes of individual texts, so this exclusion establishes exact-text separation but not independent prompts or authors. The EditLens Llama probe includes only `human_written` and `ai_generated`; assisted edits were excluded because the current models are binary classifiers. MAGE contains some closed-ended source tasks and strongly mismatched human/AI lengths; interpret its source breakdown and aggregate with those caveats.",
        "",
        "## Source breakdown on the diverse held-out set",
        "",
        "| Source | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sources = sorted(reports[0]["test"]["by_source"])
    for source in sources:
        values = [r["test"]["by_source"][source] for r in reports]
        first = values[0]
        parts = [source, f'{first["human"]} / {first["ai"]}']
        parts.extend(pct(v["fpr"]) for v in values)
        parts.extend(pct(v["tpr"]) for v in values)
        lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "## New-probe source breakdown", "",
              "| Dataset / source | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in NEW_SPLITS:
        for source in sorted(reports[0][split]["by_source"]):
            values = [r[split]["by_source"][source] for r in reports]
            first = values[0]
            parts = [f"{DISPLAY[split]} / {source}", f'{first["human"]} / {first["ai"]}']
            parts.extend(pct(v["fpr"]) for v in values)
            parts.extend(pct(v["tpr"]) for v in values)
            lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "## Length breakdown on new probes", "",
              "| Dataset / characters | Human / AI | Vast FPR | Local single FPR | Repeat2 FPR | Vast AI recall | Local single AI recall | Repeat2 AI recall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ["mage_main", "editlens_original"]:
        for length in ["under_500_chars", "500_to_999_chars", "1000_to_1999_chars", "2000_plus_chars"]:
            values = [r[split]["by_length"].get(length) for r in reports]
            if values[0] is None:
                continue
            first = values[0]
            parts = [f"{DISPLAY[split]} / {length}", f'{first["human"]} / {first["ai"]}']
            parts.extend(pct(v["fpr"]) for v in values)
            parts.extend(pct(v["tpr"]) for v in values)
            lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "## Paired human-error counts", "",
              "These counts use the same human rows for each pair. `A only` means model A falsely called AI while model B did not; `B only` is the reverse.",
              "", "| Dataset | Pair (A / B) | A only | B only |",
              "|---|---|---:|---:|"]
    for split in ["test", "mage_main", "raid_external", "paraphrase"]:
        wrong = []
        first_labels = None
        first_input_sha = None
        for i, name in enumerate(NAMES):
            folder = "comparative_ood_v1" if split in NEW_SPLITS else "final_eval"
            cache = np.load(Path("/mnt/f/pangram-at-home/runs") / name / folder / f"{split}.npz")
            if first_labels is None:
                first_labels = cache["label"]
                first_input_sha = str(cache["input_sha256"])
            else:
                assert np.array_equal(cache["label"], first_labels)
                assert str(cache["input_sha256"]) == first_input_sha
            wrong.append((cache["margin"] >= thresholds[i])[cache["label"] == 0])
        assert all(len(row) == len(wrong[0]) for row in wrong)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            only_a = int(np.sum(wrong[a] & ~wrong[b]))
            only_b = int(np.sum(wrong[b] & ~wrong[a]))
            lines.append(f"| {DISPLAY[split]} | {LABELS[a]} / {LABELS[b]} | {only_a} | {only_b} |")
    lines += [
        "", "## Interpretation", "",
        "The main diverse held-out set contains only 500 human examples, so a few additional false positives move FPR by whole percentage points. Reported gaps should be read alongside the larger human-only sets and the per-source rows. Very high scores on datasets with similar generation pipelines are weaker evidence of real-world generalization than the harder RAID and paraphrase sets.",
        "",
        "On MAGE, under-500-character human FPR is " + ", ".join(
            f"{label} {pct(report['mage_main']['by_length']['under_500_chars']['fpr'])}"
            for label, report in zip(LABELS, reports)) + ". The HSwag subset contains short, formatted human passages and accounts for " + ", ".join(
            f"{label} {report['mage_main']['by_source']['hswag']['fp']}"
            for label, report in zip(LABELS, reports)) + " human false positives out of 200 per model. This is an observed concentration, not proof that length or formatting alone causes the errors.",
        "",
        "Paraphrased AI recall is " + ", ".join(
            f"{label} {pct(report['paraphrase']['tpr'])}"
            for label, report in zip(LABELS, reports)) + "; this remains a major false-negative weakness even where ordinary AI recall is high.",
        "", "The current window classifier provides one score per 512-token window. It does not label tokens or identify exact authorship boundaries. On long documents, overlapping windows can produce a coarse heatmap; full-paper human audits are in `reports/metrics/*_pmc_fullpaper.json`.",
        "", "## Full-paper human-only audit", "",
        "The same 100 source-disjoint PMC papers contain 685,096 source tokens. Windows use size 512 and stride 256; a token is falsely highlighted if a covering window crosses that model's frozen threshold.",
        "", "| Model | Human tokens falsely highlighted | Papers with any false highlight |",
        "|---|---:|---:|",
    ]
    for label, name in zip(LABELS, NAMES):
        audit = json.loads((ROOT / "reports/metrics" / f"{name}_pmc_fullpaper.json").read_text())
        lines.append(f'| {label} | {pct(audit["token_false_highlight_rate"])} | {audit["documents_with_any_false_highlight"]} / {audit["documents"]} |')
    lines += [
        "", "## Provenance", "",
    ]
    for split in NEW_SPLITS:
        item = reports[0][split]
        lines.append(f'- {DISPLAY[split]}: {item["rows"]} rows after filtering; {item["excluded_source_overlap_rows"]} source-overlap and {item["excluded_exact_text_overlap_rows"]} exact-text-overlap rows removed; {item["source_ids"]} source IDs; SHA256 `{item["sha256"]}`.')
    target = ROOT / "reports/comparative_models_v1.md"
    target.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), layout="constrained")
    width = .24
    for axis, metric, title in [(axes[0], "fpr", "Human false-positive rate"),
                                (axes[1], "tpr", "AI recall")]:
        plotted = splits if metric == "fpr" else [s for s in splits if reports[0][s]["ai"] > 0]
        y = np.arange(len(plotted))
        for i, report in enumerate(reports):
            vals = [100 * report[s][metric] for s in plotted]
            axis.barh(y + (i - 1) * width, vals, height=width, color=COLORS[i], label=LABELS[i])
        axis.set_yticks(y, [DISPLAY[s] for s in plotted])
        axis.invert_yaxis()
        axis.set_xlabel("Percent")
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
        axis.legend(loc="lower right", ncol=3, fontsize=8)
    axes[0].set_xlim(0, max(6, max(100 * r[s]["fpr"] for r in reports for s in splits) * 1.15))
    axes[1].set_xlim(0, 105)
    fig.suptitle("Three Qwen classifiers at their frozen validation thresholds", fontsize=15)
    with PdfPages(ROOT / "reports/comparative_models_v1.pdf") as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for axis, split in zip(axes.flat, ["test", "mage_main", "raid_external", "paraphrase"]):
            for i, name in enumerate(NAMES):
                folder = "comparative_ood_v1" if split in NEW_SPLITS else "final_eval"
                cache = np.load(Path("/mnt/f/pangram-at-home/runs") / name / folder / f"{split}.npz")
                fpr, tpr, _ = roc_curve(cache["label"], cache["margin"])
                axis.plot(100 * fpr, 100 * tpr, label=LABELS[i], color=COLORS[i])
                axis.scatter(100 * reports[i][split]["fpr"], 100 * reports[i][split]["tpr"],
                             color=COLORS[i], s=30, zorder=3)
            axis.set(xlim=(0, 20), ylim=(0, 102), title=DISPLAY[split],
                     xlabel="Human false-positive rate (%)", ylabel="AI recall (%)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.suptitle("Low-FPR ROC curves; dots are frozen validation-threshold operating points", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
