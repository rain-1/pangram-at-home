"""Compare frozen v4 scores on two real co-writing datasets without refitting.

This reads external private text and cached scores, and writes aggregate results
only. The score threshold remains the one selected on the separate human
calibration set. Retrospective ROC points describe ranking, not new thresholds
selected for deployment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer

from span_data import encode_document


ROOT = Path("/mnt/f/pangram-at-home")
RUN = ROOT / "runs/qwen3_token_repeat2_v4_pilot1"
OUT = Path(__file__).resolve().parents[1] / "reports"
BINS = ((0, 20, "<20"), (20, 80, "20–79"),
        (80, 320, "80–319"), (320, 10**12, "320+"))
SOURCES = {
    "CoAuthor": (ROOT / "data/span_realistic_eval_v1/test.jsonl", "v4_realistic_locked_test"),
    "AITDNA": (ROOT / "data/span_sources_v5/normalized_aitdna_real/locked_test.jsonl", "aitdna_locked_v1"),
}


def percent(x):
    return "—" if x is None else f"{100*x:.1f}%"


def span_bin(length):
    return next(name for lo, hi, name in BINS if lo <= length < hi)


def analyze(name, source, stem, tokenizer):
    rows = [json.loads(line) for line in source.open()]
    report = json.loads((RUN / f"{stem}.json").read_text())
    threshold = report["threshold"]
    with np.load(RUN / f"{stem}_scores.npz") as cache:
        score = cache["score"].astype(np.float64)
        label = cache["label"].copy()
        doc_offsets = cache["document_offsets"].copy()
        doc_ids = cache["document_ids"].tolist()
    assert len(rows) == len(doc_offsets) - 1 == len(doc_ids)
    assert [r["id"] for r in rows] == doc_ids
    assert report["overall"]["tokens"] == len(score)
    pred = score >= threshold

    mixed_slices = [(a, b) for row, a, b in zip(rows, doc_offsets[:-1], doc_offsets[1:])
                    if row["kind"] == "mixed"]
    writer_groups = {group for row in rows for group in
                     (row.get("source_groups") or [row.get("author_group")]) if group}
    mixed_idx = np.concatenate([np.arange(a, b) for a, b in mixed_slices])
    mixed_score, mixed_label = score[mixed_idx], label[mixed_idx]
    fpr, tpr, _ = roc_curve(mixed_label, mixed_score)
    retrospective = {}
    for target in (.01, .02, .05, .10):
        # Highest available threshold with FPR at or below target; descriptive only.
        keep = np.flatnonzero(fpr <= target)
        i = keep[-1]
        retrospective[str(target)] = {"realized_fpr": float(fpr[i]),
                                       "ai_token_recall": float(tpr[i])}

    bins = defaultdict(Counter)
    docs_with_ai = docs_detected = docs_zero = 0
    for row, start, end in zip(rows, doc_offsets[:-1], doc_offsets[1:]):
        ids, offsets, labels = encode_document(row, tokenizer)
        labels = np.asarray(labels)
        valid = labels != -100
        assert valid.sum() == end - start and np.array_equal(labels[valid], label[start:end])
        valid_offsets = np.asarray(offsets)[valid]
        local_score = score[start:end]
        local_pred = pred[start:end]
        local_label = label[start:end]
        if (local_label == 1).any():
            docs_with_ai += 1
            any_hit = bool(np.any(local_pred[local_label == 1]))
            docs_detected += any_hit
            docs_zero += not any_hit
        for span in row["spans"]:
            if span["label"] != 1:
                continue
            length = span["end"] - span["start"]
            bucket = bins[span_bin(length)]
            bucket["spans"] += 1
            bucket["characters"] += length
            within = ((valid_offsets[:, 0] >= span["start"]) &
                      (valid_offsets[:, 1] <= span["end"]) & (local_label == 1))
            n = int(within.sum())
            bucket["scored_tokens"] += n
            bucket["detected_tokens"] += int(local_pred[within].sum())
            bucket["spans_with_scored_tokens"] += n > 0
            bucket["spans_with_any_highlight"] += bool(np.any(local_pred[within]))
            bucket["spans_half_covered"] += bool(n > 0 and local_pred[within].mean() >= .5)
    total_ai_chars = sum(b["characters"] for b in bins.values())
    by_span_length = {}
    for _, _, key in BINS:
        b = bins[key]
        by_span_length[key] = {
            **dict(b),
            "ai_character_share": b["characters"] / total_ai_chars,
            "scorable_span_share": b["spans_with_scored_tokens"] / b["spans"] if b["spans"] else None,
            "token_recall_at_frozen_threshold": b["detected_tokens"] / b["scored_tokens"] if b["scored_tokens"] else None,
            "any_highlight_span_recall": b["spans_with_any_highlight"] / b["spans_with_scored_tokens"] if b["spans_with_scored_tokens"] else None,
        }
    return {
        "source": name,
        "source_file": str(source),
        "cache_file": str(RUN / f"{stem}_scores.npz"),
        "documents": len(rows),
        "mixed_documents": len(mixed_slices),
        "writer_groups": len(writer_groups),
        "labeled_token_fraction": report["labeled_token_fraction"],
        "frozen_threshold": threshold,
        "overall": report["overall"],
        "by_domain": report["by_domain"],
        "mixed": {"ai_tokens": int((mixed_label == 1).sum()),
                  "human_tokens": int((mixed_label == 0).sum()),
                  "ai_token_fraction": float((mixed_label == 1).mean()),
                  "auroc": float(roc_auc_score(mixed_label, mixed_score)),
                  "ai_token_recall": float(pred[mixed_idx][mixed_label == 1].mean()),
                  "human_token_fpr": float(pred[mixed_idx][mixed_label == 0].mean())},
        "documents_with_ai": docs_with_ai,
        "documents_with_any_ai_token_detected": docs_detected,
        "documents_with_zero_ai_tokens_detected": docs_zero,
        "retrospective_mixed_roc": retrospective,
        "by_ai_span_character_length": by_span_length,
        "roc_arrays": (fpr, tpr),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(RUN / "best_adapter")
    results = {name: analyze(name, *paths, tokenizer) for name, paths in SOURCES.items()}
    OUT.mkdir(exist_ok=True)
    clean = {name: {k: v for k, v in result.items() if k != "roc_arrays"}
             for name, result in results.items()}
    (OUT / "real_collaboration_v4.json").write_text(json.dumps(clean, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"CoAuthor": "#4C78A8", "AITDNA": "#E08E45"}
    for name, result in results.items():
        fpr, tpr = result["roc_arrays"]
        axes[0].plot(100*fpr, 100*tpr, label=f"{name} (AUROC {result['mixed']['auroc']:.3f})",
                     color=colors[name], linewidth=2)
    axes[0].plot([0, 20], [0, 20], linestyle="--", color="#999999", linewidth=1)
    axes[0].set(xlim=(0, 20), ylim=(0, 100), xlabel="Human token false positive rate (%)",
                ylabel="AI token recall (%)", title="Mixed documents: ranking across thresholds")
    axes[0].legend(frameon=False)
    keys = [x[2] for x in BINS]
    x = np.arange(len(keys))
    width = .36
    for offset, name in ((-width/2, "CoAuthor"), (width/2, "AITDNA")):
        vals = [100*results[name]["by_ai_span_character_length"][key]["ai_character_share"] for key in keys]
        axes[1].bar(x+offset, vals, width, color=colors[name], label=name)
    axes[1].set(xticks=x, xticklabels=keys, ylim=(0, 100), xlabel="AI span length (characters)",
                ylabel="Share of AI characters (%)", title="What the token recall is weighted toward")
    axes[1].legend(frameon=False)
    fig.suptitle("Repeat2 v4 on real human–AI writing", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "real_collaboration_v4.pdf")
    plt.close(fig)

    lines = ["# Real collaboration: where the v4 model succeeds and fails", "",
             "All figures below are from cached scores. The frozen operating threshold was selected on a separate pure-human calibration set; neither benchmark changed model weights or threshold.", "",
             "| Measure | CoAuthor | AITDNA |", "| --- | ---: | ---: |"]
    for label, key, fmt in (
        ("Documents", "documents", str),
        ("Mixed documents", "mixed_documents", str),
        ("Writer groups", "writer_groups", str),
        ("Source tokens with usable labels", "labeled_token_fraction", percent),
    ):
        lines.append(f"| {label} | {fmt(results['CoAuthor'][key])} | {fmt(results['AITDNA'][key])} |")
    for label, key in (("AI share of labeled tokens in mixed documents", "ai_token_fraction"),
                       ("AI token recall in mixed documents", "ai_token_recall"),
                       ("Human token false positive rate in mixed documents", "human_token_fpr")):
        lines.append(f"| {label} | {percent(results['CoAuthor']['mixed'][key])} | {percent(results['AITDNA']['mixed'][key])} |")
    lines.append(f"| Token AUROC in mixed documents | {results['CoAuthor']['mixed']['auroc']:.3f} | {results['AITDNA']['mixed']['auroc']:.3f} |")
    for label, key in (("Mixed documents with any AI token detected", "documents_with_any_ai_token_detected"),
                       ("Mixed documents with no AI token detected", "documents_with_zero_ai_tokens_detected")):
        lines.append(f"| {label} | {results['CoAuthor'][key]}/{results['CoAuthor']['documents_with_ai']} | {results['AITDNA'][key]}/{results['AITDNA']['documents_with_ai']} |")
    lines += ["", "The frozen threshold gives CoAuthor 3.3% AI-token recall (233/7,026) and 0.35% human-token FPR (86/24,306). AITDNA's corresponding overall numbers are 88.4% (86,520/97,824) and 9.3% (6,974/75,094). AITDNA contains 103 pure-human documents, of which one has a false highlight. CoAuthor contains no pure-human documents in this evaluation.",
              "", "Within CoAuthor, argumentative writing has 0/1,777 AI tokens detected across 36 documents; creative writing has 233/5,249 (4.4%) across 83. These are sessions from 25 writers, so document counts should not be treated as independent writer counts.",
              "", "## Ranking at matched false positive rates", "",
              "These are **retrospective** points on each benchmark's mixed documents, with a separate threshold for each benchmark. They show score separation, not deployable calibration or a single common operating point.", "",
              "| Within-benchmark human-token FPR | CoAuthor AI recall | AITDNA AI recall |", "| ---: | ---: | ---: |"]
    for target in (.01, .02, .05, .10):
        key = str(target)
        lines.append(f"| {percent(target)} | {percent(results['CoAuthor']['retrospective_mixed_roc'][key]['ai_token_recall'])} | {percent(results['AITDNA']['retrospective_mixed_roc'][key]['ai_token_recall'])} |")
    lines += ["", "## AI span length", "",
              "The length figures describe the source's surviving AI-character spans. Token recall weights long spans much more heavily than short ones.", "",
              "| Span length | CoAuthor AI spans | CoAuthor share of AI characters | CoAuthor token recall | AITDNA AI spans | AITDNA share of AI characters | AITDNA token recall |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, _, key in BINS:
        a = results["CoAuthor"]["by_ai_span_character_length"][key]
        b = results["AITDNA"]["by_ai_span_character_length"][key]
        lines.append(f"| {key} characters | {a['spans']} | {percent(a['ai_character_share'])} | {percent(a['token_recall_at_frozen_threshold'])} | {b['spans']} | {percent(b['ai_character_share'])} | {percent(b['token_recall_at_frozen_threshold'])} |")
    lines += ["", "Some very short character spans contain no complete model token and cannot contribute to the span-bin token recall. The JSON records the number of scorable spans in each bin.",
              "", "## Interpretation", "",
              "CoAuthor is an observed GPT-3 era collaboration corpus with relatively short surviving AI inserts. AITDNA uses recent generators and contains long AI passages alongside tiny edits. In AITDNA, 81.5% of AI characters are in spans of at least 320 characters; in CoAuthor, only 7.4% are. That difference, along with generator, task, participant and provenance differences, plausibly contributes to the large recall gap. It is not a controlled causal test.",
              "", "The mixed-document AUROC gap (0.676 versus 0.918) persists across thresholds. Therefore CoAuthor's low recall cannot be explained solely by choosing a conservative threshold. The model's ranking of short, integrated AI inserts is substantially weaker on this benchmark. Neither benchmark alone estimates broad real-world accuracy.",
              "", "[ROC and span-length chart](real_collaboration_v4.pdf). Source descriptions: [CoAuthor](https://coauthor.stanford.edu/) and [AITDNA](https://huggingface.co/datasets/UKPLab/AITDNA).", ""]
    (OUT / "real_collaboration_v4.md").write_text("\n".join(lines))
    print(OUT / "real_collaboration_v4.md")


if __name__ == "__main__":
    main()
