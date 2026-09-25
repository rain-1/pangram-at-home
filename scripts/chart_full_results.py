"""Render all saved detector baselines and the first trained model as a chart."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
OUT = ROOT / "reports" / "charts"


def load(name: str):
    return json.loads((METRICS / f"{name}.json").read_text())


trained = load("segment_qwen3_17b_mixed_stage1_v1")


def row(name, file, key="test", kind="baseline", source_name=None):
    value = load(file)[key]
    if source_name:
        value = value["by_source"][source_name]
    return dict(name=name, source=file, split=key + (":" + source_name if source_name else ""), kind=kind, ai=value.get("ai"), tp=value.get("tp"), human=value.get("human", value.get("rows")), fp=value.get("fp", value.get("false_positives")))


def trained_row(name, key="test", source_name=None):
    value = trained[key]
    if source_name:
        value = value["by_source"][source_name]
    return dict(name=name, source="segment_qwen3_17b_mixed_stage1_v1", split=key + (":" + source_name if source_name else ""), kind="trained", ai=value.get("ai"), tp=value.get("tp"), human=value.get("human", value.get("rows")), fp=value.get("fp", value.get("false_positives")))


paper_sources = [trained["test"]["by_source"][name] for name in ("acl_anthology", "pmc_oa")]
trained_paper = dict(
    name="Qwen3 LoRA · mixed-trained", source="segment_qwen3_17b_mixed_stage1_v1",
    split="test:acl_anthology+pmc_oa", kind="trained",
    ai=sum(item["ai"] for item in paper_sources), tp=sum(item["tp"] for item in paper_sources),
    human=sum(item["human"] for item in paper_sources), fp=sum(item["fp"] for item in paper_sources),
)

sections = [
    ("MIXED TEST", "631 human + 631 AI · 35% papers in each class", [
        trained_row("Qwen3 LoRA · ours"),
        row("Character TF-IDF · full", "baseline_mixed_char_full"),
        row("Character TF-IDF · medium", "baseline_mixed_char_medium"),
        row("Character TF-IDF · small", "baseline_mixed_char_small"),
        row("Character TF-IDF · tiny", "baseline_mixed_char_tiny"),
        row("Word TF-IDF · medium", "baseline_mixed_word_medium"),
        row("MiniLM embedding · medium", "baseline_mixed_embedding_medium"),
        row("EditLens RoBERTa", "reference_mixed_roberta_full", kind="reference"),
        row("EditLens Llama", "reference_mixed_llama_full", kind="reference"),
    ]),
    ("PAPER ABSTRACT TEST", "221 human + 221 AI · held-out venues", [
        trained_paper,
        row("Character TF-IDF · full", "baseline_paper_char_full"),
        row("Character TF-IDF · medium", "baseline_paper_char_medium"),
        row("Character TF-IDF · small", "baseline_paper_char_small"),
        row("Character TF-IDF · tiny", "baseline_paper_char_tiny"),
        row("Word TF-IDF · full", "baseline_paper_word_full"),
        row("MiniLM embedding · full", "baseline_paper_embedding_full"),
        row("EditLens RoBERTa", "reference_paper_roberta_full", kind="reference"),
        row("EditLens Llama", "reference_paper_llama_full", kind="reference"),
    ]),
    ("GENERAL EDITLENS TEST", "2,000 human + 2,000 AI; Llama scored 500 + 500", [
        trained_row("Qwen3 LoRA · ours", "editlens_test"),
        row("Character TF-IDF · medium", "baseline_editlens_char_medium"),
        row("Word TF-IDF · medium", "baseline_editlens_word_medium"),
        row("MiniLM embedding · medium", "baseline_editlens_embedding_medium"),
        row("EditLens RoBERTa", "reference_editlens_roberta_full", kind="reference"),
        row("EditLens Llama · small test", "reference_editlens_llama_small", kind="reference"),
    ]),
    ("ENRON EMAIL TEST", "1,800 human + 1,800 AI; Llama scored 500 + 500", [
        trained_row("Qwen3 LoRA · ours", "enron_test"),
        row("Character TF-IDF · general medium", "baseline_editlens_char_medium", "test_enron"),
        row("Word TF-IDF · general medium", "baseline_editlens_word_medium", "test_enron"),
        row("MiniLM embedding · general medium", "baseline_editlens_embedding_medium", "test_enron"),
        row("EditLens RoBERTa", "reference_editlens_roberta_full", "test_enron", "reference"),
        row("EditLens Llama · small test", "reference_editlens_llama_small", "test_enron", "reference"),
    ]),
    ("MIXED-TEST PMC SUBSET", "87 human + 87 AI · papers held out from the Qwen3 mixed training split", [
        trained_row("Qwen3 LoRA · ours", source_name="pmc_oa"),
        row("Character TF-IDF · mixed full", "baseline_mixed_char_full", source_name="pmc_oa"),
        row("Character TF-IDF · mixed medium", "baseline_mixed_char_medium", source_name="pmc_oa"),
        row("Character TF-IDF · mixed small", "baseline_mixed_char_small", source_name="pmc_oa"),
        row("Character TF-IDF · mixed tiny", "baseline_mixed_char_tiny", source_name="pmc_oa"),
        row("Word TF-IDF · mixed medium", "baseline_mixed_word_medium", source_name="pmc_oa"),
        row("MiniLM embedding · mixed medium", "baseline_mixed_embedding_medium", source_name="pmc_oa"),
        row("EditLens RoBERTa", "reference_mixed_roberta_full", kind="reference", source_name="pmc_oa"),
        row("EditLens Llama", "reference_mixed_llama_full", kind="reference", source_name="pmc_oa"),
    ]),
    ("PMC-ONLY ABSTRACT TEST", "87 human + 87 AI · separate split overlaps mixed training, so Qwen3 is omitted", [
        row("Character TF-IDF · full", "baseline_pmc_char_full"),
        row("Word TF-IDF · full", "baseline_pmc_word_full"),
        row("MiniLM embedding · full", "baseline_pmc_embedding_full"),
        row("EditLens RoBERTa", "reference_pmc_roberta_full", kind="reference"),
        row("EditLens Llama", "reference_pmc_llama_full", kind="reference"),
    ]),
    ("SWAPPED-GENERATOR PAPER TEST", "182 human + 182 AI · same held-out human works, alternate generator", [
        trained_row("Qwen3 LoRA · ours", "cross_model_test"),
        row("Character TF-IDF · paper full", "baseline_paper_char_full", "cross_model_test"),
        row("Word TF-IDF · paper full", "baseline_paper_word_full", "cross_model_test"),
        row("MiniLM embedding · paper full", "baseline_paper_embedding_full", "cross_model_test"),
        row("EditLens RoBERTa", "reference_paper_roberta_full", "cross_model_test", "reference"),
        row("EditLens Llama", "reference_paper_llama_full", "cross_model_test", "reference"),
    ]),
    ("HUMAN ACL ABSTRACT AUDIT", "human only · paper/mixed audits exclude sampled paper works; general audits have a larger pool", [
        trained_row("Qwen3 LoRA · ours", "acl_human_audit"),
        row("Character TF-IDF · mixed medium", "baseline_mixed_char_medium", "acl_human_audit"),
        row("Word TF-IDF · mixed medium", "baseline_mixed_word_medium", "acl_human_audit"),
        row("Character TF-IDF · paper full", "baseline_paper_char_full", "acl_human_audit"),
        row("Character TF-IDF · general only", "baseline_editlens_char_medium", "acl_human_audit"),
        row("Word TF-IDF · general only", "baseline_editlens_word_medium", "acl_human_audit"),
        row("MiniLM embedding · general only", "baseline_editlens_embedding_medium", "acl_human_audit"),
    ]),
    ("HUMAN PMC BODY AUDIT", "261 human-only text chunks from held-out full papers · false positives only", [
        trained_row("Qwen3 LoRA · ours", "pmc_body_human_audit"),
        row("Character TF-IDF · PMC full", "baseline_pmc_char_full", "pmc_body_human_audit"),
        row("EditLens RoBERTa · PMC threshold", "reference_pmc_roberta_full", "pmc_body_human_audit", "reference"),
        row("Character TF-IDF · general only", "baseline_editlens_char_medium", "pmc_body_human_audit"),
    ]),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for title, _, rows in sections:
        for item in rows:
            assert item["human"] and item["fp"] is not None
            if item["ai"] is not None:
                assert item["ai"] > 0 and item["tp"] is not None
            csv_rows.append({"section": title, **item, "ai_recall": item["tp"] / item["ai"] if item["ai"] else "", "human_fpr": item["fp"] / item["human"]})
    with (OUT / "full_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    total = 2.65 + sum(1.09 + 0.67 * len(rows) for _, _, rows in sections) + 2.6
    fig, ax = plt.subplots(figsize=(16, 0.42 * total), dpi=170)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.axis("off")
    y = total - 1.0
    ax.text(0.025, y, "AI text detector · full pilot comparison", fontsize=21, weight="bold", color="#152337", va="center")
    y -= 0.66
    ax.text(0.025, y, "AI recall = generated texts detected     ·     Human FPR = human texts falsely flagged     ·     Dashed marker = 2% FPR target", fontsize=10, color="#53657a", va="center")
    y -= 0.65
    ax.text(0.025, y, "MODEL / TRAIN TIER", fontsize=8.5, weight="bold", color="#53657a")
    ax.text(0.33, y, "AI RECALL  ·  0–100%", fontsize=8.5, weight="bold", color="#53657a")
    ax.text(0.70, y, "HUMAN FALSE POSITIVES", fontsize=8.5, weight="bold", color="#53657a")
    y -= 0.34

    for title, note, rows in sections:
        y -= 0.47
        ax.add_patch(Rectangle((0.02, y - 0.13), 0.96, 0.71, facecolor="#e6ebf1", edgecolor="none"))
        ax.text(0.032, y + 0.31, title, va="center", fontsize=10.7, weight="bold", color="#19304a")
        ax.text(0.32, y + 0.31, note, va="center", fontsize=8.7, color="#53657a")
        y -= 0.5
        fp_max = 70 if title == "HUMAN ACL ABSTRACT AUDIT" else 40 if title == "HUMAN PMC BODY AUDIT" else 15 if title == "MIXED-TEST PMC SUBSET" else 10
        for index, item in enumerate(rows):
            if index % 2 == 0:
                ax.add_patch(Rectangle((0.02, y - 0.33), 0.96, 0.69, facecolor="#ffffff", edgecolor="none"))
            if item["kind"] == "trained":
                ax.add_patch(Rectangle((0.02, y - 0.33), 0.004, 0.69, facecolor="#6d53c9", edgecolor="none"))
            ax.text(0.035, y, item["name"], va="center", fontsize=9.4, color="#172638", weight="bold" if item["kind"] == "trained" else "normal")
            if item["ai"]:
                recall = item["tp"] / item["ai"]
                ax.add_patch(Rectangle((0.33, y - 0.105), 0.19, 0.21, facecolor="#e0e7e8", edgecolor="none"))
                ax.add_patch(Rectangle((0.33, y - 0.105), 0.19 * recall, 0.21, facecolor="#1a9b88", edgecolor="none"))
                ax.text(0.54, y, f"{recall*100:5.1f}%  ({item['tp']}/{item['ai']})", va="center", fontsize=9.0, color="#173b37")
            else:
                ax.text(0.54, y, "human only", va="center", fontsize=8.8, color="#7d8996")
            fpr = item["fp"] / item["human"]
            ax.add_patch(Rectangle((0.70, y - 0.105), 0.12, 0.21, facecolor="#f2e3df", edgecolor="none"))
            ax.add_patch(Rectangle((0.70, y - 0.105), min(fpr * 100 / fp_max, 1) * 0.12, 0.21, facecolor="#d7644e", edgecolor="none"))
            target_x = 0.70 + 0.12 * (2 / fp_max)
            ax.plot([target_x, target_x], [y - 0.16, y + 0.16], color="#9b3937", linewidth=1, linestyle=(0, (2, 2)))
            ax.text(0.835, y, f"{fpr*100:4.1f}%  ({item['fp']}/{item['human']})", va="center", fontsize=9.0, color="#7a342d", weight="bold" if fpr > 0.02 else "normal")
            y -= 0.67
        y -= 0.12

    ax.text(0.025, y - 0.14, "FPR bars use 0–10%; mixed PMC uses 0–15%, ACL audit 0–70%, and PMC body audit 0–40%. Rates are observed, not confidence bounds.", fontsize=8.8, color="#53657a", va="center")
    ax.text(0.025, y - 0.58, "Thresholds were selected on each model's validation set. All Qwen3 rows use its mixed-set threshold. Reference checkpoints had larger external training data.", fontsize=8.8, color="#53657a", va="center")
    ax.text(0.025, y - 1.02, "The PMC-only test overlaps Qwen3 training works. Paper AI comes from two small local generators; the swapped test reuses held-out human works. EditLens assets are noncommercial.", fontsize=8.8, color="#53657a", va="center")
    fig.savefig(OUT / "full_results.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / "full_results.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Wrote {OUT / 'full_results.png'} and {OUT / 'full_results.pdf'}")


if __name__ == "__main__":
    main()
