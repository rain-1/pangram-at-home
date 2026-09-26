"""Attach aggregate span-pilot evaluation metrics to the two training runs."""
from __future__ import annotations

import json
from pathlib import Path

import wandb


REPO = Path(__file__).resolve().parents[1]
RUN_IDS = {"Token single": "2jrklgmg", "Token Repeat2": "053oc7dr"}


def main():
    summary = json.loads((REPO / "reports/metrics/token_span_pilot_v3_summary.json").read_text())
    audit = json.loads((REPO / "reports/metrics/token_threshold_audit_v3.json").read_text())
    names = {"Token single": "qwen3_token_single_v3_pilot1",
             "Token Repeat2": "qwen3_token_repeat2_v3_pilot1"}
    api = wandb.Api()
    for label, run_id in RUN_IDS.items():
        run = api.run(f"eac-adsf/pangram-at-home/{run_id}")
        for set_label, item in summary["models"][label].items():
            prefix = f"span_pilot_v3/{set_label.lower().replace(' ', '_')}"
            run.summary[f"{prefix}/human_token_fpr"] = item["overall"]["fpr"]
            run.summary[f"{prefix}/ai_token_recall"] = item["overall"]["ai_recall"]
            run.summary[f"{prefix}/ai_spans_half_covered"] = item["mixed_ai_span_recall_half_covered"]
            run.summary[f"{prefix}/pure_human_docs_any_false_highlight"] = item["pure_human_document_any_false_highlight_rate"]
        doc_safe = audit["models"][names[label]]["doc_5pct"]
        for set_label in ("confirmation", "long_composite"):
            item = doc_safe[set_label]
            prefix = f"span_pilot_v3/doc_safe/{set_label}"
            run.summary[f"{prefix}/human_token_fpr"] = item["human_token_fpr"]
            run.summary[f"{prefix}/ai_token_recall"] = item["ai_token_recall"]
            run.summary[f"{prefix}/pure_human_docs_any_false_highlight"] = item[
                "pure_human_documents_with_any_false_highlight"]
        run.summary["span_pilot_v3/data_status"] = "synthetic known-origin development; confirmation parent previously examined"
        run.summary.update()
        print(label, run.url)


if __name__ == "__main__":
    main()
