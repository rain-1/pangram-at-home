"""Compare validation-chosen Qwen3 thresholds on frozen holdouts without test tuning."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "qwen3_17b_mixed_stage1_v1"
DATA_ROOT = Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home"))
CACHE = DATA_ROOT / "runs" / RUN_NAME / "score_cache_v1"
OUT = ROOT / "reports"
SPLITS = ["mixed_val", "mixed_test", "cross_model_test", "pmc_body_audit", "editlens_test", "enron_test", "acl_human_audit"]


def cutoff_for_fpr(scores: np.ndarray, labels: np.ndarray, target: float) -> float:
    humans = np.sort(scores[labels == 0])[::-1]
    allowed = math.floor(target * len(humans))
    return float(np.nextafter(humans[allowed], np.inf))


def measure(data: dict, cutoff: float, mask=None) -> dict:
    margin = data["margin"]
    label = data["label"]
    if mask is not None:
        margin, label = margin[mask], label[mask]
    prediction = margin >= cutoff
    human = label == 0
    ai = label == 1
    fp = int((prediction & human).sum())
    tp = int((prediction & ai).sum())
    return {
        "human": int(human.sum()), "ai": int(ai.sum()), "fp": fp, "tp": tp,
        "fpr": fp / int(human.sum()) if human.any() else None,
        "recall": tp / int(ai.sum()) if ai.any() else None,
    }


def main() -> None:
    data = {}
    for name in SPLITS:
        with np.load(CACHE / f"{name}.npz", allow_pickle=False) as file:
            data[name] = {
                "margin": file["margin"].astype(np.float64),
                "label": file["label"], "source": file["source"],
                "probability": file["probability"].astype(np.float64),
            }
    published = json.loads((ROOT / "reports/metrics/segment_qwen3_17b_mixed_stage1_v1.json").read_text())
    original_probability = published["threshold"]
    original_cutoff = math.log(original_probability / (1 - original_probability))
    for name, key in [("mixed_test", "test"), ("cross_model_test", "cross_model_test"), ("pmc_body_audit", "pmc_body_human_audit")]:
        old = published[key]
        observed = measure(data[name], original_cutoff)
        assert observed["fp"] == old.get("fp", old.get("false_positives")), (name, observed, old)
        if observed["ai"]:
            assert observed["tp"] == old["tp"], (name, observed, old)

    validation = data["mixed_val"]
    human_scores = validation["margin"][validation["label"] == 0]
    ai_scores = np.sort(validation["margin"][validation["label"] == 1])
    assert human_scores.max() < ai_scores.min(), "No validation separation for midpoint policy"
    policies = [
        ("original", "Original · 2% validation FPR", original_cutoff),
        ("val_1pct", "At most 1% validation FPR", cutoff_for_fpr(validation["margin"], validation["label"], 0.01)),
        ("val_0fp", "Zero validation false positives", cutoff_for_fpr(validation["margin"], validation["label"], 0)),
        ("score_50pct", "Standard AI score ≥ 50%", 0.0),
        ("val_midpoint", "Validation class-gap midpoint", (human_scores.max() + ai_scores.min()) / 2),
        ("val_99recall", "At least 99% validation AI recall", float(ai_scores[math.floor(0.01 * len(ai_scores))])),
    ]

    paper = np.isin(data["mixed_test"]["source"], ["acl_anthology", "pmc_oa"])
    pmc = data["mixed_test"]["source"] == "pmc_oa"
    results = []
    for policy_id, label, cutoff in policies:
        result = {
            "policy": policy_id, "label": label,
            "margin_cutoff": cutoff, "ai_probability_cutoff": 1 / (1 + math.exp(-cutoff)),
            "mixed_val": measure(data["mixed_val"], cutoff),
            "mixed_test": measure(data["mixed_test"], cutoff),
            "paper_subset": measure(data["mixed_test"], cutoff, paper),
            "pmc_subset": measure(data["mixed_test"], cutoff, pmc),
            "cross_model_test": measure(data["cross_model_test"], cutoff),
            "editlens_test": measure(data["editlens_test"], cutoff),
            "enron_test": measure(data["enron_test"], cutoff),
            "acl_human_audit": measure(data["acl_human_audit"], cutoff),
            "pmc_body_audit": measure(data["pmc_body_audit"], cutoff),
        }
        results.append(result)
        print(label, "score", round(result["ai_probability_cutoff"], 6),
              "mixed", result["mixed_test"]["tp"], result["mixed_test"]["fp"],
              "paper", result["paper_subset"]["tp"], result["paper_subset"]["fp"],
              "ACL FP", result["acl_human_audit"]["fp"],
              "PMC body FP", result["pmc_body_audit"]["fp"])

    payload = {
        "run_name": RUN_NAME, "score_cache_manifest": str(CACHE / "manifest.json"),
        "threshold_selection": "Only mixed validation labels and a fixed 50% probability rule; no holdout thresholds fitted",
        "note": "Scores were recomputed in batches of 8; the original run used batches of 4. Frozen mixed/cross/body holdout counts reproduce the original run at its saved cutoff.",
        "policies": results,
    }
    (OUT / "threshold-tradeoff.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "threshold-tradeoff.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["policy", "label", "margin_cutoff", "ai_probability_cutoff", "split", "human", "ai", "fp", "tp", "fpr", "recall"])
        for result in results:
            for name in SPLITS + ["paper_subset", "pmc_subset"]:
                metric = result[name]
                writer.writerow([result["policy"], result["label"], result["margin_cutoff"], result["ai_probability_cutoff"], name,
                                 metric["human"], metric["ai"], metric["fp"], metric["tp"], metric["fpr"], metric["recall"]])

    midpoint = next(row for row in results if row["policy"] == "val_midpoint")
    cutoff = midpoint["margin_cutoff"]

    def with_sources(name: str) -> dict:
        result = measure(data[name], cutoff)
        result["by_source"] = {
            source: measure(data[name], cutoff, data[name]["source"] == source)
            for source in sorted(set(data[name]["source"]))
        }
        return result

    calibrated = {
        "run_name": RUN_NAME, "threshold_policy": "validation_class_gap_midpoint",
        "threshold_source": "mixed_pyramid_v1/val_full.parquet",
        "threshold_margin": cutoff, "threshold": midpoint["ai_probability_cutoff"],
        "val": measure(data["mixed_val"], cutoff),
        "test": with_sources("mixed_test"),
        "cross_model_test": with_sources("cross_model_test"),
        "pmc_body_human_audit": measure(data["pmc_body_audit"], cutoff),
        "editlens_test": with_sources("editlens_test"),
        "enron_test": with_sources("enron_test"),
        "acl_human_audit": measure(data["acl_human_audit"], cutoff),
    }
    (OUT / "metrics/segment_qwen3_17b_mixed_stage1_v1_midpoint.json").write_text(json.dumps(calibrated, indent=2) + "\n")
    score_manifest = json.loads((CACHE / "manifest.json").read_text())
    operating_point = {
        "run_name": RUN_NAME,
        "adapter_sha256": score_manifest["adapter_sha256"],
        "policy": "midpoint between highest human and lowest AI margin on mixed validation",
        "validation_file": "mixed_pyramid_v1/val_full.parquet",
        "validation_file_sha256": score_manifest["scores"]["mixed_val"]["input_sha256"],
        "decision_rule": "AI if (AI logit - human logit) >= threshold_margin",
        "threshold_margin": cutoff,
        "equivalent_uncalibrated_ai_score_cutoff": midpoint["ai_probability_cutoff"],
        "validation_human_max_margin": float(human_scores.max()),
        "validation_ai_min_margin": float(ai_scores.min()),
        "max_length": score_manifest["max_length"],
    }
    config_path = ROOT / "configs/qwen3_stage1_operating_point.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(operating_point, indent=2) + "\n")


if __name__ == "__main__":
    main()
