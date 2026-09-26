"""Audit a stricter pure-human-document threshold without GPU inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from span_data import encode_document


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]
RUNS = ["qwen3_token_single_v3_pilot1", "qwen3_token_repeat2_v3_pilot1"]
SETS = {
    "validation": ("span_pilot_v3/val.jsonl", "span_validation_v3"),
    "confirmation": ("span_test_probe_v1/test.jsonl", "span_test_v1"),
    "long_composite": ("span_long_probe_v1/val.jsonl", "span_long_v1"),
}


def load(run: Path, tokenizer, source: str, stem: str):
    rows = [json.loads(line) for line in (ROOT / "data" / source).read_text().splitlines()]
    cache = np.load(run / f"{stem}_scores.npz")
    scores = cache["score"]
    labels = cache["label"]
    documents = []
    cursor = 0
    for row in rows:
        _, _, all_labels = encode_document(row, tokenizer)
        valid = np.asarray(all_labels) != -100
        count = int(valid.sum())
        y = np.asarray(all_labels, dtype=np.int8)[valid]
        assert np.array_equal(y, labels[cursor:cursor + count])
        documents.append((row["kind"], scores[cursor:cursor + count], y))
        cursor += count
    assert cursor == len(scores)
    return documents


def summarize(documents, threshold):
    y = np.concatenate([row[2] for row in documents])
    pred = np.concatenate([row[1] >= threshold for row in documents])
    pure_human = [row for row in documents if row[0] == "human"]
    return {"documents": len(documents), "threshold": float(threshold),
            "human_token_fpr": float(pred[y == 0].mean()),
            "ai_token_recall": float(pred[y == 1].mean()),
            "pure_human_documents_with_any_false_highlight": float(np.mean([
                bool(np.any(row[1] >= threshold)) for row in pure_human])),
            "pure_human_documents": len(pure_human)}


def main():
    output = {"role": "diagnostic operating point; 5% pure-human-document target calibrated on synthetic validation",
              "models": {}}
    for name in RUNS:
        run = ROOT / "runs" / name
        tokenizer = AutoTokenizer.from_pretrained(run / "best_adapter")
        sets = {label: load(run, tokenizer, source, stem)
                for label, (source, stem) in SETS.items()}
        control_max = np.sort(np.asarray([
            float(np.max(scores)) for kind, scores, _ in sets["validation"]
            if kind == "human"]))[::-1]
        allowed = int(np.floor(.05 * len(control_max)))
        threshold = float(np.nextafter(np.float32(control_max[allowed]),
                                       np.float32(np.inf), dtype=np.float32))
        default = json.loads((run / "span_validation_v3.json").read_text())["threshold"]
        output["models"][name] = {
            "calibration_pure_human_documents": len(control_max),
            "allowed_validation_documents_with_any_false_highlight": allowed,
            "threshold_doc_5pct": threshold,
            "threshold_token_2pct": default,
            "doc_5pct": {label: summarize(documents, threshold)
                         for label, documents in sets.items()},
            "token_2pct": {label: summarize(documents, default)
                           for label, documents in sets.items()},
        }
    path = REPO / "reports/metrics/token_threshold_audit_v3.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
