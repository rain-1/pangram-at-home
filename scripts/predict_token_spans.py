"""Predict human/AI character spans with overlapping Repeat2 token windows.

The binary pilot has no AI-assisted class. The output threshold should come
from a separate calibration set; use --threshold-json for the pilot threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

from span_data import window_starts


def predict(text: str, model, tokenizer, size: int, stride: int,
            repeat2: bool, threshold: float):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not ids:
        return {"source_tokens": 0, "spans": [], "ai_character_fraction": 0.0}
    scores = np.zeros(len(ids), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int32)
    with torch.inference_mode():
        for start in window_starts(len(ids), size, stride):
            window = ids[start:start + size]
            sequence = window + window if repeat2 else window
            x = torch.tensor([sequence], device="cuda")
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits.float()[0]
            if repeat2:
                logits = logits[-len(window):]
            margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            scores[start:start + len(window)] += margins
            counts[start:start + len(window)] += 1
    assert np.all(counts > 0)
    scores /= counts
    labels = scores >= threshold
    spans = []
    for (start, end), label in zip(offsets, labels):
        if end <= start:
            continue
        name = "ai_generated" if label else "human"
        if spans and spans[-1]["label"] == name:
            spans[-1]["end"] = end
        else:
            spans.append({"start": start, "end": end, "label": name})
    ai_chars = sum(span["end"] - span["start"] for span in spans
                   if span["label"] == "ai_generated")
    covered_chars = sum(span["end"] - span["start"] for span in spans)
    return {"source_tokens": len(ids), "windows": len(window_starts(len(ids), size, stride)),
            "spans": spans,
            "ai_character_fraction": ai_chars / covered_chars if covered_chars else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.threshold_json is None) == (args.threshold is None):
        parser.error("Provide exactly one of --threshold-json or --threshold")
    run = args.root / "runs" / args.run_name
    config = json.loads((run / "run_config.json").read_text())
    if config["task"] != "binary_token_classification":
        parser.error("This predictor requires a token-classification run")
    threshold = (json.loads(args.threshold_json.read_text())["threshold"]
                 if args.threshold_json else args.threshold)
    text = args.text_file.read_text(encoding="utf-8")
    adapter = run / "best_adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForTokenClassification.from_pretrained(
        config["base_model"], num_labels=2, dtype=torch.bfloat16, device_map={"": 0})
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()
    result = predict(text, model, tokenizer, config["max_length"], config["stride"],
                     config["repeat2"], threshold)
    result.update({"run_name": args.run_name, "threshold": threshold,
                   "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "input_characters": len(text)})
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
