"""Calibrate a trained segment adapter on frozen validation data and score holdouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from run_baselines import metrics, threshold_for_fpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--run-name", default="qwen3_17b_mixed_stage1_v1")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = args.root / "runs" / args.run_name
    config = json.loads((run / "run_config.json").read_text())
    adapter = run / "best_adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()

    def score(path: Path):
        data = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
        scores = []
        for start in range(0, len(data["text"]), args.batch_size):
            inputs = tokenizer(data["text"][start:start + args.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=config["max_length"]).to("cuda")
            with torch.inference_mode():
                output = model(**inputs).logits.float().softmax(dim=-1)[:, 1]
            scores.extend(output.cpu().numpy().tolist())
        return data, np.asarray(scores, dtype=np.float64)

    folder = args.root / "data" / "mixed_pyramid_v1"
    val, val_scores = score(folder / "val_full.parquet")
    threshold = threshold_for_fpr(val_scores, np.asarray(val["label"]), 0.02)
    result = {"run_name": args.run_name, "threshold": threshold, "val": metrics(val_scores, np.asarray(val["label"]), threshold)}
    for name, path in {
        "test": folder / "test_full.parquet",
        "cross_model_test": args.root / "data" / "paper_cross_model_test_v1" / "test.parquet",
        "pmc_body_human_audit": args.root / "data" / "pmc_body_audit_v1" / "human_test.parquet",
    }.items():
        data, scores = score(path)
        labels = np.asarray(data["label"])
        if name == "pmc_body_human_audit":
            result[name] = {"rows": len(scores), "false_positives": int((scores >= threshold).sum()), "fpr": float((scores >= threshold).mean())}
        else:
            result[name] = metrics(scores, labels, threshold)
            result[name]["by_source"] = {
                source: metrics(scores[np.asarray(data["source"]) == source], labels[np.asarray(data["source"]) == source], threshold)
                for source in sorted(set(data["source"]))
                if len(set(labels[np.asarray(data["source"]) == source])) == 2
            }
    path = args.root / "results" / f"segment_{args.run_name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
