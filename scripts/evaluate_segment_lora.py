"""Calibrate a trained segment adapter on frozen validation data and score holdouts."""

from __future__ import annotations

import argparse
import gzip
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
    parser.add_argument("--extra-only", action="store_true", help="Score additional held-out sets using the saved mixed-validation threshold")
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

    def score_texts(texts):
        scores = []
        for start in range(0, len(texts), args.batch_size):
            inputs = tokenizer(texts[start:start + args.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=config["max_length"]).to("cuda")
            with torch.inference_mode():
                output = model(**inputs).logits.float().softmax(dim=-1)[:, 1]
            scores.extend(output.cpu().numpy().tolist())
            if start and start % 1000 < args.batch_size:
                print(f"Scored {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.asarray(scores, dtype=np.float64)

    def score(path: Path):
        data = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
        return data, score_texts(data["text"])

    folder = args.root / "data" / "mixed_pyramid_v1"
    result_path = args.root / "results" / f"segment_{args.run_name}.json"
    if args.extra_only:
        result = json.loads(result_path.read_text())
        assert result["run_name"] == args.run_name
        threshold = result["threshold"]
        paths = {
            "editlens_test": args.root / "data" / "editlens_pyramid_v1" / "test_full.parquet",
            "enron_test": args.root / "data" / "editlens_pyramid_v1" / "test_enron_full.parquet",
        }
    else:
        val, val_scores = score(folder / "val_full.parquet")
        threshold = threshold_for_fpr(val_scores, np.asarray(val["label"]), 0.02)
        result = {"run_name": args.run_name, "threshold": threshold, "val": metrics(val_scores, np.asarray(val["label"]), threshold)}
        paths = {
            "test": folder / "test_full.parquet",
            "cross_model_test": args.root / "data" / "paper_cross_model_test_v1" / "test.parquet",
            "pmc_body_human_audit": args.root / "data" / "pmc_body_audit_v1" / "human_test.parquet",
        }
    for name, path in paths.items():
        print(f"Scoring {name}: {path}", flush=True)
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
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.extra_only:
        used_ids = set()
        paper_dir = args.root / "data" / "paper_pyramid_v1"
        for split in ("train", "val", "test"):
            used_ids.update(
                row["source_id"] for row in pq.read_table(
                    paper_dir / f"{split}_full.parquet", columns=["source", "source_id"]
                ).to_pylist() if row["source"] == "acl_anthology"
            )
        acl_path = args.root / "data" / "acl_abstracts_v1" / "documents.jsonl.gz"
        with gzip.open(acl_path, "rt", encoding="utf-8") as file:
            acl = [json.loads(line) for line in file]
        acl = [row for row in acl if 100 <= len(row["abstract"].split()) <= 350 and row["source_id"] not in used_ids]
        assert len(acl) == 17560, len(acl)
        print(f"Scoring acl_human_audit: {len(acl)} held-out abstracts", flush=True)
        acl_scores = score_texts([row["abstract"] for row in acl])
        result["acl_human_audit"] = {
            "rows": len(acl), "false_positives": int((acl_scores >= threshold).sum()),
            "fpr": float((acl_scores >= threshold).mean()),
            "exclusion": "all ACL works sampled for paper train/validation/test",
        }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
