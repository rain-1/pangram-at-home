"""Cache Qwen3 segment logits on frozen validation and holdout sets for threshold analysis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--run-name", default="qwen3_17b_mixed_stage1_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = args.root / "runs" / args.run_name
    config = json.loads((run / "run_config.json").read_text())
    adapter = run / "best_adapter"
    destination = run / "score_cache_v1"
    destination.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = destination / "manifest.json"
    existing_manifest = json.loads(existing_manifest_path.read_text()) if existing_manifest_path.exists() else None
    adapter_hash = sha256(adapter / "adapter_model.safetensors")
    if existing_manifest:
        assert existing_manifest["run_name"] == args.run_name
        assert existing_manifest["adapter_sha256"] == adapter_hash
        assert existing_manifest["max_length"] == config["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()

    paths = {
        "mixed_val": args.root / "data/mixed_pyramid_v1/val_full.parquet",
        "mixed_test": args.root / "data/mixed_pyramid_v1/test_full.parquet",
        "cross_model_test": args.root / "data/paper_cross_model_test_v1/test.parquet",
        "pmc_body_audit": args.root / "data/pmc_body_audit_v1/human_test.parquet",
        "editlens_test": args.root / "data/editlens_pyramid_v1/test_full.parquet",
        "enron_test": args.root / "data/editlens_pyramid_v1/test_enron_full.parquet",
    }

    def score(name: str, texts: list[str], labels: list[int], sources: list[str], input_path: Path) -> None:
        path = destination / f"{name}.npz"
        if path.exists():
            if not existing_manifest or existing_manifest["scores"][name]["input_sha256"] != sha256(input_path):
                raise RuntimeError(f"Unverified or stale score cache: {path}")
            print(f"Using existing {path}", flush=True)
            return
        margins = []
        probabilities = []
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=config["max_length"]).to("cuda")
            with torch.inference_mode():
                logits = model(**batch).logits.float()
            margins.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
            probabilities.extend(logits.softmax(dim=-1)[:, 1].cpu().numpy().tolist())
            if start and start % 1000 < args.batch_size:
                print(f"{name}: {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
        np.savez_compressed(
            path, margin=np.asarray(margins, dtype=np.float32),
            probability=np.asarray(probabilities, dtype=np.float32),
            label=np.asarray(labels, dtype=np.int8), source=np.asarray(sources, dtype=str),
        )
        print(f"Saved {path} ({len(texts)} rows)", flush=True)

    for name, path in paths.items():
        rows = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
        score(name, rows["text"], rows["label"], rows["source"], path)

    acl_path = args.root / "data/acl_abstracts_v1/documents.jsonl.gz"
    used_ids = set()
    for split in ("train", "val", "test"):
        used_ids.update(
            row["source_id"] for row in pq.read_table(
                args.root / "data/paper_pyramid_v1" / f"{split}_full.parquet",
                columns=["source", "source_id"],
            ).to_pylist() if row["source"] == "acl_anthology"
        )
    with gzip.open(acl_path, "rt", encoding="utf-8") as file:
        acl = [json.loads(line) for line in file]
    acl = [row for row in acl if 100 <= len(row["abstract"].split()) <= 350 and row["source_id"] not in used_ids]
    assert len(acl) == 17560, len(acl)
    score("acl_human_audit", [row["abstract"] for row in acl], [0] * len(acl), ["acl_anthology"] * len(acl), acl_path)
    manifest = {
        "run_name": args.run_name, "adapter_sha256": adapter_hash,
        "max_length": config["max_length"],
        "scores": {name: {"path": f"{name}.npz", "input": str(path), "input_sha256": sha256(path)} for name, path in {**paths, "acl_human_audit": acl_path}.items()},
        "acl_filter": "100-350 words, excluding every ACL source_id in paper train/val/test",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
