"""Evaluate the first Qwen3 adapter on frozen, independently sourced MAGE tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    human = labels == 0
    ai = labels == 1
    return {
        "rows": len(labels), "human": int(human.sum()), "ai": int(ai.sum()),
        "tp": int((pred & ai).sum()), "fp": int((pred & human).sum()),
        "recall": float(pred[ai].mean()) if ai.any() else None,
        "fpr": float(pred[human].mean()) if human.any() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if human.any() and ai.any() else None,
        "average_precision": float(average_precision_score(labels, scores)) if human.any() and ai.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    run = args.root / "runs/qwen3_17b_mixed_stage1_v1"
    adapter = run / "best_adapter"
    model_hash = sha256(adapter / "adapter_model.safetensors")
    operating = json.loads((repo / "configs/qwen3_stage1_operating_point.json").read_text())
    assert model_hash == operating["adapter_sha256"]
    threshold = operating["threshold_margin"]
    config = json.loads((run / "run_config.json").read_text())
    source = args.root / "data/mage_external_v1/frozen"
    manifest = json.loads((source / "manifest.json").read_text())
    output = run / "mage_external_eval_v1"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()
    report = {"run_name": "qwen3_17b_mixed_stage1_v1", "adapter_sha256": model_hash,
              "threshold_source": "mixed validation midpoint, frozen before MAGE evaluation",
              "threshold_margin": threshold, "mage_revision": manifest["revision"], "splits": {}}
    book_path = args.root / "data/standard_ebooks_v1/human.parquet"
    splits = dict(manifest["splits"])
    if book_path.exists():
        splits["standard_ebooks_human"] = {"path": str(book_path), "sha256": sha256(book_path)}
    for split, info in splits.items():
        path = Path(info["path"]) if Path(info["path"]).is_absolute() else source / info["path"]
        assert sha256(path) == info["sha256"]
        if split == "standard_ebooks_human":
            rows = pq.read_table(path, columns=["text", "label", "source", "source_id"]).to_pydict()
            rows["mage_source"] = rows["source_id"]
        else:
            rows = pq.read_table(path, columns=["text", "label", "source", "mage_source"]).to_pydict()
        labels = np.asarray(rows["label"], dtype=np.int8)
        domains = np.asarray(rows["source"], dtype=str)
        mage_sources = np.asarray(rows["mage_source"], dtype=str)
        cache = output / f"{split}.npz"
        if cache.exists():
            old = np.load(cache)
            assert len(old["margin"]) == len(labels)
            if "input_sha256" in old.files:
                assert str(old["input_sha256"]) == info["sha256"]
            else:
                # First evaluation caches predated this field; verify their row metadata.
                assert np.array_equal(old["label"], labels)
                assert np.array_equal(old["source"], domains)
                assert np.array_equal(old["mage_source"], mage_sources)
            margins = old["margin"]
            print(f"Using {cache}", flush=True)
        else:
            result = []
            for start in range(0, len(labels), args.batch_size):
                batch = tokenizer(rows["text"][start:start + args.batch_size], return_tensors="pt", padding=True,
                                  truncation=True, max_length=config["max_length"]).to("cuda")
                with torch.inference_mode():
                    logits = model(**batch).logits.float()
                result.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
                if start and start % 400 < args.batch_size:
                    print(split, min(start + args.batch_size, len(labels)), "/", len(labels), flush=True)
            margins = np.asarray(result, dtype=np.float32)
            np.savez_compressed(cache, margin=margins, label=labels, source=domains,
                                mage_source=mage_sources, input_sha256=info["sha256"])
        split_report = metrics(labels, margins, threshold)
        split_report["input_sha256"] = info["sha256"]
        split_report["by_domain"] = {domain: metrics(labels[domains == domain], margins[domains == domain], threshold)
                                     for domain in sorted(set(domains))}
        if split == "standard_ebooks_human":
            split_report["by_book"] = {book: metrics(labels[mage_sources == book], margins[mage_sources == book], threshold)
                                       for book in sorted(set(mage_sources))}
        report["splits"][split] = split_report
        (repo / "reports/metrics/segment_qwen3_mage_external_v1.json").write_text(json.dumps(report, indent=2) + "\n")
        print(split, split_report["roc_auc"], split_report["recall"], split_report["fpr"], flush=True)


if __name__ == "__main__":
    main()
