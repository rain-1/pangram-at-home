"""Compare the three passage models on additional held-out sets.

Raw text stays on the external drive. The report contains scores and aggregate
metrics only. EditLens original and Llama variants share source prompts, so
they are separate generator probes rather than independent pooled evidence.
MAGE source IDs identify individual texts; exact-text filtering cannot prove
prompt or author independence for that set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate_diverse_lora import group_metrics


REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    "editlens_original": "editlens_pyramid_v1/test_full.parquet",
    "editlens_llama": "editlens_iclr/data/test_llama-00000-of-00001.parquet",
    "mage_main": "mage_external_v1/frozen/main.parquet",
    "paper_generator_swap": "paper_cross_model_test_v1/test.parquet",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_ids(root: Path):
    ids = set()
    hashes = set()
    for split in ("train", "val", "test"):
        path = root / "data/diverse_pyramid_v1" / f"{split}_full.parquet"
        columns = pq.read_table(path, columns=["source_id", "text"]).to_pydict()
        ids.update(columns["source_id"])
        hashes.update(text_hash(text) for text in columns["text"])
    return ids, hashes


def text_hash(text: str):
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).digest()


def read_dataset(root: Path, name: str, excluded_ids: set[str], excluded_hashes: set[bytes]):
    path = root / "data" / DATASETS[name]
    table = pq.read_table(path)
    raw = table.to_pydict()
    if name == "editlens_llama":
        eligible = [i for i, value in enumerate(raw["text_type"])
                    if value in {"human_written", "ai_generated"}]
    else:
        eligible = list(range(table.num_rows))
    source_clean = [i for i in eligible if raw["source_id"][i] not in excluded_ids]
    indexes = [i for i in source_clean if text_hash(raw["text"][i]) not in excluded_hashes]
    assert indexes and all(raw["source_id"][i] not in excluded_ids for i in indexes)
    labels = np.asarray([int(raw["text_type"][i] == "ai_generated")
                         if name == "editlens_llama" else int(raw["label"][i])
                         for i in indexes], dtype=np.int8)
    assert (labels == 0).any() and (labels == 1).any()
    texts = [raw["text"][i] for i in indexes]
    sources = np.asarray([raw["source"][i] for i in indexes], dtype=str)
    generators = np.asarray([
        "human" if label == 0 else
        (raw["model"][i] if name.startswith("editlens") else
         raw["generator"][i] if name == "paper_generator_swap" else "MAGE AI")
        for i, label in zip(indexes, labels)], dtype=str)
    metadata = {"path": str(path), "sha256": sha(path), "input_rows": table.num_rows,
                "excluded_task_scope_rows": table.num_rows - len(eligible),
                "excluded_source_overlap_rows": len(eligible) - len(source_clean),
                "excluded_exact_text_overlap_rows": len(source_clean) - len(indexes),
                "rows": len(indexes), "human": int((labels == 0).sum()),
                "ai": int((labels == 1).sum()),
                "source_ids": len({raw["source_id"][i] for i in indexes}),
                "source_id_overlap_with_diverse_splits": 0,
                "normalized_text_overlap_with_diverse_splits": 0}
    return texts, labels, sources, generators, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = args.root / "runs" / args.run_name
    config = json.loads((run / "run_config.json").read_text())
    operating = json.loads((REPO / "reports/metrics" / f"{args.run_name}.json").read_text())
    adapter = run / "best_adapter"
    adapter_hash = sha(adapter / "adapter_model.safetensors")
    assert adapter_hash == operating["adapter_sha256"]
    threshold = operating["threshold"]
    excluded_ids, excluded_hashes = reference_ids(args.root)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2, dtype=torch.bfloat16, device_map={"": 0})
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()
    cache_dir = run / "comparative_ood_v1"
    cache_dir.mkdir(exist_ok=True)
    output = {"run_name": args.run_name, "adapter_sha256": adapter_hash,
              "threshold": threshold, "threshold_source": operating["threshold_source"],
              "role": "additional frozen evaluation; no new threshold fitting",
              "note": "EditLens original and Llama variants share source prompts and are not independent pooled samples",
              "splits": {}}
    with torch.inference_mode():
        for name in DATASETS:
            texts, labels, sources, generators, metadata = read_dataset(
                args.root, name, excluded_ids, excluded_hashes)
            cache = cache_dir / f"{name}.npz"
            if cache.exists():
                old = np.load(cache)
                assert str(old["input_sha256"]) == metadata["sha256"]
                assert str(old["adapter_sha256"]) == adapter_hash
                assert len(old["margin"]) == len(labels)
                margins = old["margin"]
            else:
                margins = []
                for start in range(0, len(texts), args.batch_size):
                    encoded = tokenizer(texts[start:start + args.batch_size],
                                        truncation=True, max_length=config["max_length"])
                    if config.get("repeat2", False):
                        encoded = {key: [values + values for values in batch_values]
                                   for key, batch_values in encoded.items()}
                    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to("cuda")
                    logits = model(**batch).logits.float()
                    margins.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
                    if start and start % 500 < args.batch_size:
                        print(name, min(start + args.batch_size, len(texts)), "/", len(texts), flush=True)
                margins = np.asarray(margins, dtype=np.float32)
                np.savez_compressed(cache, margin=margins, label=labels,
                                    source=sources, generator=generators,
                                    chars=np.asarray([len(text) for text in texts]),
                                    input_sha256=metadata["sha256"], adapter_sha256=adapter_hash)
            result = {**metadata, **group_metrics(margins, labels, threshold)}
            result["by_source"] = {source: group_metrics(margins[sources == source],
                                                        labels[sources == source], threshold)
                                   for source in sorted(set(sources))}
            result["by_generator"] = {generator: group_metrics(margins[generators == generator],
                                                              labels[generators == generator], threshold)
                                      for generator in sorted(set(generators))}
            lengths = np.asarray([len(text) for text in texts])
            bins = {"under_500_chars": lengths < 500,
                    "500_to_999_chars": (lengths >= 500) & (lengths < 1000),
                    "1000_to_1999_chars": (lengths >= 1000) & (lengths < 2000),
                    "2000_plus_chars": lengths >= 2000}
            result["by_length"] = {key: group_metrics(margins[mask], labels[mask], threshold)
                                   for key, mask in bins.items() if mask.any()}
            output["splits"][name] = result
            print(name, Counter(labels), "FPR", result["fpr"],
                  "AI recall", result["tpr"], flush=True)
    target = REPO / "reports/metrics" / f"{args.run_name}_comparative_ood_v1.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
