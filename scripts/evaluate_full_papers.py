"""Audit whole-paper false highlighting on known-human, held-out PMC bodies."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from span_data import window_starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pool", choices=["all_raw", "audit_chunks"], default="all_raw")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    run = root / "runs" / args.run_name
    config = json.loads((run / "run_config.json").read_text())
    report = json.loads((Path(__file__).resolve().parents[1] / "reports/metrics" /
                         f"{args.run_name}.json").read_text())
    threshold = report["threshold"]
    operating_thresholds = {key: value["threshold"] for key, value in
                            report["operating_points"].items()}
    chunk_audit_ids = set(pq.read_table(root / "data/pmc_body_audit_v1/human_test.parquet",
                                        columns=["source_id"]).column(0).to_pylist())
    raw_documents = []
    with gzip.open(root / "data/pmc_pilot_v1/documents.jsonl.gz", "rt") as file:
        for line in file:
            row = json.loads(line)
            if row["publication_date"] < "2023-01-01" and len(row.get("body", "")) >= 3000:
                raw_documents.append(row)
    eligible = ({row["source_id"] for row in raw_documents} if args.pool == "all_raw"
                else chunk_audit_ids & {row["source_id"] for row in raw_documents})
    seen = set()
    for split in ("train", "val", "test"):
        seen.update(pq.read_table(root / "data/diverse_pyramid_v1" /
                                  f"{split}_full.parquet", columns=["source_id"]).column(0).to_pylist())
    independent = eligible - seen
    chosen = set(sorted(independent, key=lambda key: hashlib.sha256(
        ("full-paper-audit-v1:" + key).encode()).digest())[:args.limit])
    assert chosen, "No independent PMC bodies found"
    documents = [row for row in raw_documents if row["source_id"] in chosen]
    assert len(documents) == len(chosen), (len(documents), len(chosen))
    tokenizer = AutoTokenizer.from_pretrained(run / "best_adapter")
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2, dtype=torch.bfloat16, device_map={"": 0})
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, run / "best_adapter").eval()
    size = config["max_length"]
    stride = size // 2
    rows = []
    with torch.inference_mode():
        for index, document in enumerate(documents, 1):
            ids = tokenizer(document["body"], add_special_tokens=False)["input_ids"]
            starts = window_starts(len(ids), size, stride)
            sums = np.zeros(len(ids), dtype=np.float64)
            counts = np.zeros(len(ids), dtype=np.int32)
            for offset in range(0, len(starts), args.batch_size):
                batch_starts = starts[offset:offset + args.batch_size]
                pieces = [ids[start:start + size] for start in batch_starts]
                if config.get("repeat2", False):
                    pieces = [piece + piece for piece in pieces]
                encoded = tokenizer.pad({"input_ids": pieces}, padding=True,
                                        return_tensors="pt").to("cuda")
                logits = model(**encoded).logits.float()
                margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                for start, margin in zip(batch_starts, margins):
                    end = min(start + size, len(ids))
                    sums[start:end] += float(margin)
                    counts[start:end] += 1
            assert np.all(counts > 0)
            scores = sums / counts
            flagged = scores >= threshold
            rows.append({"source_id": document["source_id"],
                         "publication_date": document["publication_date"],
                         "journal": document.get("journal"),
                         "license": document["license"], "tokens": len(ids),
                         "windows": len(starts), "false_highlight_tokens": int(flagged.sum()),
                         "false_highlight_fraction": float(flagged.mean()),
                         "any_false_highlight": bool(flagged.any()),
                         "max_window_averaged_margin": float(scores.max()),
                         "operating_points": {key: {"false_highlight_tokens": int((scores >= point).sum()),
                                                    "any_false_highlight": bool((scores >= point).any())}
                                              for key, point in operating_thresholds.items()}})
            print(index, "/", len(documents), document["source_id"],
                  "tokens", len(ids), "false fraction", f"{flagged.mean():.3f}", flush=True)
    output = {"run_name": args.run_name, "role": "source-disjoint pre-2023 PMC full-paper human audit",
              "pool": args.pool,
              "threshold": threshold, "threshold_source": "diverse 512-token validation passages at <=2% FPR",
              "source_ids_selected_by": "Exclude all diverse train/val/test source IDs; then SHA256(full-paper-audit-v1:source_id), first N",
              "eligible_ids": len(eligible), "independent_ids": len(independent),
              "excluded_overlap_ids": len(eligible - independent),
              "chunk_audit_ids": len(chunk_audit_ids),
              "chunk_audit_overlap_ids": len(chunk_audit_ids & seen),
              "window_size": size, "stride": stride,
              "documents": len(rows), "tokens": sum(row["tokens"] for row in rows),
              "windows": sum(row["windows"] for row in rows),
              "token_false_highlight_rate": sum(row["false_highlight_tokens"] for row in rows) /
                                            sum(row["tokens"] for row in rows),
              "documents_with_any_false_highlight": sum(row["any_false_highlight"] for row in rows),
              "operating_points": {key: {"threshold": point,
                  "token_false_highlight_rate": sum(row["operating_points"][key]["false_highlight_tokens"]
                                                    for row in rows) / sum(row["tokens"] for row in rows),
                  "documents_with_any_false_highlight": sum(row["operating_points"][key]["any_false_highlight"]
                                                            for row in rows)}
                                   for key, point in operating_thresholds.items()},
              "per_document": rows}
    path = Path(__file__).resolve().parents[1] / "reports/metrics" / f"{args.run_name}_pmc_fullpaper.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path, output["token_false_highlight_rate"], flush=True)


if __name__ == "__main__":
    main()
