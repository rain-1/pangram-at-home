"""Create a human-only full-paper prose audit from held-out PMC test papers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def chunks(body: str) -> list[str]:
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    result = []
    current = []
    words = 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if count > 350:
            # Long extracted paragraphs are often table/caption contamination.
            continue
        if words + count > 300 and words >= 150:
            result.append(" ".join(current))
            current, words = [], 0
        current.append(paragraph)
        words += count
    if words >= 150:
        result.append(" ".join(current))
    return [item for item in result if 150 <= len(item.split()) <= 350]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    root = args.root
    test_path = root / "data" / "pmc_pyramid_v1" / "test_full.parquet"
    test_ids = {row["source_id"] for row in pq.read_table(test_path, columns=["source_id"]).to_pylist()}
    documents_path = root / "data" / "pmc_pilot_v1" / "documents.jsonl.gz"
    with gzip.open(documents_path, "rt", encoding="utf-8") as file:
        documents = [json.loads(line) for line in file]
    rows = []
    for doc in documents:
        if doc["source_id"] not in test_ids:
            continue
        available = chunks(doc["body"])
        if not available:
            continue
        indices = sorted(set([0, len(available) // 2, len(available) - 1]))
        for index in indices:
            value = available[index]
            rows.append({
                "text_id": f"{doc['source_id']}:body:{index}",
                "text": value,
                "label": 0,
                "source": "pmc_oa_body",
                "source_id": doc["source_id"],
                "license": doc["license"],
                "publication_date": doc["publication_date"],
                "text_sha256": hashlib.sha256(value.encode()).hexdigest(),
            })
    rows.sort(key=lambda row: row["text_id"])
    output = root / "data" / "pmc_body_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "human_test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    manifest = {
        "source": "pmc_pilot_v1; papers in pmc_pyramid_v1/test_full only",
        "rows": len(rows),
        "papers": len({row["source_id"] for row in rows}),
        "rights": "per-item CC BY 4.0",
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_pyramid_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
