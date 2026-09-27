#!/usr/bin/env python3
"""Normalize an OpAI-Bench training CSV into the project's candidate span shape.

This is an acquisition helper only. It does not feed a training pipeline.
The caller supplies the raw CSV and output JSONL paths; the raw file stays
outside Git. Character offsets are preserved as zero-based, end-exclusive spans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCE = "OpAI-Bench1/OpAI-Bench"
REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
LICENSE = "Apache-2.0 (dataset card); upstream seed-text terms require per-source review"


def normalize(
    raw_csv: Path, output_jsonl: Path, excluded_record_ids_file: Path | None = None
) -> dict[str, int | str]:
    counts: dict[str, int] = {
        "rows": 0,
        "human_seed": 0,
        "mixed": 0,
        "ai_only": 0,
        "excluded_invalid_offsets": 0,
        "excluded_overlap_group_rows": 0,
    }
    raw_digest = hashlib.sha256()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            raw_digest.update(block)

    excluded_record_ids: set[str] = set()
    excluded_groups: set[str] = set()
    if excluded_record_ids_file:
        excluded_record_ids = set(json.loads(excluded_record_ids_file.read_text())["record_ids"])
        with raw_csv.open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row["record_id"] in excluded_record_ids:
                    excluded_groups.add(row["document_hash_id"])

    with raw_csv.open("r", encoding="utf-8", newline="") as source, output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for row in csv.DictReader(source):
            if row["document_hash_id"] in excluded_groups:
                counts["excluded_overlap_group_rows"] += 1
                continue
            text = row["text"]
            char_spans = json.loads(row["ai_spans_char"] or "[]")
            ai_spans: list[tuple[int, int]] = []
            prev_end = 0
            invalid = False
            for pair in char_spans:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"Malformed span for record {row['record_id']}: {pair!r}")
                start, end = map(int, pair)
                if not (0 <= start < end <= len(text)) or start < prev_end:
                    invalid = True
                    break
                ai_spans.append((start, end))
                prev_end = end
            if invalid:
                counts["excluded_invalid_offsets"] += 1
                continue

            # The project intake schema expects a full text partition. Fill
            # unlabeled gaps as human-origin, retaining AI ranges as label 1.
            spans: list[dict[str, int]] = []
            cursor = 0
            for start, end in ai_spans:
                if cursor < start:
                    spans.append({"start": cursor, "end": start, "label": 0})
                spans.append({"start": start, "end": end, "label": 1})
                cursor = end
            if cursor < len(text):
                spans.append({"start": cursor, "end": len(text), "label": 0})
            if not spans and text:
                spans = [{"start": 0, "end": len(text), "label": 0}]

            ratio = float(row["ai_char_ratio"])
            kind = "human" if ratio == 0 else "ai" if ratio == 1 else "mixed"
            bucket = {"human": "human_seed", "mixed": "mixed", "ai": "ai_only"}[kind]
            counts[bucket] += 1
            counts["rows"] += 1
            record = {
                "id": row["record_id"],
                "text": text,
                "spans": spans,
                "kind": kind,
                "source": SOURCE,
                "source_revision": REVISION,
                "source_license": LICENSE,
                "group_id": row["document_hash_id"],
                "source_document_id": row["id"],
                "domain": row["domain"],
                "generator": row["generator"],
                "version": row["version"],
                "split": row["split"],
                "span_unit": "character",
                "span_convention": "0-based, end-exclusive",
                "span_label_meaning": "1=AI-generated text and 0=carried human-seed text in a synthetic edit trajectory; not observed human/AI keystroke provenance",
                "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts["raw_sha256"] = raw_digest.hexdigest()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--exclude-record-ids-file",
        type=Path,
        help="JSON object with a record_ids array; all rows from those source document groups are excluded",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            normalize(args.raw_csv, args.output_jsonl, args.exclude_record_ids_file), indent=2
        )
    )


if __name__ == "__main__":
    main()
