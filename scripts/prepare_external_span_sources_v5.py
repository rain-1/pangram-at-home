"""Normalize pinned span corpora while preserving their source roles and splits.

The output is a candidate corpus. This script does not change the current
training mix or the locked evaluation files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


LLMTRACE_REV = "332053804f8798177ff2ecb1148e1839ed429ba3"
AITDNA_REV = "21d77f7d44833507750c0da309295c4147c1bda5"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def spans_from_ai_intervals(text: str, intervals: list[list[int]]) -> list[dict]:
    previous = 0
    out: list[dict] = []
    for start, end in intervals:
        if not isinstance(start, int) or not isinstance(end, int) or not previous <= start < end <= len(text):
            raise ValueError(f"Invalid AI interval {(start, end)} for text length {len(text)}")
        if previous < start:
            out.append({"start": previous, "end": start, "label": 0})
        out.append({"start": start, "end": end, "label": 1})
        previous = end
    if previous < len(text):
        out.append({"start": previous, "end": len(text), "label": 0})
    if not out or out[0]["start"] != 0 or out[-1]["end"] != len(text):
        raise ValueError("Spans do not cover document")
    if any(left["end"] != right["start"] for left, right in zip(out, out[1:])):
        raise ValueError("Span coverage has a gap")
    return out


def aitdna_document(source: dict, index: int) -> dict:
    metadata = source["metadata"]
    author_hash = sha(metadata["author"])[:24]
    text = ""
    spans = []
    for item in source["data"]:
        part = item["text"]
        role = item["author"]
        if role not in ("User", "Bot") or not part:
            raise ValueError(f"AITDNA {index}: unsupported role or empty span")
        start = len(text)
        text += part
        label = 0 if role == "User" else 1
        if spans and spans[-1]["label"] == label:
            spans[-1]["end"] = len(text)
        else:
            spans.append({"start": start, "end": len(text), "label": label})
    if bool(metadata["human_only"]) and any(span["label"] == 1 for span in spans):
        raise ValueError(f"AITDNA {index}: human-only flag has Bot text")
    if not text:
        raise ValueError(f"AITDNA {index}: empty document")
    labels = {span["label"] for span in spans}
    kind = "mixed" if labels == {0, 1} else "ai" if labels == {1} else "human"
    return {"id": f"aitdna_span:{index}:{sha(text)[:16]}", "text": text,
            "spans": spans, "source": "AITDNA", "domain": metadata["task"],
            "kind": kind,
            "construction": "user_study_surviving_edit_provenance",
            "author_group": f"aitdna_author:{author_hash}",
            "group_id": f"aitdna_author:{author_hash}",
            "human_only": bool(metadata["human_only"]),
            "generator": metadata.get("model"), "source_revision": AITDNA_REV,
            "source_view": "span", "source_row": index,
            "label_definition": "surviving User/Bot characters in authors' span projection",
            "text_sha256": sha(text)}


def convert_llmtrace(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True)
    manifest = {"source_revision": LLMTRACE_REV, "language": "eng",
                "role": "candidate train/validation/test; not merged into active detector",
                "splits": {}}
    all_groups = {}
    all_hashes = {}
    for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        source_path = source / f"{source_split}.jsonl"
        output_path = destination / f"{target_split}.jsonl"
        counts = Counter()
        groups = set()
        hashes = set()
        with source_path.open() as original, output_path.open("w") as output:
            for line_number, line in enumerate(original, start=1):
                row = json.loads(line)
                if row["lang"] != "eng":
                    continue
                text = row["text"]
                intervals = row["ai_char_intervals"]
                try:
                    spans = spans_from_ai_intervals(text, intervals)
                except ValueError:
                    # Never clip or shift supplied supervision to make it fit.
                    counts[f"excluded_invalid_intervals:{row['lang']}"] += 1
                    continue
                label = row["label"]
                if (label == "human" and intervals or label == "ai" and intervals != [[0, len(text)]]
                        or label == "mixed" and (not intervals or intervals == [[0, len(text)]])):
                    counts["excluded_label_interval_mismatch"] += 1
                    continue
                gid = f"llmtrace:{row['topic_id']}"
                thash = sha(text)
                normalized = {"id": f"llmtrace:{source_split}:{line_number}",
                              "text": text, "spans": spans,
                              "source": "LLMTrace_detection", "domain": row["data_type"],
                              "kind": label,
                              "construction": row.get("prompt_type") or f"source_{label}",
                              "group_id": gid, "generator": row.get("model"),
                              "prompt_type": row.get("prompt_type"),
                              "source_label": label, "source_revision": LLMTRACE_REV,
                              "source_split": source_split, "source_line": line_number,
                              "text_sha256": thash}
                output.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                counts["documents"] += 1
                counts[f"label:{label}"] += 1
                counts[f"domain:{row['data_type']}"] += 1
                counts["multiple_ai_intervals"] += len(intervals) > 1
                counts["shorter_than_80_words"] += len(text.split()) < 80
                counts["human_characters"] += sum(s["end"]-s["start"] for s in spans if s["label"] == 0)
                counts["ai_characters"] += sum(s["end"]-s["start"] for s in spans if s["label"] == 1)
                groups.add(gid)
                hashes.add(thash)
        manifest["splits"][target_split] = {
            "source_sha256": sha(source_path.read_bytes()),
            "output_sha256": sha(output_path.read_bytes()),
            "topic_groups": len(groups), "exact_texts": len(hashes),
            "counts": dict(counts)}
        all_groups[target_split], all_hashes[target_split] = groups, hashes
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if all_groups[left] & all_groups[right] or all_hashes[left] & all_hashes[right]:
            raise ValueError(f"LLMTrace source splits overlap: {left}, {right}")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def convert_aitdna(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True)
    source_path = source / "span/test-00000-of-00001.parquet"
    rows = pq.read_table(source_path).to_pylist()
    output_path = destination / "locked_test.jsonl"
    counts = Counter()
    authors = defaultdict(int)
    with output_path.open("w") as output:
        for index, source_row in enumerate(rows):
            row = aitdna_document(source_row, index)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            authors[row["author_group"]] += 1
            counts["documents"] += 1
            counts[f"task:{row['domain']}"] += 1
            counts["human_only"] += row["human_only"]
            counts["mixed_with_bot"] += any(span["label"] == 1 for span in row["spans"])
            counts["human_characters"] += sum(span["end"] - span["start"] for span in row["spans"] if span["label"] == 0)
            counts["ai_characters"] += sum(span["end"] - span["start"] for span in row["spans"] if span["label"] == 1)
    manifest = {"source_revision": AITDNA_REV,
                "role": "locked external test, no training or threshold calibration",
                "definition": "User/Bot surviving-character span view; all configurations are projections of the same 362 documents",
                "source_sha256": sha(source_path.read_bytes()),
                "output_sha256": sha(output_path.read_bytes()),
                "author_groups": len(authors), "counts": dict(counts)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home/data/span_sources_v5"))
    args = parser.parse_args()
    llm_dest = args.root / "normalized_llmtrace_en"
    ait_dest = args.root / "normalized_aitdna_real"
    if llm_dest.exists() or ait_dest.exists():
        raise SystemExit("Refusing to overwrite normalized corpora")
    llm = convert_llmtrace(args.root / "llmtrace", llm_dest)
    ait = convert_aitdna(args.root / "aitdna", ait_dest)
    print(json.dumps({"llmtrace": {s: data["counts"]["documents"] for s, data in llm["splits"].items()},
                      "aitdna": ait["counts"]}, indent=2))


if __name__ == "__main__":
    main()
