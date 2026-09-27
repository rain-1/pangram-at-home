"""Normalize a small pinned HF source set into candidate span JSONL.

The downloaded source snapshots and normalized output live under
/mnt/f/pangram-at-home/data/candidate_span_sources/ai/ and are not committed.
OpAI has direct character span supervision. GEN provides full-document
human/AI-generation pairs; its edit rows stay raw because the released card
does not claim span annotations for them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack


BASE = Path("/mnt/f/pangram-at-home/data/candidate_span_sources/ai")
GEN_TRAIN_PHRASE_HIT_GROUPS = {
    "118", "1379", "1464", "1531", "4025", "4107", "4120", "4196",
    "4240", "4354", "4435", "4542", "4557", "4587", "4594", "464",
    "4667", "467", "4856", "776", "780",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def span_partition(text: str, ai_spans: list[list[int]]) -> list[dict]:
    """Turn sorted [start,end) AI offsets into full-coverage binary spans."""
    result: list[dict] = []
    pos = 0
    for pair in ai_spans:
        if len(pair) != 2:
            raise ValueError(f"Invalid character span: {pair!r}")
        start, end = map(int, pair)
        if not (pos <= start < end <= len(text)):
            raise ValueError(f"Out-of-order/out-of-range span {pair!r}, len={len(text)}")
        if pos < start:
            result.append({"start": pos, "end": start, "label": 0})
        result.append({"start": start, "end": end, "label": 1})
        pos = end
    if pos < len(text):
        result.append({"start": pos, "end": len(text), "label": 0})
    if not result and text:
        result = [{"start": 0, "end": len(text), "label": 0}]
    assert all(text[s["start"]:s["end"]] for s in result)
    assert (not text and not result) or (result[0]["start"] == 0 and result[-1]["end"] == len(text))
    assert all(a["end"] == b["start"] for a, b in zip(result, result[1:]))
    return result


def normalize_opai(src: Path, out: Path) -> dict:
    count = 0
    versions: dict[str, int] = {}
    groups: set[str] = set()
    with src.open(newline="", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for row in csv.DictReader(fin):
            if row["split"] != "train" or row["domain"] != "reports" or row["generator"] != "gpt-5.4":
                raise ValueError("Unexpected non-train or non-report row in pinned OpAI source file")
            text = row["text"]
            raw_spans = json.loads(row["ai_spans_char"])
            spans = span_partition(text, raw_spans)
            ai_chars = sum(end - start for start, end in raw_spans)
            group = f"reports:{row['id']}"
            groups.add(group)
            record = {
                "id": f"opai:{row['record_id']}",
                "text": text,
                "spans": spans,
                "kind": "ai" if text and ai_chars == len(text) else ("mixed" if raw_spans else "human"),
                "construction": "source_revision_trajectory",
                "source": "opai_bench:reports",
                "domain": "reports",
                "source_ids": [row["id"]],
                "source_groups": [group],
                "group_id": group,
                "source_split": row["split"],
                "generator": row["generator"],
                "version": row["version"],
                "version_index": int(row["version_index"]),
                "edit_operation": row["edit_operation"],
                "document_hash_id": row["document_hash_id"],
                "text_sha256": sha256_text(text),
                "ai_char_spans_source": raw_spans,
                "license": "Apache-2.0 (dataset card); source-material rights/terms remain source-dependent",
                "source_url": "https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            versions[row["version"]] = versions.get(row["version"], 0) + 1
    return {"rows": count, "source_groups": len(groups), "versions": versions}


def gen_group_split(prompt_id: str) -> str:
    # Stable whole-prompt split: 80/10/10 by first SHA256 byte.
    bucket = hashlib.sha256(prompt_id.encode()).digest()[0] % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def normalize_gen(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {s: {"rows": 0, "groups": set(), "label_model_domain": {}} for s in ("train", "validation", "test")}
    files = (("human.jsonl", 0), ("ai_generated.jsonl", 1))
    with ExitStack() as stack:
        outputs = {split: stack.enter_context((out_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
                   for split in ("train", "validation", "test")}
        for filename, expected_label in files:
            with (raw_dir / filename).open(encoding="utf-8") as fin:
                for row_index, line in enumerate(fin, 1):
                    row = json.loads(line)
                    if int(row["label"]) != expected_label:
                        raise ValueError(f"Unexpected label in {filename}: {row['label']}")
                    text = row["text"]
                    prompt_id = str(row["prompt_id"])
                    split = gen_group_split(prompt_id)
                    if split == "train" and prompt_id in GEN_TRAIN_PHRASE_HIT_GROUPS:
                        continue
                    record = {
                        "id": f"gen:{row['text_type']}:{prompt_id}:{row['model']}:{row_index}",
                        "text": text,
                        "spans": ([{"start": 0, "end": len(text), "label": expected_label}] if text else []),
                        "kind": "ai" if expected_label else "human",
                        "construction": "paired_prompt_full_document",
                        "source": f"gen:{row['source']}",
                        "domain": row["source"],
                        "source_ids": [f"{row['source']}:{prompt_id}"],
                        "source_groups": [f"gen-prompt:{prompt_id}"],
                        "group_id": f"gen-prompt:{prompt_id}",
                        "source_split": split,
                        "prompt_id": prompt_id,
                        "prompt": row.get("prompt"),
                        "generator": row["model"] if expected_label else "human",
                        "text_sha256": sha256_text(text),
                        "license": "CC-BY-NC-4.0; human seed source licenses may differ",
                        "source_url": "https://huggingface.co/datasets/szyszy/GEN",
                    }
                    outputs[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    slot = stats[split]
                    slot["rows"] += 1
                    slot["groups"].add(prompt_id)
                    key = f"{row['source']}|{row['model']}|{expected_label}"
                    slot["label_model_domain"][key] = slot["label_model_domain"].get(key, 0) + 1
    for slot in stats.values():
        slot["groups"] = len(slot["groups"])
    stats["excluded_current_train_phrase_groups"] = sorted(GEN_TRAIN_PHRASE_HIT_GROUPS)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    opai_dir = args.base / "opai_bench/default_train_reports_gpt54"
    gen_dir = args.base / "gen_noncommercial"
    opai_output = opai_dir / "normalized_train.jsonl"
    gen_output = gen_dir / "normalized_by_prompt_split"
    result = {
        "opai_reports_gpt54_train": normalize_opai(opai_dir / "reports_gpt-5.4.csv", opai_output),
        "gen_full_document_pairs": normalize_gen(gen_dir, gen_output),
    }
    manifest = args.base / "normalized_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
