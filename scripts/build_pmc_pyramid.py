"""Freeze balanced paper-abstract splits from licensed PMC and local generation."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(text: str, *, generated: bool = False) -> str:
    text = text.strip()
    if generated:
        # The small instruction model sometimes emits a paper title or an
        # "Abstract" heading despite the prompt. Remove the format wrapper.
        text = re.sub(r"(?im)^\s*(?:\*\*)?title\s*[:\-].*?(?:\n|$)", "", text, count=1)
    text = re.sub(r"(?i)\*\*\s*(abstract|background|methods?|results?|conclusions?)\s*\*\*\s*[:\-]?", "", text)
    text = re.sub(r"(?i)\b(abstract|background|methods?|results?|conclusions?)\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\n\t")


def write(path: Path, pairs: list[tuple[dict, dict]]) -> dict:
    rows = [row for pair in pairs for row in pair]
    rows.sort(key=lambda row: digest("paper-output-v1:" + row["text_id"]))
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {
        "path": path.name,
        "pairs": len(pairs),
        "rows": len(rows),
        "human": len(pairs),
        "ai": len(pairs),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--journal-cap", type=int, default=20)
    args = parser.parse_args()
    source = args.root / "data" / "pmc_pilot_v1"
    output = args.root / "data" / "pmc_pyramid_v1"
    output.mkdir(parents=True, exist_ok=True)
    docs = {d["source_id"]: d for d in map(json.loads, gzip.open(source / "documents.jsonl.gz", "rt", encoding="utf-8"))}
    generated = {g["source_id"]: g for g in map(json.loads, (source / "generated_qwen.jsonl").open(encoding="utf-8"))}
    candidates = []
    rejections = collections.Counter()
    for source_id, g in generated.items():
        doc = docs.get(source_id)
        if not doc or doc["license"] != "CC BY 4.0" or doc["publication_date"] > "2022-12-31":
            rejections["source_or_rights"] += 1
            continue
        if g["human_abstract_sha256"] != doc["abstract_sha256"]:
            rejections["source_hash"] += 1
            continue
        human = normalize(doc["abstract"])
        ai = normalize(g["text"], generated=True)
        hw, aw = len(human.split()), len(ai.split())
        if not (120 <= hw <= 280 and 0.6 <= aw / hw <= 1.6 and aw >= 80):
            rejections["length"] += 1
            continue
        if digest(human.casefold()) == digest(ai.casefold()):
            rejections["identical"] += 1
            continue
        candidates.append((doc, g, human, ai))
    candidates.sort(key=lambda x: digest("paper-selection-v1:" + x[0]["source_id"]))
    journal_counts = collections.Counter()
    pairs = []
    for doc, gen, human, ai in candidates:
        journal = doc["journal"] or "unknown"
        if journal_counts[journal] >= args.journal_cap:
            rejections["journal_cap"] += 1
            continue
        journal_counts[journal] += 1
        base = {
            "group_id": doc["source_id"],
            "source": "pmc_oa",
            "source_id": doc["source_id"],
            "canonical_url": doc["canonical_url"],
            "doi": doc["doi"],
            "journal": journal,
            "publication_date": doc["publication_date"],
            "license": doc["license"],
            "license_url": doc["license_url"],
        }
        pairs.append((
            {**base, "text_id": doc["source_id"] + ":human", "text": human, "label": 0, "generator": "human"},
            {**base, "text_id": doc["source_id"] + ":ai", "text": ai, "label": 1, "generator": gen["model"]},
        ))
    # A journal-disjoint test set reveals one form of domain shift.
    by_journal = collections.defaultdict(list)
    for pair in pairs:
        by_journal[pair[0]["journal"]].append(pair)
    target_test = round(len(pairs) * 0.18)
    test_journals = set()
    total_test = 0
    for journal in sorted(by_journal, key=lambda j: digest("paper-journal-test-v1:" + j)):
        if total_test >= target_test:
            break
        test_journals.add(journal)
        total_test += len(by_journal[journal])
    test = [pair for pair in pairs if pair[0]["journal"] in test_journals]
    remaining = [pair for pair in pairs if pair[0]["journal"] not in test_journals]
    remaining.sort(key=lambda pair: digest("paper-val-v1:" + pair[0]["source_id"]))
    val_count = round(len(pairs) * 0.18)
    val = remaining[:val_count]
    train = remaining[val_count:]
    manifest = {
        "source": "pmc_pilot_v1 + Qwen/Qwen2.5-0.5B-Instruct",
        "inputs_sha256": {
            "documents": hashlib.sha256((source / "documents.jsonl.gz").read_bytes()).hexdigest(),
            "generation": hashlib.sha256((source / "generated_qwen.jsonl").read_bytes()).hexdigest(),
        },
        "rights": "PMC human text: CC BY 4.0 per item; AI text: generated locally",
        "human_cutoff": "2022-12-31",
        "label_map": {"human": 0, "ai": 1},
        "journal_cap": args.journal_cap,
        "selection_rejections": dict(rejections),
        "test_journals": sorted(test_journals),
        "splits": {},
    }
    for name, content in [("train", train), ("val", val), ("test", test)]:
        tiers = {"tiny": content[:min(30, len(content))]}
        if name == "train":
            tiers["small"] = content[:min(150, len(content))]
        tiers["full"] = content
        manifest["splits"][name] = {}
        for tier, subset in tiers.items():
            manifest["splits"][name][tier] = write(output / f"{name}_{tier}.parquet", subset)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "candidates": len(candidates),
        "selected_pairs": len(pairs),
        "rejections": dict(rejections),
        "splits": {k: {t: v["pairs"] for t, v in x.items()} for k, x in manifest["splits"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
