"""Build a held-out paper test with generators swapped across source domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_paper_pyramid import GENERATOR_REVISIONS, normalize


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    root = args.root
    standard_test = root / "data" / "paper_pyramid_v1" / "test_full.parquet"
    humans = {
        row["source_id"]: row
        for row in pq.read_table(standard_test).to_pylist() if row["label"] == 0
    }
    generator_paths = [
        root / "data" / "pmc_pilot_v1" / "generated_smollm.jsonl",
        root / "data" / "acl_abstracts_v1" / "generated_qwen.jsonl",
    ]
    rows = []
    rejected = 0
    for path in generator_paths:
        with path.open(encoding="utf-8") as file:
            generations = [json.loads(line) for line in file]
        for g in generations:
            human = humans.get(g["source_id"])
            if human is None:
                continue
            ai = normalize(g["text"], human["title"])
            ratio = len(ai.split()) / len(human["text"].split())
            if len(ai.split()) < 80 or not (0.6 <= ratio <= 1.6) or digest(ai.casefold()) == digest(human["text"].casefold()):
                rejected += 1
                continue
            rows.append(human)
            rows.append({
                **human,
                "text_id": human["source_id"] + ":cross-ai",
                "text": ai,
                "text_sha256": digest(ai.casefold()),
                "label": 1,
                "generator": g["model"],
                "generator_revision": GENERATOR_REVISIONS[g["model"]],
                "prompt_version": g["prompt_version"],
            })
    rows.sort(key=lambda row: row["text_id"])
    assert len(rows) % 2 == 0
    assert {row["source_id"] for row in rows} <= set(humans)
    assert all(
        row["generator"] == ("HuggingFaceTB/SmolLM2-1.7B-Instruct" if row["source"] == "pmc_oa" else "Qwen/Qwen2.5-0.5B-Instruct")
        for row in rows if row["label"] == 1
    )
    output = root / "data" / "paper_cross_model_test_v1"
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, compression="zstd")
    manifest = {
        "source": "paper_pyramid_v1/test_full with Qwen/SmolLM generators swapped",
        "purpose": "generator shift evaluation only; human works already occur in standard paper test",
        "pairs": len(rows) // 2,
        "rows": len(rows),
        "rejected_for_length": rejected,
        "test_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "standard_test_sha256": hashlib.sha256(standard_test.read_bytes()).hexdigest(),
        "generation_sha256": {path.name + ":" + path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in generator_paths},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
