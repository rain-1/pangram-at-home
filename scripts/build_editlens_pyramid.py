"""Freeze nested, balanced binary EditLens research splits.

EditLens is CC BY-NC-SA 4.0 and must not be mixed into the reusable corpus.
AI-edited examples are kept in the raw download for a later, separate task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = "pangram/editlens_iclr"
REVISION = "34ac1ade5a814c4cea098ef70328d92880d1f1c0"
TIERS = {
    "train": {"tiny": 1000, "small": 5000, "medium": 20000, "large": 37600},
    "val": {"tiny": 200, "small": 800, "full": 1506},
    "test": {"tiny": 200, "small": 1000, "full": 4000},
    "test_enron": {"tiny": 200, "small": 1000, "full": 3600},
}
COLS = ["text_id", "text", "text_type", "model", "source", "source_id"]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_split(raw_dir: Path, split: str) -> list[dict]:
    files = sorted((raw_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet in {raw_dir / 'data'}")
    rows = []
    for file in files:
        table = pq.read_table(file, columns=COLS)
        for row in table.to_pylist():
            if row["text_type"] not in {"human_written", "ai_generated"}:
                continue
            text = row["text"]
            if not text or len(text.strip()) < 100:
                continue
            label = 0 if row["text_type"] == "human_written" else 1
            row["label"] = label
            row["group_id"] = f'{row["source"]}:{row["source_id"]}'
            row["text_sha256"] = digest(" ".join(text.casefold().split()))
            rows.append(row)
    return rows


def select(rows: list[dict], per_class: int, seed: str) -> list[dict]:
    selected = []
    for label in (0, 1):
        candidates = [r for r in rows if r["label"] == label]
        candidates.sort(key=lambda r: digest(f'{seed}:{label}:{r["text_id"]}'))
        if len(candidates) < per_class:
            raise ValueError(f"Need {per_class} label={label} rows, found {len(candidates)}")
        selected.extend(candidates[:per_class])
    selected.sort(key=lambda r: digest(f'{seed}:output:{r["text_id"]}'))
    return selected


def write_parquet(path: Path, rows: list[dict]) -> str:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    root = args.root
    raw_dir = root / "data" / "editlens_iclr"
    out = root / "data" / "editlens_pyramid_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": REPO,
        "revision": REVISION,
        "license": "CC BY-NC-SA 4.0; noncommercial research only",
        "seed": "editlens-pyramid-v1",
        "label_map": {"human": 0, "ai_generated": 1},
        "excludes": ["ai_edited"],
        "splits": {},
    }
    seen_text = set()
    # Protect the frozen evaluation sets before removing leakage from training.
    split_order = ["test", "test_enron", "val", "train"]
    all_rows = {}
    for split in split_order:
        rows = read_split(raw_dir, split)
        kept = []
        for row in rows:
            if row["text_sha256"] not in seen_text:
                kept.append(row)
                seen_text.add(row["text_sha256"])
        all_rows[split] = kept
    for split in ["train", "val", "test", "test_enron"]:
        rows = all_rows[split]
        available = min(sum(r["label"] == 0 for r in rows), sum(r["label"] == 1 for r in rows))
        manifest["splits"][split] = {}
        for tier, requested in TIERS[split].items():
            per_class = min(requested // 2, available)
            subset = select(rows, per_class, "editlens-pyramid-v1")
            path = out / f"{split}_{tier}.parquet"
            sha = write_parquet(path, subset)
            manifest["splits"][split][tier] = {
                "path": path.name,
                "rows": len(subset),
                "human": per_class,
                "ai": per_class,
                "sha256": sha,
                "sources": {source: sum(r["source"] == source for r in subset) for source in sorted({r["source"] for r in subset})},
            }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
