"""Build nested 50/50 human-AI splits with 35% paper content.

The general-text component is gated EditLens CC BY-NC-SA. These mixed splits
are for local noncommercial research, not a redistributable training corpus.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PAPER_SHARE = 0.35


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def paper_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[row["group_id"]][row["label"]] = row
    result = [((v[0]), (v[1])) for v in groups.values() if set(v) == {0, 1}]
    result.sort(key=lambda pair: digest("mixed-paper-v1:" + pair[0]["group_id"]))
    return result


def normalize_paper(row: dict) -> dict:
    return {
        "text_id": "paper:" + row["text_id"],
        "text": row["text"], "label": row["label"],
        "source": row["source"], "source_id": row["source_id"],
        "group_id": "paper:" + row["group_id"],
        "genre": "paper_abstract",
        "license": row["license"],
        "generator": row["generator"],
    }


def normalize_general(row: dict) -> dict:
    return {
        "text_id": "editlens:" + row["text_id"],
        "text": row["text"], "label": row["label"],
        "source": row["source"], "source_id": row["source_id"],
        "group_id": "editlens:" + row["group_id"],
        "genre": "general",
        "license": "CC BY-NC-SA 4.0 research benchmark",
        "generator": row["model"],
    }


def write(path: Path, paper: list[tuple[dict, dict]], general: dict[int, list[dict]], total_pairs: int) -> dict:
    paper_count = round(total_pairs * PAPER_SHARE)
    general_count = total_pairs - paper_count
    if paper_count > len(paper) or any(general_count > len(general[label]) for label in (0, 1)):
        raise ValueError("Requested tier exceeds available source data")
    rows = [normalize_paper(row) for pair in paper[:paper_count] for row in pair]
    rows += [normalize_general(row) for label in (0, 1) for row in general[label][:general_count]]
    rows.sort(key=lambda row: digest("mixed-output-v1:" + row["text_id"]))
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {
        "path": path.name,
        "pairs": total_pairs,
        "rows": len(rows),
        "human": total_pairs,
        "ai": total_pairs,
        "paper_pairs": paper_count,
        "general_pairs": general_count,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    root = args.root
    output = root / "data" / "mixed_pyramid_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "paper_pyramid_v1 + editlens_pyramid_v1",
        "license": "local noncommercial research; EditLens CC BY-NC-SA 4.0",
        "paper_share_target": PAPER_SHARE,
        "label_map": {"human": 0, "ai": 1},
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        paper = paper_pairs(load(root / "data" / "paper_pyramid_v1" / f"{split}_full.parquet"))
        general_rows = load(root / "data" / "editlens_pyramid_v1" / f"{split}_full.parquet" if split != "train" else
                            root / "data" / "editlens_pyramid_v1" / "train_large.parquet")
        general = {}
        for label in (0, 1):
            general[label] = [row for row in general_rows if row["label"] == label]
            general[label].sort(key=lambda row: digest(f"mixed-general-v1:{split}:{label}:" + row["text_id"]))
        maximum = min(
            len(paper) / PAPER_SHARE,
            len(general[0]) / (1 - PAPER_SHARE),
            len(general[1]) / (1 - PAPER_SHARE),
        )
        full_pairs = int(maximum)
        while round(full_pairs * PAPER_SHARE) > len(paper) or full_pairs - round(full_pairs * PAPER_SHARE) > min(len(general[0]), len(general[1])):
            full_pairs -= 1
        requested = {"tiny": 100, "small": 500}
        if split == "train":
            requested["medium"] = 1500
        requested["full"] = full_pairs
        tiers = {}
        for tier, count in requested.items():
            count = min(count, full_pairs)
            tiers[tier] = write(output / f"{split}_{tier}.parquet", paper, general, count)
        manifest["splits"][split] = tiers
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({s: {t: x["rows"] for t, x in tiers.items()} for s, tiers in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
