"""Check the frozen split contracts before training or publishing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

FOLDERS = {
    "editlens": "editlens_pyramid_v1",
    "pmc": "pmc_pyramid_v1",
    "paper": "paper_pyramid_v1",
    "mixed": "mixed_pyramid_v1",
    "cross": "paper_cross_model_test_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(FOLDERS))
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--frozen-dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    data = args.root / "data" / FOLDERS[args.dataset]
    manifest = json.loads((data / "manifest.json").read_text())
    frozen = args.frozen_dir / f"{FOLDERS[args.dataset]}.json"
    if frozen.exists():
        assert manifest == json.loads(frozen.read_text()), f"Manifest differs from frozen {frozen}"
    if args.dataset == "cross":
        path = data / "test.parquet"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["test_sha256"]
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == manifest["rows"]
        groups = {}
        for row in rows:
            groups.setdefault(row["source_id"], set()).add(row["label"])
        assert len(groups) == manifest["pairs"]
        assert all(labels == {0, 1} for labels in groups.values())
        print(json.dumps({"dataset": "cross", "passed": True, "pairs": len(groups)}, indent=2))
        return
    full_groups = {}
    full_hashes = {}
    full_venues = {}
    summary = {}
    for split, tiers in manifest["splits"].items():
        previous = set()
        for tier, info in tiers.items():
            path = data / info["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], path
            table = pq.read_table(path)
            rows = table.to_pylist()
            ids = {row["text_id"] for row in rows}
            assert len(ids) == len(rows), (split, tier, "duplicate IDs")
            assert len(rows) == info["rows"], (split, tier, "row count")
            assert sum(row["label"] == 0 for row in rows) == sum(row["label"] == 1 for row in rows), (split, tier, "balance")
            assert previous <= ids, (split, tier, "non-nested tier")
            previous = ids
            if args.dataset == "mixed":
                for label in (0, 1):
                    subset = [row for row in rows if row["label"] == label]
                    paper = sum(row["genre"] == "paper_abstract" for row in subset)
                    assert abs(paper / len(subset) - 0.35) <= 1 / len(subset)
            summary[f"{split}_{tier}"] = len(rows)
        full = pq.read_table(data / list(tiers.values())[-1]["path"]).to_pylist()
        full_groups[split] = {row["group_id"] for row in full}
        if args.dataset in {"pmc", "paper"}:
            venue_key = "journal" if args.dataset == "pmc" else "venue"
            full_venues[split] = {
                (row["source"], row[venue_key]) for row in full
            }
        full_hashes[split] = {
            hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).hexdigest()
            for row in full
        }
    splits = list(full_groups)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            assert not full_groups[left] & full_groups[right], (left, right, "group leakage")
            assert not full_hashes[left] & full_hashes[right], (left, right, "exact text leakage")
    if args.dataset in {"pmc", "paper"}:
        assert not full_venues["train"] & full_venues["test"], "venue leakage into paper test"
    print(json.dumps({"dataset": args.dataset, "passed": True, "rows": summary}, indent=2))


if __name__ == "__main__":
    main()
