"""Freeze a source-disjoint synthetic confirmation set for the token pilot."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from build_span_pilot_v2 import document, pools


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("/mnt/f/pangram-at-home/data")
    parent = root / "diverse_pyramid_v1"
    output = root / "span_test_probe_v1"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    test_file = parent / "test_full.parquet"
    test = pq.read_table(test_file).to_pylist()
    for key in ("text_sha256", "group_id"):
        test_values = {row[key] for row in test}
        for split in ("train", "val"):
            earlier = pq.read_table(parent / f"{split}_full.parquet", columns=[key]).to_pydict()[key]
            assert not test_values.intersection(earlier), (key, split)
    by_source, paired = pools(test)
    rng = random.Random(243)
    schedule = (["human", "ai", "paired_mixed", "paired_mixed", "paired_mixed",
                 "same_source_mixed"] * 84)[:500]
    rng.shuffle(schedule)
    rows = [document("test", index, kind, by_source, paired, rng, "varied")
            for index, kind in enumerate(schedule)]
    output.mkdir(parents=True)
    target = output / "test.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {"role": "synthetic token-pilot confirmation; parent diverse test was previously examined",
                "labels": {"human": 0, "ai_generated": 1},
                "ai_assisted_supported": False,
                "parent_sha256": sha(test_file), "sha256": sha(target),
                "documents": len(rows), "kinds": dict(Counter(row["kind"] for row in rows)),
                "constructions": dict(Counter(row["construction"] for row in rows)),
                "seed": 243, "separator_policy": "varied"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
