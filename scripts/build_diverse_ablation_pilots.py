"""Create equal-size, balanced 4k-row training mixes for data ablations.

All examples come from the existing training split. Validation is copied
unchanged, and no test data is read. Raw text stays outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_diverse_pyramid import SHARES


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quotas(weights: dict[str, float], count: int) -> dict[str, int]:
    total = sum(weights.values())
    exact = {name: count * weight / total for name, weight in weights.items()}
    result = {name: int(value) for name, value in exact.items()}
    for name in sorted(weights, key=lambda key: (exact[key] - result[key], key), reverse=True):
        if sum(result.values()) == count:
            break
        result[name] += 1
    assert sum(result.values()) == count
    return result


def mixes() -> dict[str, dict[str, float]]:
    result = {"control_35paper": SHARES}
    for category in SHARES:
        result[f"without_{category}"] = {k: v for k, v in SHARES.items() if k != category}
    for paper_share in (.20, .50):
        result[f"paper_{int(paper_share * 100)}pct"] = {
            k: paper_share if k == "paper" else v * (1 - paper_share) / (1 - SHARES["paper"])
            for k, v in SHARES.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--pairs", type=int, default=2000)
    args = parser.parse_args()
    source = args.root / "data/diverse_pyramid_v1"
    train_path = source / "train_full.parquet"
    val_path = source / "val_full.parquet"
    rows = pq.read_table(train_path).to_pylist()
    pool = {(domain, label): sorted((row for row in rows if row["domain"] == domain and row["label"] == label),
                                    key=lambda row: digest("ablation-v1:" + row["text_id"]))
            for domain in SHARES for label in (0, 1)}
    for name, weights in mixes().items():
        quota = quotas(weights, args.pairs)
        selected = []
        for domain, pairs in quota.items():
            for label in (0, 1):
                candidates = pool[(domain, label)]
                if pairs > len(candidates):
                    raise ValueError(f"{name} needs {pairs} {domain}/{label} rows; only {len(candidates)} available")
                selected.extend(candidates[:pairs])
        selected.sort(key=lambda row: digest("ablation-v1-output:" + row["text_id"]))
        assert len(selected) == args.pairs * 2
        assert len({row["text_sha256"] for row in selected}) == len(selected)
        output = args.root / "data" / f"diverse_ablation_v1_{name}"
        output.mkdir(parents=True, exist_ok=True)
        train_out = output / "train_full.parquet"
        pq.write_table(pa.Table.from_pylist(selected), train_out, compression="zstd")
        val_out = output / "val_full.parquet"
        if not val_out.exists():
            val_out.write_bytes(val_path.read_bytes())
        assert file_hash(val_out) == file_hash(val_path)
        manifest = {"name": name, "rows": len(selected), "human": args.pairs, "ai": args.pairs,
                    "category_pairs": quota, "source_train_sha256": file_hash(train_path),
                    "train_sha256": file_hash(train_out), "val_sha256": file_hash(val_out),
                    "selection": "deterministic category/label stratification of training rows; fixed row count",
                    "test_data_used": False}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(name, quota, flush=True)


if __name__ == "__main__":
    main()
