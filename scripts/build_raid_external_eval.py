"""Freeze a small, model-stratified RAID holdout excluded from training."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import heapq
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PER_DOMAIN = 100
PER_MODEL_CANDIDATES = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(row: dict) -> int:
    return int(hashlib.sha256(("raid-eval-v1:" + row["id"]).encode()).hexdigest()[:16], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    args = parser.parse_args()
    root = args.root / "data/raid_external_v1"
    source = root / "raw/train_none.csv"
    heaps = collections.defaultdict(list)
    counts = collections.Counter()
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            domain = row["domain"]
            model = row["model"]
            text = row["generation"].strip()
            words = len(text.split())
            if row["attack"] != "none" or not 20 <= words <= 640:
                continue
            if domain in {"code", "czech_news", "german_news"}:
                continue
            key = (domain, model)
            counts[key] += 1
            score = -rank(row)
            heap = heaps[key]
            item = (score, row["id"], row)
            limit = PER_DOMAIN if model == "human" else PER_MODEL_CANDIDATES
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    domains = sorted({domain for domain, _ in heaps})
    chosen = []
    for domain in domains:
        humans = sorted((item[2] for item in heaps.get((domain, "human"), [])), key=rank)
        models = sorted(model for d, model in heaps if d == domain and model != "human")
        generators = {model: sorted((item[2] for item in heaps[(domain, model)]), key=rank) for model in models}
        ai = []
        while len(ai) < PER_DOMAIN:
            advanced = False
            for model in models:
                if generators[model] and len(ai) < PER_DOMAIN:
                    ai.append(generators[model].pop(0))
                    advanced = True
            if not advanced:
                break
        if len(humans) < PER_DOMAIN or len(ai) < PER_DOMAIN:
            print(f"skipping {domain}: human={len(humans)} ai={len(ai)}")
            continue
        for row in humans[:PER_DOMAIN] + ai[:PER_DOMAIN]:
            text = row["generation"].strip()
            chosen.append({"text_id": "raid:" + row["id"], "text": text,
                           "label": int(row["model"] != "human"), "source": domain,
                           "source_id": row["source_id"], "group_id": row["source_id"],
                           "generator": row["model"], "decoding": row["decoding"],
                           "text_sha256": hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()})
    hashes = [row["text_sha256"] for row in chosen]
    assert len(hashes) == len(set(hashes))
    for split in ("train", "val", "test"):
        path = args.root / f"data/diverse_pyramid_v1/{split}_full.parquet"
        existing = set(pq.read_table(path, columns=["text_sha256"]).to_pydict()["text_sha256"])
        assert not existing & set(hashes), f"RAID overlaps diverse {split}"
    chosen.sort(key=lambda row: rank({"id": row["text_id"]}))
    out = root / "frozen.parquet"
    pq.write_table(pa.Table.from_pylist(chosen), out, compression="zstd")
    manifest = {"source": "RAID official train_none.csv, used only for external evaluation",
                "source_url": "https://dataset.raid-bench.xyz/train_none.csv",
                "raw_sha256": sha256(source), "output_sha256": sha256(out),
                "rows": len(chosen), "by_domain_label": dict(sorted((f"{d}:{l}", sum(r["source"] == d and r["label"] == l for r in chosen))
                                                            for d in domains for l in (0, 1))),
                "generator_count": len({r["generator"] for r in chosen if r["label"] == 1}),
                "role": "external evaluation; no RAID rows in diverse training or validation"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(chosen), "generator_count": manifest["generator_count"],
                      "by_domain_label": manifest["by_domain_label"]}, indent=2))


if __name__ == "__main__":
    main()
