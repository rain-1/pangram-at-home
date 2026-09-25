"""Build balanced, source-aware research splits from paper, EditLens and MAGE.

MAGE's published test set is divided into validation and test halves for this
run. Its results from the first model remain a frozen, pre-training audit.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SHARES = {
    "paper": .35,
    "reference_education": .20,
    "creative": .20,
    "social_qa": .10,
    "reviews": .10,
    "news": .05,
}
MAGE_DOMAIN = {
    "sci": "paper", "eli5": "reference_education", "squad": "reference_education",
    "roct": "creative", "wp": "creative", "cmv": "social_qa", "tldr": "social_qa",
    "yelp": "reviews", "xsum": "news",
}
EDITLENS_DOMAIN = {
    "fineweb_edu": "reference_education", "reddit_writing_prompts": "creative",
    "amazon_reviews": "reviews", "google_reviews": "reviews", "news": "news",
}
LENGTH_BINS = (40, 80, 160, 320, 640)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def length_bin(text: str) -> int | None:
    words = len(text.split())
    if words < 40 or words > 640:
        return None
    return next(i for i, upper in enumerate(LENGTH_BINS) if words <= upper)


def normalize(row: dict, *, family: str, domain: str, label: int, genre: str,
              generator: str, license_name: str) -> dict:
    return {"text_id": f"{family}:{row['text_id']}", "text": row["text"],
            "label": label, "source": f"{family}:{domain}", "source_family": family,
            "domain": genre, "source_id": str(row.get("source_id", row["text_id"])),
            "group_id": f"{family}:{row['group_id']}", "generator": generator,
            "license": license_name, "text_sha256": digest(" ".join(row["text"].casefold().split()))}


def mage_pairs(path: Path, split: str) -> dict[str, list[tuple[dict, dict]]]:
    buckets = collections.defaultdict(lambda: {0: [], 1: []})
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            source_rows = list(csv.DictReader(file))
    else:
        source_rows = pq.read_table(path).to_pylist()
    for index, row in enumerate(source_rows):
        source = row.get("mage_source", row.get("src", ""))
        domain = source.split("_", 1)[0]
        if domain not in MAGE_DOMAIN:
            continue
        bin_index = length_bin(row["text"])
        if bin_index is None:
            continue
        label = int(row["label"])
        if path.suffix == ".csv":
            label = 0 if label == 1 else 1  # MAGE original labels are reversed.
            half = "train"
        else:
            half = "val" if int(digest(f"mage-half:{domain}:{label}:{row['text_sha256']}")[:8], 16) % 2 else "test"
            if half != split:
                continue
        text_hash = digest(" ".join(row["text"].casefold().split()))
        normalized = normalize({"text_id": text_hash, "text": row["text"],
                                "group_id": text_hash, "source_id": text_hash},
                               family="mage", domain=domain, label=label,
                               genre=MAGE_DOMAIN[domain], generator=source if label else "human",
                               license_name="MAGE Apache-2.0 dataset; underlying source rights need review")
        buckets[(domain, bin_index)][label].append(normalized)
    pools = collections.defaultdict(list)
    for (domain, bin_index), classes in sorted(buckets.items()):
        for label in (0, 1):
            classes[label].sort(key=lambda r: digest(f"mage-{split}:{r['text_id']}"))
        for human, ai in zip(classes[0], classes[1]):
            pools[f"mage:{domain}"].append((human, ai))
    return pools


def paired_parquet(path: Path, family: str) -> dict[str, list[tuple[dict, dict]]]:
    groups = collections.defaultdict(lambda: {0: [], 1: []})
    for row in pq.read_table(path).to_pylist():
        source = row["source"]
        category = EDITLENS_DOMAIN[source] if family == "editlens" else "paper"
        if length_bin(row["text"]) is None:
            continue
        clean = normalize(row, family=family, domain=source, label=row["label"], genre=category,
                          generator=row.get("model", row.get("generator", "human")),
                          license_name=("CC BY-NC-SA 4.0 research" if family == "editlens" else row["license"]))
        groups[(source, clean["group_id"])][row["label"]].append(clean)
    pools = collections.defaultdict(list)
    for (source, group), classes in groups.items():
        if classes[0] and classes[1]:
            human_words = len(classes[0][0]["text"].split())
            ai_words = len(classes[1][0]["text"].split())
            if max(human_words, ai_words) > 2 * min(human_words, ai_words):
                continue
            pools[f"{family}:{source}"].append((classes[0][0], classes[1][0]))
    for source in pools:
        pools[source].sort(key=lambda pair: digest(f"{family}:{pair[0]['group_id']}"))
    return pools


def quota(total: int) -> dict[str, int]:
    counts = {category: int(total * share) for category, share in SHARES.items()}
    for category in SHARES:
        if sum(counts.values()) < total:
            counts[category] += 1
    assert sum(counts.values()) == total
    return counts


def choose(pools: dict[str, list[tuple[dict, dict]]], total: int) -> dict[str, list[tuple[dict, dict]]]:
    by_category = collections.defaultdict(list)
    for source, rows in pools.items():
        if rows:
            by_category[rows[0][0]["domain"]].append(source)
    selected = {}
    for category, count in quota(total).items():
        sources = sorted(by_category[category])
        positions = {source: 0 for source in sources}
        chosen = []
        seen = set()
        while len(chosen) < count:
            advanced = False
            for source in sources:
                while positions[source] < len(pools[source]):
                    pair = pools[source][positions[source]]
                    positions[source] += 1
                    if any(row["text_sha256"] in seen for row in pair):
                        continue
                    chosen.append(pair)
                    seen.update(row["text_sha256"] for row in pair)
                    advanced = True
                    break
                if len(chosen) == count:
                    break
            if not advanced:
                raise ValueError(f"Insufficient {category}: {len(chosen)} of {count}; sources {[(s,len(pools[s])) for s in sources]}")
        selected[category] = chosen
    return selected


def write(path: Path, chosen: dict[str, list[tuple[dict, dict]]], pairs: int) -> dict:
    rows = [row for category, count in quota(pairs).items() for pair in chosen[category][:count] for row in pair]
    rows.sort(key=lambda row: digest("diverse-v1:" + row["text_id"]))
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    by_source = collections.Counter(row["source"] for row in rows if row["label"] == 0)
    return {"path": path.name, "rows": len(rows), "human": pairs, "ai": pairs,
            "category_pairs": quota(pairs), "source_pairs": dict(sorted(by_source.items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    args = parser.parse_args()
    root = args.root / "data"
    output = root / "diverse_pyramid_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"label_map": {"human": 0, "AI": 1}, "category_targets": SHARES,
                "role": "local research; source licenses differ; do not redistribute raw text",
                "missing_categories": ["essays", "general_web", "professional_finance"],
                "splits": {}}
    mage = root / "mage_external_v1"
    frozen = mage / "frozen/main.parquet"
    for split, tiers in {"train": {"tiny": 100, "small": 500, "medium": 2000, "full": 5000},
                         "val": {"tiny": 100, "small": 250, "full": 400},
                         "test": {"tiny": 100, "small": 300, "full": 500}}.items():
        pools = mage_pairs(mage / "raw/train.csv" if split == "train" else frozen, split)
        for family, folder, name in [
            ("paper", "paper_pyramid_v1", f"{split}_full.parquet"),
            ("editlens", "editlens_pyramid_v1", "train_large.parquet" if split == "train" else f"{split}_full.parquet"),
        ]:
            for source, pairs in paired_parquet(root / folder / name, family).items():
                pools[source].extend(pairs)
        chosen = choose(pools, max(tiers.values()))
        manifest["splits"][split] = {tier: write(output / f"{split}_{tier}.parquet", chosen, size)
                                      for tier, size in tiers.items()}
        print(split, {k: x["rows"] for k, x in manifest["splits"][split].items()}, flush=True)
    # An exact text in two splits invalidates evaluation even if IDs differ.
    hashes = {}
    for split in ("train", "val", "test"):
        rows = pq.read_table(output / f"{split}_full.parquet", columns=["text_sha256"]).to_pydict()["text_sha256"]
        assert len(rows) == len(set(rows)), f"duplicates within {split}"
        hashes[split] = set(rows)
    assert not (hashes["train"] & hashes["val"] or hashes["train"] & hashes["test"] or hashes["val"] & hashes["test"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
