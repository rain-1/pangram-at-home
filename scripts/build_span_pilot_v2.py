"""Build known-origin binary span examples from frozen, disjoint source splits.

Most mixed documents use human/AI rows paired by the same source group. Some
same-source joins are retained as an explicit, less realistic stress stratum.
No token in this dataset is labeled AI-assisted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
            if part.strip()]


def excerpt(text: str, rng: random.Random, target_words: int) -> str:
    parts = sentences(text)
    start = rng.randrange(len(parts))
    chosen = []
    for part in parts[start:]:
        chosen.append(part)
        if len(" ".join(chosen).split()) >= target_words:
            break
    candidate = " ".join(chosen).strip()
    if len(candidate.split()) < 20:
        words = text.split()
        start = rng.randrange(max(1, len(words) - min(target_words, len(words)) + 1))
        candidate = " ".join(words[start:start + target_words])
    return candidate


def two_excerpts(text: str, rng: random.Random, target_words: int) -> tuple[str, str]:
    parts = sentences(text)
    if len(parts) >= 4:
        middle = len(parts) // 2
        return excerpt(" ".join(parts[:middle]), rng, target_words), excerpt(
            " ".join(parts[middle:]), rng, target_words)
    words = text.split()
    middle = len(words) // 2
    return (" ".join(words[:min(middle, target_words)]),
            " ".join(words[middle:middle + target_words]))


def pools(rows: list[dict]):
    by_source = defaultdict(lambda: {0: [], 1: []})
    by_group = defaultdict(list)
    for row in rows:
        by_source[row["source"]][int(row["label"])].append(row)
        by_group[row["group_id"]].append(row)
    paired = defaultdict(list)
    for group in by_group.values():
        if len(group) == 2 and {int(row["label"]) for row in group} == {0, 1}:
            paired[group[0]["source"]].append({int(row["label"]): row for row in group})
    return by_source, paired


def document(split: str, index: int, kind: str, by_source, paired, rng: random.Random,
             separator_policy: str):
    if kind == "paired_mixed":
        source = rng.choice(sorted(paired))
        pair = rng.choice(paired[source])
        first = rng.randrange(2)
        labels = [first, 1 - first]
        if rng.random() < .45:
            labels.append(first)
        target = rng.choice([30, 55, 90])
        pieces = {}
        for label in {0, 1}:
            if labels.count(label) == 2:
                pieces[label] = list(two_excerpts(pair[label]["text"], rng, target))
            else:
                pieces[label] = [excerpt(pair[label]["text"], rng, target)]
        selected = [(pair[label], pieces[label].pop(0), label) for label in labels]
        construction = "matched_source_pair"
    elif kind == "same_source_mixed":
        source = rng.choice(sorted(s for s, classes in by_source.items()
                                   if classes[0] and classes[1]))
        first = rng.randrange(2)
        labels = [first, 1 - first]
        selected = [(rng.choice(by_source[source][label]), None, label) for label in labels]
        selected = [(row, excerpt(row["text"], rng, rng.choice([30, 55, 90])), label)
                    for row, _, label in selected]
        construction = "unmatched_same_source_join"
    else:
        label = 0 if kind == "human" else 1
        source = rng.choice(sorted(s for s, classes in by_source.items() if classes[label]))
        row = rng.choice(by_source[source][label])
        selected = [(row, excerpt(row["text"], rng, rng.choice([120, 240, 400])), label)]
        construction = "unaltered_source_excerpt"
    text = ""
    spans = []
    for row, piece, label in selected:
        if separator_policy == "varied" and rng.random() < .35:
            parts = sentences(piece)
            if len(parts) >= 2:
                middle = len(parts) // 2
                piece = " ".join(parts[:middle]) + "\n\n" + " ".join(parts[middle:])
        if text:
            text += (rng.choice([" ", "\n", "\n\n"])
                     if separator_policy == "varied" else "\n\n")
        start = len(text)
        text += piece
        spans.append({"start": start, "end": len(text), "label": label})
    assert all(span["end"] > span["start"] for span in spans)
    return {"id": f"{split}_{index:05d}", "text": text, "spans": spans,
            "kind": "mixed" if kind.endswith("mixed") else kind,
            "construction": construction, "source": source,
            "domain": selected[0][0]["domain"],
            "source_ids": [row["text_id"] for row, _, _ in selected],
            "source_groups": [row["group_id"] for row, _, _ in selected]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--train-docs", type=int, default=2400)
    parser.add_argument("--val-docs", type=int, default=400)
    parser.add_argument("--output-folder", default="span_pilot_v2")
    parser.add_argument("--separator-policy", choices=["paragraph_only", "varied"],
                        default="paragraph_only")
    args = parser.parse_args()
    parent = args.root / "data/diverse_pyramid_v1"
    output = args.root / "data" / args.output_folder
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    inputs = {split: pq.read_table(parent / f"{split}_full.parquet").to_pylist()
              for split in ("train", "val")}
    for key in ("text_sha256", "group_id"):
        assert not ({row[key] for row in inputs["train"]}
                    & {row[key] for row in inputs["val"]}), key
    output.mkdir(parents=True)
    manifest = {"labels": {"human": 0, "ai_generated": 1},
                "ai_assisted_supported": False,
                "role": "synthetic development pilot; matched pairs improve topic control but joins remain artificial",
                "test_data_used": False, "separator_policy":args.separator_policy,"splits": {}}
    for split, count, seed in (("train", args.train_docs, 43), ("val", args.val_docs, 143)):
        rng = random.Random(seed)
        by_source, paired = pools(inputs[split])
        assert paired
        schedule = (["human", "ai", "paired_mixed", "paired_mixed", "paired_mixed",
                     "same_source_mixed"] * ((count + 5) // 6))[:count]
        rng.shuffle(schedule)
        rows = [document(split, index, kind, by_source, paired, rng, args.separator_policy)
                for index, kind in enumerate(schedule)]
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {
            "documents": len(rows), "kinds": dict(Counter(row["kind"] for row in rows)),
            "constructions": dict(Counter(row["construction"] for row in rows)),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "sha256": sha(path), "parent_sha256": sha(parent / f"{split}_full.parquet"),
            "seed": seed}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
