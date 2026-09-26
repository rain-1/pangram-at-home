"""Build nested 5k/10k/20k span training tiers from two audited sources.

One quarter of each tier comes from the existing v4 synthetic training pool;
three quarters comes from source-split LLMTrace English documents with at least
80 words. Holdouts remain separate. This is a data-size study, not a new
synthetic-join generator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


ROOT = Path("/mnt/f/pangram-at-home/data")
SIZES = (5000, 10000, 20000)
SEED = 20260926


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


def write_rows(path: Path, values: list[dict]) -> None:
    with path.open("w") as file:
        for row in values:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_text(values: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in values:
        key = row.get("text_sha256") or hashlib.sha256(row["text"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def excluded_groups(source_rows: list[dict], audit: dict, source: str, refs: tuple[str, ...]) -> set[str]:
    by_id = {row["id"]: row["group_id"] for row in source_rows}
    hits = audit["sources"][source]["first_match_ids"]
    groups = {by_id[item["id"]] for ref in refs for item in hits.get(ref, [])}
    for ref in refs:
        found = audit["sources"][source]["matches"].get(ref, {}).get("rows_with_any_match", 0)
        if found > len(hits.get(ref, [])):
            raise ValueError(f"Audit truncated {source}/{ref}; cannot exclude all groups")
    return groups


def interleave_strata(values: list[dict], size: int, seed: int) -> list[dict]:
    """Sample proportional domain/kind strata, then order them evenly."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in values:
        buckets[(row["domain"], row["kind"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    total = len(values)
    raw = {key: size * len(bucket) / total for key, bucket in buckets.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = size - sum(quotas.values())
    for key in sorted(raw, key=lambda key: (raw[key] - quotas[key], key), reverse=True)[:remainder]:
        quotas[key] += 1
    assert all(quotas[key] <= len(bucket) for key, bucket in buckets.items())
    selected = {key: buckets[key][:quotas[key]] for key in buckets}
    emitted = Counter()
    ordered = []
    while len(ordered) < size:
        index = len(ordered) + 1
        candidates = [key for key in selected if emitted[key] < len(selected[key])]
        key = max(candidates, key=lambda k: (index * len(selected[k]) / size - emitted[k],
                                             len(selected[k]), k))
        ordered.append(selected[key][emitted[key]])
        emitted[key] += 1
    return ordered


def describe(values: list[dict]) -> dict:
    kinds = Counter(row["kind"] for row in values)
    sources = Counter("LLMTrace" if row["source"] == "LLMTrace_detection" else "span_v4_parent"
                      for row in values)
    domains = Counter(("llmtrace:" if row["source"] == "LLMTrace_detection" else "v4:") + row["domain"]
                      for row in values)
    chars = Counter()
    for row in values:
        for span in row["spans"]:
            if span["label"] in (0, 1):
                chars[str(span["label"])] += span["end"] - span["start"]
    return {"documents": len(values), "kinds": dict(kinds), "sources": dict(sources),
            "domains": dict(domains), "labeled_characters": dict(chars),
            "ai_character_fraction": chars["1"] / (chars["0"] + chars["1"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-folder", default="span_size_curve_v5")
    args = parser.parse_args()
    root = args.root
    output = root / args.output_folder
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    old_train_path = root / "span_training_v4/train.jsonl"
    old_val_path = root / "span_training_v4/val.jsonl"
    llm_root = root / "span_sources_v5/normalized_llmtrace_en"
    llm_train_path, llm_val_path, llm_test_path = [llm_root / f"{name}.jsonl" for name in ("train", "val", "test")]
    audit_path = root / "span_sources_v5/overlap_audit.json"
    audit = json.loads(audit_path.read_text())
    old_train, old_val = rows(old_train_path), rows(old_val_path)
    # Some v4 composites are exact duplicates because their source excerpts
    # were reused. Retain only the first identical text before counting sizes.
    old_train = unique_text(old_train)
    old_val = unique_text(old_val)
    source_train, source_val, source_test = rows(llm_train_path), rows(llm_val_path), rows(llm_test_path)
    for key, path in (("llmtrace_train", llm_train_path), ("llmtrace_val", llm_val_path),
                      ("llmtrace_test", llm_test_path)):
        if audit["sources"][key]["file_sha256"] != digest(path):
            raise ValueError(f"Stale overlap audit for {key}")
    excluded = {
        "train": excluded_groups(source_train, audit, "llmtrace_train", ("old_span_val", "diverse_frozen_test")),
        "val": excluded_groups(source_val, audit, "llmtrace_val", ("old_span_train", "old_span_val", "diverse_frozen_test")),
        "test": excluded_groups(source_test, audit, "llmtrace_test", ("old_span_train", "old_span_val", "diverse_frozen_test")),
    }
    def eligible(values, split):
        return [row for row in values if row["group_id"] not in excluded[split]
                and len(row["text"].split()) >= 80]
    llm_train = eligible(source_train, "train")
    llm_val = eligible(source_val, "val")
    llm_test = eligible(source_test, "test")
    old_full = len(old_train)
    llm_full = 20000 - old_full
    if len(llm_train) < llm_full or old_full < 4900 or len(llm_val) < 600 or len(llm_test) < 2000:
        raise ValueError("Insufficient eligible documents for size curve")
    old_order = interleave_strata(old_train, old_full, SEED + 1)
    # The upstream English pool is human-character heavy. Select nearly all
    # eligible AI and mixed documents so the final character labels approach
    # balance while keeping long, substantial examples only.
    kind_targets = {"human": 3650 + (llm_full - 15000), "ai": 4500, "mixed": 6850}
    llm_selected = []
    for index, (kind, target) in enumerate(kind_targets.items()):
        candidates = [row for row in llm_train if row["kind"] == kind]
        if len(candidates) < target:
            raise ValueError(f"Need {target} {kind} LLMTrace documents; have {len(candidates)}")
        llm_selected.extend(interleave_strata(candidates, target, SEED + 20 + index))
    llm_order = interleave_strata(llm_selected, llm_full, SEED + 2)
    old_val_order = interleave_strata(old_val, 200, SEED + 3)
    llm_val_order = interleave_strata(llm_val, 600, SEED + 4)
    llm_test_order = interleave_strata(llm_test, 2000, SEED + 5)
    output.mkdir(parents=True)
    manifest = {"role": "nested 5k/10k/20k data-size study; no assisted class",
                "seed": SEED, "source_rule": "approximately 25% unique prior v4 composites; remainder LLMTrace English documents >=80 words",
                "excluded_duplicate_v4_train_documents": 5000 - old_full,
                "llmtrace_kind_targets_20k_tier": kind_targets,
                "excluded_llmtrace_topic_groups": {key: sorted(value) for key, value in excluded.items()},
                "parents_sha256": {str(path): digest(path) for path in (old_train_path, old_val_path,
                                  llm_train_path, llm_val_path, llm_test_path, audit_path)},
                "tiers": {}}
    evaluation = old_val_order + llm_val_order
    test = llm_test_order
    for size in SIZES:
        folder = output / f"size_{size}"
        folder.mkdir()
        old_count = min(size // 4, old_full)
        training = old_order[:old_count] + llm_order[:size - old_count]
        random.Random(SEED + size).shuffle(training)
        write_rows(folder / "train.jsonl", training)
        write_rows(folder / "val.jsonl", evaluation)
        write_rows(folder / "test_llmtrace.jsonl", test)
        entry = {"train": describe(training), "val": describe(evaluation),
                 "llmtrace_test": describe(test),
                 "files_sha256": {name: digest(folder / name) for name in
                                  ("train.jsonl", "val.jsonl", "test_llmtrace.jsonl")}}
        manifest["tiers"][str(size)] = entry
        (folder / "manifest.json").write_text(json.dumps({**manifest, "active_tier": size,
                                                           "tier": entry}, indent=2) + "\n")
        print(size, entry["train"]["sources"], entry["train"]["kinds"],
              round(entry["train"]["ai_character_fraction"], 3), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for left, right in ((5000, 10000), (10000, 20000)):
        a = {row["id"] for row in rows(output / f"size_{left}/train.jsonl")}
        b = {row["id"] for row in rows(output / f"size_{right}/train.jsonl")}
        if not a <= b:
            raise AssertionError(f"Tier {left} is not nested in {right}")
    print(output)


if __name__ == "__main__":
    main()
