"""Build traceable binary span data from frozen diverse train/val parents.

Long examples are synthetic joins of source excerpts, never represented as
continuous source documents. Every labeled segment maps to exact source bytes.
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

from span_data import encode_document, window_starts


DOMAIN_WEIGHTS = {"paper": 25, "creative": 20, "reference_education": 20,
                  "reviews": 15, "social_qa": 15, "news": 5}
KINDS = {"unaltered_source": 20, "same_label_short_join": 10,
         "same_label_long_join": 20, "matched_pair_short_mix": 25,
         "same_source_long_mix": 25}
SEPARATORS = (" ", "\n", "\n\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quota_schedule(weights: dict[str, int], count: int, rng: random.Random) -> list[str]:
    raw = {name: count * value / sum(weights.values()) for name, value in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    for name in sorted(weights, key=lambda name: (raw[name] - counts[name], name), reverse=True)[:count - sum(counts.values())]:
        counts[name] += 1
    schedule = [name for name, size in counts.items() for _ in range(size)]
    rng.shuffle(schedule)
    return schedule


def exact_excerpt(row: dict, target_words: int, rng: random.Random, *, full: bool = False) -> tuple[str, int, int]:
    """Return a contiguous, unmodified substring and its source offsets."""
    text = row["text"]
    words = list(re.finditer(r"\S+", text))
    if not words:
        raise ValueError(f"Empty source row {row['text_id']}")
    if full or len(words) <= target_words:
        start, end = words[0].start(), words[-1].end()
    else:
        first = rng.randrange(len(words) - target_words + 1)
        start, end = words[first].start(), words[first + target_words - 1].end()
    return text[start:end], start, end


class SourcePool:
    def __init__(self, rows: list[dict], seed: int, max_reuse: int):
        self.rng = random.Random(seed)
        self.max_reuse = max_reuse
        self.uses: Counter[str] = Counter()
        self.by_source: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        grouped: dict[str, dict[int, dict]] = defaultdict(dict)
        for row in rows:
            self.by_source[row["source"]][int(row["label"])].append(row)
            grouped[row["group_id"]][int(row["label"])] = row
        self.pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for group in grouped.values():
            if set(group) == {0, 1} and group[0]["source"] == group[1]["source"]:
                self.pairs[group[0]["source"]].append((group[0], group[1]))
        for classes in self.by_source.values():
            for entries in classes.values():
                self.rng.shuffle(entries)
        for entries in self.pairs.values():
            self.rng.shuffle(entries)
        self.domain_sources: dict[str, list[str]] = defaultdict(list)
        for source, classes in self.by_source.items():
            self.domain_sources[classes[0][0]["domain"] if classes[0] else classes[1][0]["domain"]].append(source)

    def source(self, domain: str, labels: list[int], *, paired: bool = False) -> str:
        candidates = []
        for source in self.domain_sources[domain]:
            if paired:
                available = any(all(self.uses[row["text_id"]] < self.max_reuse for row in pair)
                                for pair in self.pairs[source])
            else:
                available = all(sum(self.uses[row["text_id"]] < self.max_reuse
                                    for row in self.by_source[source][label]) >= labels.count(label)
                                for label in set(labels))
            if available:
                # Normalize load by source population to prevent a small source
                # from being reused disproportionately.
                entries = self.by_source[source][0] + self.by_source[source][1]
                load = sum(self.uses[row["text_id"]] for row in entries) / len(entries)
                candidates.append((load, self.rng.random(), source))
        if not candidates:
            raise ValueError(f"Exhausted {domain} for {labels}; max reuse={self.max_reuse}")
        return min(candidates)[2]

    def pick(self, source: str, label: int, exclude: set[str]) -> dict:
        candidates = [row for row in self.by_source[source][label]
                      if row["text_id"] not in exclude and self.uses[row["text_id"]] < self.max_reuse]
        if not candidates:
            raise ValueError(f"Exhausted {source} class {label}")
        least = min(self.uses[row["text_id"]] for row in candidates)
        row = self.rng.choice([row for row in candidates if self.uses[row["text_id"]] == least])
        self.uses[row["text_id"]] += 1
        return row

    def pair(self, source: str) -> tuple[dict, dict]:
        candidates = [pair for pair in self.pairs[source]
                      if all(self.uses[row["text_id"]] < self.max_reuse for row in pair)]
        if not candidates:
            raise ValueError(f"Exhausted paired source {source}")
        least = min(sum(self.uses[row["text_id"]] for row in pair) for pair in candidates)
        pair = self.rng.choice([pair for pair in candidates
                                if sum(self.uses[row["text_id"]] for row in pair) == least])
        for row in pair:
            self.uses[row["text_id"]] += 1
        return pair


def make_document(pool: SourcePool, split: str, index: int, domain: str, construction: str) -> dict:
    rng = pool.rng
    first = index % 2
    if construction == "matched_pair_short_mix" and not any(pool.pairs[source] for source in pool.domain_sources[domain]):
        construction = "same_source_short_mix"
    if construction == "unaltered_source":
        labels = [first]
    elif construction == "same_label_short_join":
        labels = [first] * rng.choice((2, 3))
    elif construction == "same_label_long_join":
        labels = [first] * rng.choice((5, 6, 7))
    elif construction in ("matched_pair_short_mix", "same_source_short_mix"):
        labels = [first, 1 - first]
    elif construction == "same_source_long_mix":
        length = rng.choice((5, 6, 7))
        switch = rng.randrange(1, length - 1)
        labels = [first] * switch + [1 - first] * (length - switch)
        if rng.random() < .5 and length >= 6:
            labels[-1] = first
    else:
        raise ValueError(construction)
    source = pool.source(domain, labels, paired=construction == "matched_pair_short_mix")
    if construction == "matched_pair_short_mix":
        pair = pool.pair(source)
        origins = [pair[label] for label in labels]
    else:
        seen: set[str] = set()
        origins = []
        for label in labels:
            row = pool.pick(source, label, seen)
            origins.append(row)
            seen.add(row["text_id"])
    text = ""
    spans = []
    components = []
    for row, label in zip(origins, labels):
        target = (rng.choice((50, 90, 140)) if "short" in construction else
                  rng.choice((90, 140, 200)))
        piece, source_start, source_end = exact_excerpt(row, target, rng,
                                                        full=construction == "unaltered_source")
        if text:
            text += rng.choice(SEPARATORS)
        start = len(text)
        text += piece
        end = len(text)
        spans.append({"start": start, "end": end, "label": label})
        components.append({"text_id": row["text_id"], "source_id": row["source_id"],
                           "group_id": row["group_id"], "source": row["source"],
                           "source_family": row["source_family"], "generator": row["generator"],
                           "license": row["license"], "parent_text_sha256": row["text_sha256"],
                           "source_start": source_start, "source_end": source_end,
                           "output_start": start, "output_end": end, "label": label})
    assert all(text[c["output_start"]:c["output_end"]] ==
               row["text"][c["source_start"]:c["source_end"]]
               for c, row in zip(components, origins))
    kind = "mixed" if len(set(labels)) == 2 else ("human" if labels[0] == 0 else "ai")
    return {"id": f"{split}_{index:06d}", "text": text, "spans": spans, "kind": kind,
            "construction": construction, "source": source, "domain": domain,
            "source_ids": [row["text_id"] for row in origins],
            "source_groups": [row["group_id"] for row in origins],
            "components": components, "text_sha256": digest(text.encode())}


def summarize(rows: list[dict], tokenizer) -> dict:
    token_counts: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    lengths = []
    windows = 0
    for row in rows:
        ids, offsets, labels = encode_document(row, tokenizer)
        lengths.append(len(ids))
        windows += len(window_starts(len(ids)))
        token_counts.update(str(label) for label in labels if label in (0, 1))
        for (start, end), label in zip(offsets, labels):
            if label not in (0, 1):
                continue
            part = next((c for c in row["components"]
                         if c["output_start"] <= start and end <= c["output_end"]), None)
            if part:
                source_tokens[part["source"]] += 1
                domain_tokens[row["domain"]] += 1
    lengths.sort()
    all_ids = [c["text_id"] for row in rows for c in row["components"]]
    uses = Counter(all_ids)
    return {"documents": len(rows), "windows_512_stride_256": windows,
            "token_lengths": {name: lengths[min(len(lengths)-1, int(q*(len(lengths)-1)))]
                              for name, q in (("min", 0), ("p25", .25), ("median", .5),
                                              ("p75", .75), ("p90", .9), ("max", 1))},
            "documents_over_512_tokens": sum(n > 512 for n in lengths),
            "documents_over_1024_tokens": sum(n > 1024 for n in lengths),
            "labeled_tokens": dict(token_counts), "source_labeled_tokens": dict(source_tokens),
            "domain_labeled_tokens": dict(domain_tokens),
            "kinds": dict(Counter(row["kind"] for row in rows)),
            "constructions": dict(Counter(row["construction"] for row in rows)),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "source_documents": len(uses), "source_uses": len(all_ids),
            "max_source_reuse": max(uses.values()),
            "source_reuse_histogram": dict(Counter(uses.values()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--output-folder", default="span_training_v4")
    parser.add_argument("--train-docs", type=int, default=5000)
    parser.add_argument("--val-docs", type=int, default=600)
    parser.add_argument("--train-max-reuse", type=int, default=5)
    parser.add_argument("--val-max-reuse", type=int, default=6)
    args = parser.parse_args()
    output = args.root / "data" / args.output_folder
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    parent = args.root / "data/diverse_pyramid_v1"
    source_rows = {split: pq.read_table(parent / f"{split}_full.parquet").to_pylist()
                   for split in ("train", "val")}
    for key in ("text_sha256", "group_id", "source_id"):
        assert not ({row[key] for row in source_rows["train"]} &
                    {row[key] for row in source_rows["val"]}), key
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.root / "models/Qwen3-1.7B")
    output.mkdir(parents=True)
    manifest = {"label_map": {"human": 0, "ai_generated": 1},
                "ai_assisted_supported": False, "role": "synthetic span-training development",
                "description": "Original substrings joined into synthetic documents; joins do not establish realistic mixed authorship",
                "test_data_used": False, "domain_targets_percent": DOMAIN_WEIGHTS,
                "construction_targets_percent": KINDS, "splits": {}}
    for split, count, seed, max_reuse in (("train", args.train_docs, 431, args.train_max_reuse),
                                           ("val", args.val_docs, 1431, args.val_max_reuse)):
        rng = random.Random(seed)
        domains = quota_schedule(DOMAIN_WEIGHTS, count, rng)
        constructions = quota_schedule(KINDS, count, rng)
        pool = SourcePool(source_rows[split], seed, max_reuse)
        rows = [make_document(pool, split, i, domain, construction)
                for i, (domain, construction) in enumerate(zip(domains, constructions))]
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        summary = summarize(rows, tokenizer)
        summary.update({"seed": seed, "max_reuse_limit": max_reuse,
                        "parent_file": str(parent / f"{split}_full.parquet"),
                        "parent_sha256": digest((parent / f"{split}_full.parquet").read_bytes()),
                        "sha256": digest(path.read_bytes())})
        manifest["splits"][split] = summary
        print(split, summary["documents"], "windows", summary["windows_512_stride_256"],
              "long", summary["documents_over_512_tokens"], flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
