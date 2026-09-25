"""Freeze diverse MAGE holdouts for out-of-source evaluation of the first model."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"
FILES = {"main": "test.csv", "gpt4_ood": "test_ood_set_gpt.csv", "paraphrase": "test_ood_set_gpt_para.csv"}
PER_CLASS = {"main": 200, "gpt4_ood": 150, "paraphrase": 150}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> str:
    return digest("mage-eval-v1:" + row["mage_source"] + ":" + row["text_sha256"])


def select_ai(rows: list[dict], count: int) -> list[dict]:
    # Equal opportunity per generator/prompt subtype, avoiding one large subtype.
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["mage_source"]].append(row)
    for group in groups.values():
        group.sort(key=key)
    result = []
    while len(result) < count:
        advanced = False
        for source in sorted(groups):
            if groups[source] and len(result) < count:
                result.append(groups[source].pop(0))
                advanced = True
        if not advanced:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    root = args.root / "data/mage_external_v1"
    output = root / "frozen"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "yaful/MAGE", "revision": REVISION,
                "label_map": {"0": "human", "1": "AI"},
                "role": "external evaluation only; excluded from first Qwen3 training and threshold selection",
                "splits": {}}
    previous_hashes = set()
    for name, filename in FILES.items():
        path = root / "raw" / filename
        groups = collections.defaultdict(lambda: {0: [], 1: []})
        with path.open(newline="", encoding="utf-8-sig") as file:
            for source_row in csv.DictReader(file):
                text = source_row["text"].strip()
                if len(text.split()) < 20:
                    continue
                label = 0 if int(source_row["label"]) == 1 else 1
                source = source_row["src"]
                domain = source.split("_", 1)[0]
                hashed = digest(" ".join(text.casefold().split()))
                groups[domain][label].append({
                    "text_id": "mage:" + name + ":" + hashed,
                    "text": text, "label": label, "source": domain,
                    "source_id": hashed, "group_id": hashed,
                    "mage_source": source, "text_sha256": hashed,
                })
        chosen = []
        for domain in sorted(groups):
            human = sorted(groups[domain][0], key=key)[:PER_CLASS[name]]
            ai = select_ai(groups[domain][1], PER_CLASS[name])
            pairs = min(len(human), len(ai))
            assert pairs >= 100, (name, domain, len(human), len(ai))
            chosen.extend(human[:pairs] + ai[:pairs])
        # A repeated text in a published test set should appear only once.
        deduped = {}
        for row in sorted(chosen, key=key):
            deduped.setdefault(row["text_sha256"], row)
        assert len(deduped) == len(chosen), (name, "duplicate selected text")
        if name == "main":
            assert not previous_hashes & set(deduped)
        previous_hashes.update(deduped)
        chosen.sort(key=key)
        out = output / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(chosen), out, compression="zstd")
        manifest["splits"][name] = {
            "raw_file": filename, "raw_sha256": file_hash(path), "path": out.name,
            "sha256": file_hash(out), "rows": len(chosen),
            "human": sum(row["label"] == 0 for row in chosen),
            "ai": sum(row["label"] == 1 for row in chosen),
            "by_domain_label": {f"{domain}:{label}": sum(row["source"] == domain and row["label"] == label for row in chosen)
                                for domain in sorted(groups) for label in (0, 1)},
            "ai_source_count": len({row["mage_source"] for row in chosen if row["label"] == 1}),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {k: info[k] for k in ("rows", "human", "ai", "ai_source_count")}
                      for name, info in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
