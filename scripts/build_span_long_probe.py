"""Make a long-window stitching diagnostic from the span pilot validation rows.

Components are reused from the calibration set, so this is a software/length
stress test, not an independent generalization benchmark.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def main():
    source = Path("/mnt/f/pangram-at-home/data/span_pilot_v3/val.jsonl")
    output = Path("/mnt/f/pangram-at-home/data/span_long_probe_v1")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["domain"], row["kind"])].append(row)
    rng = random.Random(144)
    documents = []
    schedule = ["human"] * 30 + ["ai"] * 30 + ["mixed"] * 60
    rng.shuffle(schedule)
    for index, kind in enumerate(schedule):
        domains = sorted({domain for domain, _ in list(groups)
                          if all(len(groups.get((domain, needed), [])) >= count
                                 for needed, count in (
                                     [(kind, 6)] if kind != "mixed" else
                                     [("mixed", 4), ("human", 1), ("ai", 1)]))})
        domain = rng.choice(domains)
        if kind == "mixed":
            components = (rng.sample(groups[(domain, "mixed")], 4) +
                          rng.sample(groups[(domain, "human")], 1) +
                          rng.sample(groups[(domain, "ai")], 1))
            rng.shuffle(components)
        else:
            components = rng.sample(groups[(domain, kind)], 6)
        text = ""
        spans = []
        for component in components:
            if text:
                text += rng.choice([" ", "\n", "\n\n"])
            shift = len(text)
            text += component["text"]
            spans.extend({"start": span["start"] + shift,
                          "end": span["end"] + shift,
                          "label": span["label"]} for span in component["spans"])
        documents.append({"id": f"long_{index:04d}", "text": text,
                          "spans": spans, "kind": kind,
                          "construction": "same-domain composite of span_pilot_v3 validation rows",
                          "source": "composite", "domain": domain,
                          "component_ids": [row["id"] for row in components],
                          "source_ids": [source_id for row in components
                                         for source_id in row["source_ids"]],
                          "source_groups": [group for row in components
                                            for group in row["source_groups"]]})
    output.mkdir(parents=True)
    target = output / "val.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents))
    manifest = {"role": "long-window stitching diagnostic, not independent evaluation",
                "parent": str(source),
                "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "documents": len(documents), "kinds": dict(Counter(schedule)), "seed": 144}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
