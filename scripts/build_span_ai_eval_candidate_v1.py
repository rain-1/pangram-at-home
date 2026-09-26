"""Freeze the Human Detectors article pairs for future external span evaluation.

The original includes annotator opinions and detector outputs; those fields are
discarded. Article copyright is retained by original publishers. Research-only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
import requests


URL = "https://raw.githubusercontent.com/jenna-russell/human_detectors/main/human_detectors.json"
ROOT = Path("/mnt/f/pangram-at-home/data")
OUTPUT = ROOT / "span_ai_eval_candidate_v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shingles(text: str, n: int = 24) -> set[bytes]:
    words = re.findall(r"\b\w+\b", text.casefold())
    return {hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words) - n + 1))}


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen candidate: {OUTPUT}")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    source = response.json()
    assert isinstance(source, dict) and len(source) == 300
    rows = list(source.values())
    by_link = defaultdict(list)
    for row in rows:
        assert row["ground_truth"] in {"Human-written", "AI-generated"}
        by_link[row["link"]].append(row)
    assert len(by_link) == 150
    assert all(Counter(r["ground_truth"] for r in pair) ==
               {"Human-written": 1, "AI-generated": 1} for pair in by_link.values())

    # Scan every existing diverse parent text, which also supplies the span-v3
    # and synthetic span-v4 source excerpts. Exact and long shared passages fail.
    parent_hashes, parent_shingles = set(), set()
    for split in ("train", "val", "test"):
        for text in pq.read_table(ROOT / "diverse_pyramid_v1" / f"{split}_full.parquet",
                                  columns=["text"]).column("text").to_pylist():
            parent_hashes.add(sha(text.encode()))
            parent_shingles.update(shingles(text))

    output_rows = []
    for link in sorted(by_link):
        for original in sorted(by_link[link], key=lambda row: row["ground_truth"]):
            text = original["article"].strip()
            assert len(text.split()) >= 100
            assert sha(text.encode()) not in parent_hashes
            assert not shingles(text) & parent_shingles
            label = 0 if original["ground_truth"] == "Human-written" else 1
            group = "human_detectors:article:" + sha(link.encode())[:24]
            output_rows.append({
                "id": f"human_detectors:{original['id']}", "text": text,
                "spans": [{"start": 0, "end": len(text), "label": label}],
                "kind": "human" if label == 0 else "ai",
                "construction": "published human article or study-recorded AI article",
                "source": "human_detectors", "domain": "nonfiction_article",
                "source_ids": [group], "source_groups": [group],
                "generator": "human" if label == 0 else original["generation_model"],
                "publication": original["source"], "publication_url": link,
                "human_author": original["author"], "prompt_id": original["prompt_id"],
                "text_sha256": sha(text.encode()),
            })
    assert len(output_rows) == 300
    assert Counter(row["kind"] for row in output_rows) == {"human": 150, "ai": 150}
    OUTPUT.mkdir(parents=True)
    path = OUTPUT / "test.jsonl"
    with path.open("w") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "role": "fresh research-only external evaluation candidate; not added to current run",
        "source": "Russell, Karpinska, Iyyer, Human Detectors (2025)",
        "source_url": URL, "source_sha256": sha(response.content),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_sha256": sha(path.read_bytes()), "rows": len(output_rows),
        "article_groups": len(by_link),
        "labels": dict(Counter(row["kind"] for row in output_rows)),
        "ai_generators": dict(Counter(row["generator"] for row in output_rows if row["kind"] == "ai")),
        "word_length": {"min": min(len(row["text"].split()) for row in output_rows),
                        "median": sorted(len(row["text"].split()) for row in output_rows)[150],
                        "max": max(len(row["text"].split()) for row in output_rows)},
        "overlap_check": "no exact text or normalized 24-word shingle overlap with diverse train/val/test parent",
        "rights": "Research-only local evaluation; original publisher article rights are not conferred by the repository's code license; no training or text redistribution.",
        "human_label_caveat": "Published articles have named authors and publisher links, but 2024-era author workflow is not independently verified as AI-free.",
        "task_caveat": "Whole-article human/AI labels; no authentic mixed-authorship boundaries. Paraphrased and humanized AI variants should be reported separately.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "article_groups", "labels", "ai_generators", "word_length")}, indent=2))


if __name__ == "__main__":
    main()
