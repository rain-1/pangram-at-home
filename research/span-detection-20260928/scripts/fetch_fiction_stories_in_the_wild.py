"""Acquire the authors' fixed 2020 controlled-writing release, outside Git."""
import argparse
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import requests

GIST = "edc87a274ae6dcedf43774a8d9e91bb4"
REVISION = "15a32423073310bfd521a851d10367321248254f"
PAPER = "https://aclanthology.org/2020.nuse-1.6/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "/mnt/f/pangram-at-home/data/fiction_candidates_v1/stories_in_the_wild"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    response = requests.get(f"https://api.github.com/gists/{GIST}/{REVISION}", timeout=30)
    response.raise_for_status()
    meta = response.json()
    date = next(x["committed_at"] for x in meta["history"] if x["version"] == REVISION)
    assert date.startswith("2020-")
    url = meta["files"]["stories.csv"]["raw_url"]
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw = response.content
    source_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    args.output.mkdir(parents=True)
    (args.output / "stories.csv").write_bytes(raw)
    (args.output / "source_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows, seen, exclusions = [], set(), Counter()
    for source in source_rows:
        text = source["story"].strip()
        if len(text.split()) < 80:
            exclusions["under_80_words"] += 1
            continue
        norm = sha(" ".join(re.findall(r"\w+", text.casefold())).encode())
        if norm in seen:
            exclusions["normalized_duplicate"] += 1
            continue
        seen.add(norm)
        sid = "stories_in_the_wild:" + source["story_id"]
        rows.append({
            "text": text, "source": "StoriesInTheWild controlled writing study",
            "source_id": sid, "work_id": sid, "author_ids": [],
            "author_id_status": "No stable author identifier supplied; story IDs do not establish author independence",
            "prompt_id": "stories_in_the_wild:" + source["data_type"],
            "writing_setup": source["writing_setup"],
            "publication_date": "2020-05-13", "collection_date": None,
            "source_revision": REVISION, "source_revision_date": date,
            "canonical_url": f"https://gist.github.com/talaugust/{GIST}/{REVISION}",
            "human_origin_status": "2020 volunteer writing study with recorded writing process, released by study authors; source-level human provenance",
            "strict_human_candidate": True,
            "rights_status": "Authors publicly released research data and request citation; no separate data license found in pinned release",
            "clean_sha256": sha(text.encode()), "source_sha256": sha(raw),
            "role": "human-fiction candidate; no training or evaluation split assigned",
        })
    target = args.output / "records.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "source": PAPER, "revision": REVISION, "revision_date": date,
        "raw_url": url, "raw_sha256": sha(raw), "raw_rows": len(source_rows),
        "paper_reported_rows": 1630, "retained_rows": len(rows), "exclusions": dict(exclusions),
        "prompt_counts": dict(Counter(row["prompt_id"] for row in rows)),
        "writing_setup_counts": dict(Counter(row["writing_setup"] for row in rows)),
        "words": sum(len(row["text"].split()) for row in rows),
        "records_sha256": sha(target.read_bytes()),
        "limitations": [
            "Pinned stories.csv has 1563 rows although paper describes 1630; discrepancy unresolved.",
            "Only five image prompts; hold out whole prompt groups or reserve whole corpus for evaluation.",
            "No stable author IDs; do not claim author-disjoint splits.",
            "Writing study is strong human provenance, not a forensic guarantee for every sentence.",
            "Original demographics remain only in raw research release; processed rows omit them.",
            "No AI counterparts; supports human FPR measurement, not balanced accuracy or recall alone.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
