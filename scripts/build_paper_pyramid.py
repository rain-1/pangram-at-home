"""Build a balanced, source-diverse PMC + ACL paper abstract pyramid."""

from __future__ import annotations

import argparse
import collections
import difflib
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

GENERATOR_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value: str, title: str = "") -> str:
    value = value.strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if title and len(lines) > 1:
        first = re.sub(r"(?i)^\*{0,2}(?:title|abstract)\*{0,2}\s*[:\-]\s*", "", lines[0]).strip(" *\"")
        if difflib.SequenceMatcher(None, first.casefold(), title.casefold()).ratio() > 0.72:
            lines = lines[1:]
    value = " ".join(lines)
    value = re.sub(r"(?i)\*\*\s*(abstract|background|methods?|results?|conclusions?)\s*\*\*\s*[:\-]?", "", value)
    value = re.sub(r"(?i)\b(abstract|background|methods?|results?|conclusions?)\s*:\s*", "", value)
    return " ".join(value.split()).strip(" -")


def load_jsonl(path: Path, compressed: bool = False) -> list[dict]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def paper_pairs(root: Path, source: str) -> tuple[list[tuple[dict, dict]], collections.Counter]:
    data_dir = root / "data" / ("pmc_pilot_v1" if source == "pmc" else "acl_abstracts_v1")
    generator_file = "generated_qwen.jsonl" if source == "pmc" else "generated_smollm.jsonl"
    documents = {d["source_id"]: d for d in load_jsonl(data_dir / "documents.jsonl.gz", True)}
    generations = load_jsonl(data_dir / generator_file)
    rejected = collections.Counter()
    pairs = []
    seen = set()
    for g in generations:
        doc = documents.get(g["source_id"])
        if doc is None or g["source_id"] in seen:
            rejected["missing_or_duplicate"] += 1
            continue
        seen.add(g["source_id"])
        year = int(doc["publication_date"][:4]) if source == "pmc" else doc["year"]
        if year > 2022 or doc["license"] != "CC BY 4.0":
            rejected["date_or_license"] += 1
            continue
        if g["human_abstract_sha256"] != doc["abstract_sha256"]:
            rejected["human_hash"] += 1
            continue
        human = normalize(doc["abstract"])
        ai = normalize(g["text"], doc["title"])
        hw, aw = len(human.split()), len(ai.split())
        if not (120 <= hw <= 280 and aw >= 80 and 0.6 <= aw / hw <= 1.6):
            rejected["length"] += 1
            continue
        if digest(human.casefold()) == digest(ai.casefold()):
            rejected["identical"] += 1
            continue
        venue = doc["journal"] if source == "pmc" else doc["venue"]
        base = {
            "source": "pmc_oa" if source == "pmc" else "acl_anthology",
            "source_id": doc["source_id"],
            "group_id": doc["source_id"],
            "venue": venue,
            "year": year,
            "canonical_url": doc["canonical_url"],
            "title": doc["title"],
            "license": doc["license"],
            "license_url": doc.get("license_url", doc.get("license_evidence_url")),
        }
        pairs.append((
            {**base, "text_id": doc["source_id"] + ":human", "text": human,
             "text_sha256": digest(human.casefold()), "label": 0, "generator": "human",
             "generator_revision": "", "prompt_version": ""},
            {**base, "text_id": doc["source_id"] + ":ai", "text": ai,
             "text_sha256": digest(ai.casefold()), "label": 1, "generator": g["model"],
             "generator_revision": GENERATOR_REVISIONS[g["model"]],
             "prompt_version": g["prompt_version"]},
        ))
    return pairs, rejected


def write(path: Path, pairs: list[tuple[dict, dict]]) -> dict:
    rows = [row for pair in pairs for row in pair]
    rows.sort(key=lambda row: digest("paper-pyramid-output-v1:" + row["text_id"]))
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {
        "path": path.name,
        "pairs": len(pairs),
        "rows": len(rows),
        "human": len(pairs),
        "ai": len(pairs),
        "by_source": dict(collections.Counter(pair[0]["source"] for pair in pairs)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--pmc-journal-cap", type=int, default=20)
    args = parser.parse_args()
    all_pairs = []
    rejections = {}
    for source in ["pmc", "acl"]:
        pairs, rejected = paper_pairs(args.root, source)
        rejections[source] = dict(rejected)
        if source == "pmc":
            pairs.sort(key=lambda pair: digest("pmc-journal-cap-v1:" + pair[0]["source_id"]))
            counts = collections.Counter()
            capped = []
            for pair in pairs:
                venue = pair[0]["venue"]
                if counts[venue] < args.pmc_journal_cap:
                    capped.append(pair)
                    counts[venue] += 1
            rejections[source]["journal_cap"] = len(pairs) - len(capped)
            pairs = capped
        all_pairs.extend(pairs)
    # Keep venues disjoint between train and test within each source.
    by_source_venue = collections.defaultdict(list)
    for pair in all_pairs:
        by_source_venue[(pair[0]["source"], pair[0]["venue"])].append(pair)
    test_groups = set()
    for source in ["pmc_oa", "acl_anthology"]:
        source_groups = [key for key in by_source_venue if key[0] == source]
        source_groups.sort(key=lambda key: digest("paper-test-venue-v1:" + ":".join(key)))
        source_count = sum(len(by_source_venue[key]) for key in source_groups)
        target = round(source_count * 0.18)
        selected = 0
        for key in source_groups:
            if selected >= target:
                break
            test_groups.add(key)
            selected += len(by_source_venue[key])
    test = [pair for pair in all_pairs if (pair[0]["source"], pair[0]["venue"]) in test_groups]
    remaining = [pair for pair in all_pairs if (pair[0]["source"], pair[0]["venue"]) not in test_groups]
    val, train = [], []
    for source in ["pmc_oa", "acl_anthology"]:
        source_pairs = [pair for pair in remaining if pair[0]["source"] == source]
        source_pairs.sort(key=lambda pair: digest("paper-val-v1:" + pair[0]["source_id"]))
        val_count = round(sum(pair[0]["source"] == source for pair in all_pairs) * 0.18)
        val.extend(source_pairs[:val_count])
        train.extend(source_pairs[val_count:])
    output = args.root / "data" / "paper_pyramid_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "official PMC OA JATS + ACL Anthology XML, locally generated AI",
        "human_cutoff": "2022-12-31",
        "rights": "CC BY 4.0 verified per PMC article / ACL 2016+ publisher policy",
        "generator_revisions": GENERATOR_REVISIONS,
        "label_map": {"human": 0, "ai": 1},
        "rejections": rejections,
        "test_venues": [list(key) for key in sorted(test_groups)],
        "splits": {},
    }
    for name, content in [("train", train), ("val", val), ("test", test)]:
        content.sort(key=lambda pair: digest("paper-tier-v1:" + pair[0]["source_id"]))
        sizes = {"tiny": min(50, len(content))}
        if name == "train":
            sizes["small"] = min(200, len(content))
            sizes["medium"] = min(500, len(content))
        else:
            sizes["small"] = min(100, len(content))
        sizes["full"] = len(content)
        manifest["splits"][name] = {
            tier: write(output / f"{name}_{tier}.parquet", content[:count])
            for tier, count in sizes.items()
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "pairs": len(all_pairs), "rejections": rejections,
        "splits": {k: {t: v["pairs"] for t, v in x.items()} for k, x in manifest["splits"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
