"""Group whole essays before making author-probe splits; never detector training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_url(url):
    parsed = urlsplit(url)
    return urlunsplit(("https", parsed.netloc.lower().removeprefix("www."),
                       parsed.path.rstrip("/"), "", ""))


def content_fingerprints(text):
    words = re.findall(r"\w+", text.casefold())
    # Content-selected windows, rather than positional subsampling, also match
    # after inserted prefaces. Fingerprints nominate candidate duplicate pairs.
    return {value for i in range(max(0, len(words) - 23))
            if (value := hashlib.blake2b(" ".join(words[i:i+24]).encode(), digest_size=8).digest())[0] < 32}


def group_essays(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(i)] = find(j)
    seen = {}
    prints = []
    candidates = defaultdict(set)
    for i, row in enumerate(rows):
        text = " ".join(re.findall(r"\w+", row["text"].casefold()))
        keys = [("text", sha(text))]
        if row.get("canonical_url"):
            keys.append(("url", canonical_url(row["canonical_url"])))
        if row.get("group_id"):
            keys.append(("source_group", str(row["group_id"])))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
        fingerprint = content_fingerprints(row["text"])
        prints.append(fingerprint)
        possible = set().union(*(candidates[value] for value in fingerprint)) if fingerprint else set()
        for j in possible:
            overlap = len(fingerprint & prints[j])
            shorter = min(len(fingerprint), len(prints[j]))
            if shorter >= 10 and overlap >= 10 and overlap / shorter >= .6:
                join(i, j)
        for value in fingerprint:
            candidates[value].add(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def partition(groups):
    by_author = defaultdict(list)
    conflicts = []
    for group in groups:
        authors = {r["author_id"] for r in group}
        if len(authors) != 1:
            conflicts.extend(group)
            continue
        # Select one version per connected group, avoiding repeated essays.
        ordered = sorted(group, key=lambda r: (r.get("publication_date") or "9999", r["document_id"]))
        row = dict(ordered[0])
        row["probe_group_id"] = "author_work:" + sha("\n".join(sorted(r["document_id"] for r in group)))[:24]
        row["group_document_ids"] = [r["document_id"] for r in group]
        row["label"] = row["author_id"]
        by_author[row["author_id"]].append(row)
    splits = {"train": [], "val": [], "test": []}
    for author, rows in sorted(by_author.items()):
        if len(rows) < 7:
            raise ValueError(f"{author} has only {len(rows)} independent essays; need at least 7")
        rows.sort(key=lambda r: (r.get("publication_date") or "9999", r["probe_group_id"]))
        holdout = max(1, round(.15 * len(rows)))
        for name, selected in (("train", rows[:-2*holdout]),
                               ("val", rows[-2*holdout:-holdout]), ("test", rows[-holdout:])):
            splits[name].extend(selected)
    return splits, conflicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home/data"))
    parser.add_argument("--pool", choices=["pre2023_candidates", "attributed"], default="pre2023_candidates")
    parser.add_argument("--output-folder", default="author_attribution_probe_v1")
    args = parser.parse_args()
    output = args.root / args.output_folder
    if output.exists():
        raise SystemExit(f"Refusing to overwrite frozen splits: {output}")
    inputs = sorted((args.root / "attribution_authors_v1").glob("*/records.jsonl"))
    if len(inputs) < 2:
        raise ValueError("Both author acquisition packages are required")
    candidates, excluded = [], []
    for path in inputs:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for key in ("author_id", "document_id", "text", "attribution_eligible", "human_origin_status"):
                if key not in row:
                    raise ValueError(f"{path}: missing {key}")
            permitted = row["attribution_eligible"] and len(row["text"].split()) >= 150
            if args.pool == "pre2023_candidates":
                permitted &= bool(row.get("strict_human_candidate"))
                permitted &= bool(row.get("publication_date") and row["publication_date"][:4] <= "2022")
            (candidates if permitted else excluded).append(row)
    splits, conflicts = partition(group_essays(candidates))
    authors = set(r["author_id"] for r in candidates)
    if len(authors) < 4:
        raise ValueError(f"Expected all four requested authors, found {sorted(authors)}")
    assert all({r["author_id"] for r in rows} == authors for rows in splits.values())
    groups = {name: {r["probe_group_id"] for r in rows} for name, rows in splits.items()}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    output.mkdir(parents=True)
    manifest = {"role": "separate author attribution probe; never human/AI detector training",
                "pool": args.pool, "human_authorship_guaranteed": False,
                "split_policy": "whole-work connected groups; oldest train, then validation, newest test, per author",
                "grouping": "source groups, canonical URL, normalized text equality, and >=60% sampled 24-word containment",
                "selection": "one earliest dated version per connected group; conflicting author groups excluded",
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "excluded_documents": len(excluded), "conflicting_author_documents": len(conflicts), "splits": {}}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][name] = {"documents": len(rows), "authors": dict(Counter(r["author_id"] for r in rows)),
                                   "words": sum(len(r["text"].split()) for r in rows),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
