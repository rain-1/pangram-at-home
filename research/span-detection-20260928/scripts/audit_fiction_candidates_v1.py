"""Read-only provenance/count/overlap audit; never promotes candidates to training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fingerprints(text):
    words = normalized(text).split()
    values = set()
    for i in range(max(0, len(words) - 23)):
        value = hashlib.blake2b(" ".join(words[i:i + 24]).encode(), digest_size=8).digest()
        if value[0] < 16:  # content-selected 1/16 sample, robust to moved prefixes
            values.add(value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home/data"))
    args = parser.parse_args()
    root = args.root
    reference_paths = list((root / "diverse_pyramid_v1").glob("*_full.parquet"))
    reference_paths += [root / "raid_external_v1/frozen.parquet",
                        root / "standard_ebooks_v1/human.parquet"]
    reference_hashes, reference_prints = defaultdict(set), defaultdict(set)
    refs = {}
    for path in sorted(reference_paths):
        if not path.exists():
            raise FileNotFoundError(path)
        source = str(path.relative_to(root))
        table = pq.read_table(path, columns=["text"])
        refs[source] = {"rows": len(table), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for row in table.to_pylist():
            text = row["text"]
            reference_hashes[digest(normalized(text))].add(source)
            for value in fingerprints(text):
                reference_prints[value].add(source)

    packages, seen, duplicate_groups = {}, defaultdict(list), []
    # Explicit packages exclude acquisition retries/backups in sibling folders.
    inputs = [root / "fiction_candidates_v1" / name / "records.jsonl"
              for name in ("original_online", "fanfiction", "historical", "stories_in_the_wild")]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in inputs:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        overlap, counts, authors, works, lengths = [], Counter(), set(), set(), []
        for i, row in enumerate(rows):
            text = row["text"]
            clean_hash = row.get("clean_sha256")
            if clean_hash and clean_hash != digest(text):
                raise ValueError(f"Clean text hash mismatch: {path}:{i + 1}")
            lengths.append(len(text.split()))
            for key in ("source", "human_origin_status", "rights_status"):
                counts.update({f"{key}:{row.get(key, '<missing>')}": 1})
            authors.update(row.get("author_ids") or [])
            if row.get("work_id"):
                works.add(str(row["work_id"]))
            counts["missing_author_ids"] += not bool(row.get("author_ids"))
            counts["missing_work_id"] += not bool(row.get("work_id"))
            normhash = digest(normalized(text))
            sid = row.get("source_id") or row.get("id") or f"row:{i + 1}"
            seen[normhash].append({"package": path.parent.name, "source_id": sid})
            matching = Counter()
            for value in fingerprints(text):
                matching.update(reference_prints.get(value, set()))
            exact = sorted(reference_hashes.get(normhash, set()))
            if matching or exact:
                overlap.append({"source_id": sid, "exact_reference_matches": exact,
                                "shared_sampled_24word_windows": dict(matching)})
        packages[path.parent.name] = {
            "rows": len(rows), "words": sum(lengths), "unique_work_ids": len(works),
            "unique_author_identifiers": len(authors), "counts": dict(counts),
            "min_words": min(lengths, default=0), "max_words": max(lengths, default=0),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference_overlap_rows": overlap,
        }
    duplicate_groups = [group for group in seen.values() if len(group) > 1]
    output = {"role": "candidate audit only; no training or split assignment",
              "overlap_method": "normalized exact equality and content-selected 1/16 sample of 24-word windows",
              "limitations": "No sampled match is not proof of independence. Common phrases and quotations can produce matches. Author identifiers may be problem-scoped or aliases: their count need not equal real people. Review work/author metadata as well. Only listed reference files checked.",
              "references": refs, "packages": packages,
              "exact_duplicate_candidate_groups": duplicate_groups}
    target = root / "fiction_candidates_v1/audit.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in value.items() if k not in ("counts", "reference_overlap_rows")}
                      | {"reference_overlap_rows": len(value["reference_overlap_rows"])}
                      for name, value in packages.items()}, indent=2))


if __name__ == "__main__":
    main()
