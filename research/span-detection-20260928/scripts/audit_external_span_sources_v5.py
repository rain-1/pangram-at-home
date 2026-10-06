"""Read-only exact and sampled phrase overlap audit for new span corpora."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


def words(text):
    return re.findall(r"\w+", text.casefold())


def exact_hash(text):
    return hashlib.sha256(" ".join(words(text)).encode()).digest()


def phrase_hashes(text):
    tokens = words(text)
    result = set()
    for i in range(max(0, len(tokens) - 23)):
        h = hashlib.blake2b(" ".join(tokens[i:i+24]).encode(), digest_size=8).digest()
        if h[0] < 16:
            result.add(h)
    return result


def jsonl_rows(path):
    with path.open() as file:
        for line in file:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home/data"))
    args = ap.parse_args()
    root = args.root
    references = {
        "old_span_train": root / "span_training_v4/train.jsonl",
        "old_span_val": root / "span_training_v4/val.jsonl",
        "human_calibration": root / "span_human_eval_v2/calibration.jsonl",
        "locked_human_test": root / "span_human_eval_v2/test.jsonl",
        "locked_coauthor_test": root / "span_realistic_eval_v1/test.jsonl",
    }
    reference_hashes = {}
    reference_phrases = {}
    counts = {}
    for name, path in references.items():
        e, p = set(), set()
        n = 0
        for row in jsonl_rows(path):
            text = row["text"]
            e.add(exact_hash(text))
            p.update(phrase_hashes(text))
            n += 1
        reference_hashes[name] = e
        reference_phrases[name] = p
        counts[name] = {"rows": n, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "exact_hashes": len(e), "sampled_phrase_hashes": len(p)}
    diverse_path = root / "diverse_pyramid_v1/test_full.parquet"
    e, p = set(), set()
    for row in pq.read_table(diverse_path, columns=["text"]).to_pylist():
        e.add(exact_hash(row["text"]))
        p.update(phrase_hashes(row["text"]))
    reference_hashes["diverse_frozen_test"] = e
    reference_phrases["diverse_frozen_test"] = p
    counts["diverse_frozen_test"] = {"rows": pq.read_metadata(diverse_path).num_rows,
                                     "file_sha256": hashlib.sha256(diverse_path.read_bytes()).hexdigest(),
                                     "exact_hashes": len(e), "sampled_phrase_hashes": len(p)}

    base = root / "span_sources_v5"
    sources = {
        "llmtrace_train": base / "normalized_llmtrace_en/train.jsonl",
        "llmtrace_val": base / "normalized_llmtrace_en/val.jsonl",
        "llmtrace_test": base / "normalized_llmtrace_en/test.jsonl",
        "aitdna_locked_test": base / "normalized_aitdna_real/locked_test.jsonl",
        "damasha_candidate": base / "normalized_damasha_clean/candidate_only.jsonl",
    }
    results = {}
    for name, path in sources.items():
        c = defaultdict(Counter)
        hit_ids = defaultdict(list)
        protected_groups = defaultdict(set)
        n = 0
        for row in jsonl_rows(path):
            n += 1
            th = exact_hash(row["text"])
            ph = phrase_hashes(row["text"])
            for reference in references | {"diverse_frozen_test": diverse_path}:
                exact = th in reference_hashes[reference]
                common = len(ph & reference_phrases[reference])
                if exact or common:
                    c[reference]["rows_with_any_match"] += 1
                    c[reference]["exact_rows"] += exact
                    c[reference]["rows_with_3plus_sampled_24word_matches"] += common >= 3
                    if len(hit_ids[reference]) < 40:
                        hit_ids[reference].append({"id": row["id"], "exact": exact,
                                                   "sampled_phrase_matches": common})
                    if row.get("group_id") and reference in ("locked_human_test", "locked_coauthor_test", "diverse_frozen_test"):
                        protected_groups[reference].add(row["group_id"])
        results[name] = {"rows": n, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "matches": {reference: dict(value) for reference, value in c.items()},
                         "first_match_ids": dict(hit_ids),
                         "topic_groups_touching_locked_test": {reference: len(groups) for reference, groups in protected_groups.items()}}
    output = {"role": "audit only; no training selection",
              "method": "normalized exact text and content-selected 1/16 sample of 24-word windows",
              "caveat": "A missing sampled hit does not prove semantic independence; a phrase hit can be boilerplate or quotation. Inspect source work and prompt lineage before using matches to exclude groups.",
              "references": counts, "sources": results}
    dest = base / "overlap_audit.json"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {"rows": item["rows"], "matches": item["matches"],
                             "groups_touching_locked_test": item["topic_groups_touching_locked_test"]}
                      for name, item in results.items()}, indent=2))


if __name__ == "__main__":
    main()
