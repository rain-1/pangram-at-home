"""Freeze independent, pure-human span calibration and research evaluation rows.

The input Parquets already live outside Git. The locked set excludes records
scored in earlier passage-model audits. Writers posts are split by connected
components of thread and owner, so neither relation crosses the split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


ROOT = Path("/mnt/f/pangram-at-home/data")
SOURCES = {
    "standard_ebooks_v1": "human.parquet",
    "federal_reserve_beige_book_v1": "human.parquet",
    "stackexchange_writers_v1": "human_eval.parquet",
    "persuade_essays_v1": "human_eval.parquet",
}
SE_AUDIT_PREFIX = "stackexchange-audit-v1:"
ESSAY_AUDIT_PREFIX = "persuade-audit-v1:"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ranked(values, salt: str):
    return sorted(values, key=lambda value: sha((salt + str(value)).encode()))


def audit_ids(rows: list[dict], prefix: str, count: int = 1000) -> set[str]:
    return {row["text_id"] for row in sorted(
        rows, key=lambda row: sha((prefix + row["text_id"]).encode()))[:count]}


def components(rows: list[dict]) -> list[list[dict]]:
    """Connected components over Writers thread IDs and owner IDs."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for field in ("group_id", "owner_user_id"):
        first = {}
        for i, row in enumerate(rows):
            value = row[field]
            if value in first:
                parent[find(i)] = find(first[value])
            else:
                first[value] = i
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[find(i)].append(row)
    return list(grouped.values())


def doc(text: str, source: str, domain: str, source_ids: list[str],
        source_groups: list[str], construction: str, role: str,
        provenance: list[dict]) -> dict:
    digest = sha((source + "\n" + text).encode())
    return {
        "id": f"human_eval_v2:{role}:{source}:{digest[:20]}",
        "text": text,
        "spans": [{"start": 0, "end": len(text), "label": 0}],
        "kind": "human", "construction": construction,
        "source": source, "domain": domain,
        "source_ids": source_ids, "source_groups": source_groups,
        "source_records": provenance,
        "human_provenance": "source archive / competition provenance; residual label uncertainty documented in manifest",
    }


def from_row(row: dict, role: str) -> dict:
    source = row["source"]
    return doc(row["text"], source, row["domain"],
               [str(row["source_id"])], [str(row["group_id"])],
               "unaltered source passage", role,
               [{"text_id": row["text_id"], "canonical_url": row.get("canonical_url"),
                 "clean_sha256": row.get("clean_sha256"),
                 "date": row.get("created_at") or row.get("original_publication_date")
                 or row.get("collection_period") or row.get("snapshot_date")}])


def make_long(rows: list[dict], role: str) -> dict:
    text = "\n\n".join(row["text"] for row in rows)
    out = doc(text, rows[0]["source"], rows[0]["domain"],
              [str(row["source_id"]) for row in rows],
              [str(row["group_id"]) for row in rows],
              "consecutive passages from one original work/report", role,
              [{"text_id": row["text_id"], "canonical_url": row.get("canonical_url"),
                "clean_sha256": row.get("clean_sha256")} for row in rows])
    return out


def calibration_work_rows(rows: list[dict], role: str, long_blocks: int = 4) -> list[dict]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    output = []
    for group in sorted(by_group):
        # Extraction text IDs encode the source order, including chapter order
        # for books and report order for Beige Book releases.
        ordered = sorted(by_group[group], key=lambda row: int(row["text_id"].rsplit(":", 1)[-1]))
        for row in ordered[:6]:
            output.append(from_row(row, role))
        # Long examples per work/report use disjoint original passage ranges.
        for block in range(long_blocks):
            start = 6 + block * 3
            chunk = ordered[start:start + 3]
            if len(chunk) >= 3:
                output.append(make_long(chunk, role))
    return output


def word_shingles(text: str, width: int = 24) -> set[bytes]:
    words = re.findall(r"\b\w+\b", text.casefold())
    return {hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words) - width + 1))}


def parent_fingerprints(root: Path) -> tuple[set[str], set[bytes]]:
    parent = root / "diverse_pyramid_v1"
    hashes = set()
    shingles = set()
    for split in ("train", "val", "test"):
        table = pq.read_table(parent / f"{split}_full.parquet", columns=["text", "text_sha256"])
        for row in table.to_pylist():
            hashes.add(row["text_sha256"])
            hashes.add(sha(row["text"].encode()))
            shingles.update(word_shingles(row["text"]))
    return hashes, shingles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "span_human_eval_v2"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite frozen corpus: {output}")

    source_rows = {name: pq.read_table(root / name / filename).to_pylist()
                   for name, filename in SOURCES.items()}
    books = source_rows["standard_ebooks_v1"]
    beige = source_rows["federal_reserve_beige_book_v1"]
    writers = source_rows["stackexchange_writers_v1"]
    essays = source_rows["persuade_essays_v1"]
    writers_audit = audit_ids(writers, SE_AUDIT_PREFIX)
    essays_audit = audit_ids(essays, ESSAY_AUDIT_PREFIX)

    writer_comps = components(writers)
    fresh_comps = [comp for comp in writer_comps if not any(
        row["text_id"] in writers_audit for row in comp)]
    examined_comps = [comp for comp in writer_comps if any(
        row["text_id"] in writers_audit for row in comp)]
    fresh_comps = ranked(fresh_comps, "span-human-eval-v2:component:")
    # Fill around half of the untouched components; keep each whole component.
    fresh_cal, fresh_eval = [], []
    for comp in fresh_comps:
        (fresh_cal if len(fresh_cal) < 500 else fresh_eval).extend(comp)
    # Earlier audit text is eligible for calibration but never for locked eval.
    examined_rows = [row for comp in examined_comps for row in comp]
    examined_cal = ranked(examined_rows, "span-human-eval-v2:prior-audit-cal:")[:100]

    essay_fresh = [row for row in essays if row["text_id"] not in essays_audit]
    by_length = defaultdict(list)
    for row in essay_fresh:
        words = row["word_count"]
        band = "short" if words < 300 else "medium" if words < 650 else "long"
        by_length[band].append(row)
    selected_essays = []
    for band in ("short", "medium", "long"):
        selected_essays.extend(ranked(by_length[band], f"span-human-eval-v2:essay:{band}:")[:1000])

    calibration = (calibration_work_rows(books, "calibration", long_blocks=14) +
                   calibration_work_rows(beige, "calibration") +
                   [from_row(row, "calibration") for row in fresh_cal + examined_cal])
    locked = ([from_row(row, "locked_eval") for row in fresh_eval] +
              [from_row(row, "locked_eval") for row in selected_essays])
    # Exact-parent text checks. The span pilot and confirmation set use the
    # diverse pyramid as parents, so this covers their source excerpts too.
    parent_hashes, parent_shingles = parent_fingerprints(root)
    for role, records in (("calibration", calibration), ("locked_eval", locked)):
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate document IDs in {role}")
        for record in records:
            if sha(record["text"].encode()) in parent_hashes:
                raise AssertionError(f"Text overlaps diverse parent: {record['id']}")
            if word_shingles(record["text"]) & parent_shingles:
                raise AssertionError(f"24-word passage overlaps diverse parent: {record['id']}")
            assert record["spans"] == [{"start": 0, "end": len(record["text"]), "label": 0}]
    cal_ids = {item["text_id"] for row in calibration for item in row["source_records"]}
    eval_ids = {item["text_id"] for row in locked for item in row["source_records"]}
    if cal_ids & eval_ids:
        raise AssertionError("Source text IDs cross calibration and locked evaluation")
    cal_groups = {g for row in calibration for g in row["source_groups"]}
    eval_groups = {g for row in locked for g in row["source_groups"]}
    if cal_groups & eval_groups:
        raise AssertionError("Source groups cross calibration and locked evaluation")
    writer_eval_ids = {item["text_id"] for row in locked if row["source"] == "writers.stackexchange.com"
                       for item in row["source_records"]}
    essay_eval_ids = {item["text_id"] for row in locked if row["source"] == "persuade_2.0"
                      for item in row["source_records"]}
    assert not writer_eval_ids & writers_audit
    assert not essay_eval_ids & essays_audit
    assert not ({r["owner_user_id"] for r in fresh_cal + examined_cal} &
                {r["owner_user_id"] for r in fresh_eval})

    output.mkdir(parents=True)
    for role, records in (("calibration", calibration), ("locked_eval", locked)):
        path = output / ("test.jsonl" if role == "locked_eval" else "calibration.jsonl")
        with path.open("w") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "role": "independent pure-human calibration and frozen evaluation for binary span model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {name: {"path": str(root / name / filename),
                                "sha256": sha((root / name / filename).read_bytes())}
                         for name, filename in SOURCES.items()},
        "parent": "diverse_pyramid_v1 train/val/test; span pilot and confirmation derive from these",
        "parent_overlap_check": "exact UTF-8 SHA-256 and all normalized 24-word shingles against diverse_pyramid_v1 train/val/test full texts; zero matches; span pilot and confirmation source excerpts derive from this parent",
        "previous_audits": {"writers_rows": len(writers_audit), "essay_rows": len(essays_audit),
                            "writers_thread_and_owner_components_touched": len(examined_comps),
                            "writers_rows_in_touched_components": len(examined_rows),
                            "writers_rows_in_untouched_components": sum(map(len, fresh_comps)),
                            "standard_ebooks": "all passages previously evaluated",
                            "beige_book": "all passages previously evaluated"},
        "splits": {},
        "limitations": [
            "Public-domain works and government reports were previously examined in passage-model audits; calibration only.",
            "Locked evaluation uses previously unscored records from source families already inspected; it is source-family holdout, not a new-platform blind test.",
            "Writers post and last edit dates precede 2023, but the 2024 dump and metadata do not prove every passage is human authored.",
            "PERSUADE essays are student writing from 2021–2022; source author IDs are unavailable, so essay-level IDs cannot guarantee distinct students.",
            "PERSUADE text is CC BY-NC-SA 4.0 and sensitive student writing; Writers posts are CC BY-SA. Both remain local research evaluation data only.",
            "Only ten book works and 32 Beige Book reports underlie the calibration fiction/finance examples; chunk count is not work count.",
        ],
    }
    for role, records in (("calibration", calibration), ("locked_eval", locked)):
        path = output / ("test.jsonl" if role == "locked_eval" else "calibration.jsonl")
        manifest["splits"][role] = {
            "rows": len(records), "sha256": sha(path.read_bytes()),
            "by_source": dict(Counter(row["source"] for row in records)),
            "by_domain": dict(Counter(row["domain"] for row in records)),
            "distinct_source_groups": len({g for row in records for g in row["source_groups"]}),
            "distinct_source_record_ids": len({item["text_id"] for row in records for item in row["source_records"]}),
            "word_length": {"min": min(len(row["text"].split()) for row in records),
                            "median": sorted(len(row["text"].split()) for row in records)[len(records)//2],
                            "max": max(len(row["text"].split()) for row in records)},
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
