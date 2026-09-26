"""Parse only DAMASHA clean tags into auditable binary character spans."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


REVISION = "17061412e00ff91dd86ef2f05efe82b08ac3d669"
OPEN = "<AI_Start>"
CLOSE = "</AI_End>"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def parse_tags(marked: str) -> tuple[str, list[dict]]:
    text = ""
    spans = []
    cursor = 0
    role = 0
    while cursor < len(marked):
        next_open = marked.find(OPEN, cursor)
        next_close = marked.find(CLOSE, cursor)
        occurrences = [(i, 1, OPEN) for i in (next_open,) if i >= 0]
        occurrences += [(i, 0, CLOSE) for i in (next_close,) if i >= 0]
        if occurrences:
            position, target_role, marker = min(occurrences)
        else:
            position, target_role, marker = len(marked), role, ""
        if position > cursor:
            piece = marked[cursor:position]
            start = len(text)
            text += piece
            if spans and spans[-1]["label"] == role:
                spans[-1]["end"] = len(text)
            else:
                spans.append({"start": start, "end": len(text), "label": role})
        if marker:
            if target_role == role:
                raise ValueError("Repeated or unmatched authorship tag")
            role = target_role
            cursor = position + len(marker)
        else:
            break
    if role != 0 or not spans or spans[0]["start"] != 0 or spans[-1]["end"] != len(text):
        raise ValueError("Unclosed or empty tagged document")
    if any(a["end"] != b["start"] for a, b in zip(spans, spans[1:])):
        raise ValueError("Span gap")
    if not any(span["label"] == 1 for span in spans):
        raise ValueError("No AI span")
    return text, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home/data/span_sources_v5"))
    args = ap.parse_args()
    src = args.root / "damasha/DAMASHA_Final_No_ADV.csv"
    dest = args.root / "normalized_damasha_clean"
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    dest.mkdir(parents=True)
    csv.field_size_limit(10_000_000)
    output = dest / "candidate_only.jsonl"
    counts = Counter()
    seen = set()
    with src.open(newline="") as input_file, output.open("w") as output_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["hybrid_text", "has_pair "]:
            raise ValueError("Unexpected DAMASHA source columns")
        for line_number, row in enumerate(reader, start=2):
            counts["source_rows"] += 1
            try:
                text, spans = parse_tags(row["hybrid_text"])
            except ValueError:
                counts["excluded_malformed_markup"] += 1
                continue
            thash = sha(text)
            if thash in seen:
                counts["excluded_exact_duplicate_text"] += 1
                continue
            seen.add(thash)
            result = {"id": f"damasha_clean:line:{line_number}",
                      "text": text, "spans": spans,
                      "source": "DAMASHA clean published aggregate",
                      "source_revision": REVISION, "source_line": line_number,
                      "group_id": None, "upstream_source": None,
                      "source_provenance_status": "aggregate CSV has no upstream corpus, document or prompt ID",
                      "text_sha256": thash}
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts["candidate_documents"] += 1
            counts["multiple_ai_spans"] += sum(s["label"] == 1 for s in spans) > 1
            counts["shorter_than_80_words"] += len(text.split()) < 80
            counts["human_characters"] += sum(s["end"] - s["start"] for s in spans if s["label"] == 0)
            counts["ai_characters"] += sum(s["end"] - s["start"] for s in spans if s["label"] == 1)
    manifest = {"source_revision": REVISION,
                "role": "candidate only; no train/validation/test split assigned",
                "source_sha256": sha(src.read_bytes()),
                "output_sha256": sha(output.read_bytes()),
                "counts": dict(counts),
                "caveat": "CSV includes the authors' new records plus TriBERT and M4GT; source IDs are absent, so do not count these as independent corpora or randomly split for evaluation."}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
