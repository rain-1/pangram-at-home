#!/usr/bin/env python3
"""Build a held-out, provenance-labelled span eval slice from Stanford CoAuthor.

Input text stays outside the repository. Source events are Quill deltas, whose
indices count UTF-16 code units; emitted JSON offsets count Python characters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path


def worker_hash(worker_id: str) -> str:
    return hashlib.sha256(("pangram-coauthor-v1:" + worker_id).encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def char_index_at_utf16(chars: list[str], units: int, assume_bmp: bool = False) -> int:
    """Convert a Quill UTF-16 cursor offset to a Python codepoint index."""
    if assume_bmp:
        if units < 0 or units > len(chars):
            raise ValueError(f"Quill offset {units} is past document length {len(chars)}")
        return units
    pos = 0
    for i, ch in enumerate(chars):
        if pos == units:
            return i
        pos += 2 if ord(ch) > 0xFFFF else 1
        if pos > units:
            raise ValueError(f"Quill offset {units} splits a UTF-16 surrogate pair")
    if pos == units:
        return len(chars)
    raise ValueError(f"Quill offset {units} is past document length {pos}")


def apply_events(events: list[dict]) -> tuple[str, list[int | None], int]:
    """Replay deltas and return final text, labels (0 human/1 AI/None masked), edits masked."""
    initializers = [e for e in events if e.get("eventName") == "system-initialize"]
    if len(initializers) != 1:
        raise ValueError(f"expected one system-initialize event, found {len(initializers)}")
    init = initializers[0]
    if init is None or not isinstance(init.get("currentDoc"), str):
        raise ValueError("session has no system-initialize document")
    chars = list(init["currentDoc"])
    utf16_length = sum(2 if ord(ch) > 0xFFFF else 1 for ch in chars)
    has_nonbmp = utf16_length != len(chars)
    # Initial text contains the writing prompt and starter, without token authorship events.
    labels: list[int | None] = [None] * len(chars)
    origins: list[int | None] = [None] * len(chars)
    edit_contexts: list[int | None] = [None] * len(chars)
    inserted_by_event: list[int | None] = [None] * len(chars)
    tainted_origins: set[int] = set()
    tainted_edit_events = 0

    for ev in events:
        if ev.get("eventName") not in ("text-insert", "text-delete"):
            continue
        delta = ev.get("textDelta")
        if not isinstance(delta, dict):
            raise ValueError("text delta is not a Quill delta object")
        source = ev.get("eventSource")
        if source not in ("user", "api"):
            raise ValueError(f"unknown text event source: {source!r}")
        insertion_label = 1 if source == "api" else 0
        origin = int(ev.get("eventNum", 0)) if source == "api" else None
        event_id = int(ev.get("eventNum", 0))
        cursor_units = 0
        changed_ai = False
        edited_contexts: set[int] = set()
        inside_ai = False
        for op in delta.get("ops", []):
            if "retain" in op:
                cursor_units += int(op["retain"])
            elif "delete" in op:
                at = char_index_at_utf16(chars, cursor_units, assume_bmp=not has_nonbmp)
                delete_units = int(op["delete"])
                if not has_nonbmp:
                    end = at + delete_units
                    if end > len(chars):
                        raise ValueError("delete runs past document end")
                else:
                    remain_units = delete_units
                    end = at
                    while remain_units > 0 and end < len(chars):
                        w = 2 if ord(chars[end]) > 0xFFFF else 1
                        if w > remain_units:
                            raise ValueError("delete splits a UTF-16 surrogate pair")
                        remain_units -= w
                        end += 1
                    if remain_units:
                        raise ValueError("delete runs past document end")
                if source == "user" and (any(o is not None for o in origins[at:end])
                                           or any(c is not None for c in edit_contexts[at:end])):
                    changed_ai = True
                    deleted_origins = {o for o in origins[at:end] if o is not None}
                    edited_contexts.update(deleted_origins)
                    edited_contexts.update(c for c in edit_contexts[at:end] if c is not None)
                    tainted_origins.update(deleted_origins)
                del chars[at:end]
                del labels[at:end]
                del origins[at:end]
                del edit_contexts[at:end]
                del inserted_by_event[at:end]
                utf16_length -= delete_units
            elif "insert" in op:
                inserted = op["insert"]
                if not isinstance(inserted, str):
                    raise ValueError("non-text insertion encountered")
                at = char_index_at_utf16(chars, cursor_units, assume_bmp=not has_nonbmp)
                if source == "user":
                    # Insertion inside a surviving generated suggestion is an edit, so mask it.
                    left = origins[at - 1] if at else None
                    right = origins[at] if at < len(origins) else None
                    left_context = edit_contexts[at - 1] if at else None
                    right_context = edit_contexts[at] if at < len(edit_contexts) else None
                    inherited_contexts = {x for x in (left, right, left_context, right_context) if x is not None}
                    inside_ai = (left is not None and left == right) or any(
                        x in tainted_origins for x in inherited_contexts
                    ) or left_context is not None or right_context is not None
                    if inside_ai:
                        changed_ai = True
                        edited_contexts.update(inherited_contexts)
                        tainted_origins.update(x for x in inherited_contexts if x in origins)
                inserted_chars = list(inserted)
                chars[at:at] = inserted_chars
                # Multi-character user deltas can be pasted or composed input. The
                # source event alone cannot prove who authored that block of text.
                label = insertion_label if source == "api" or (len(inserted_chars) == 1 and not inside_ai) else None
                labels[at:at] = [label] * len(inserted_chars)
                origins[at:at] = [origin] * len(inserted_chars)
                context = next(iter(edited_contexts), None) if source == "user" else None
                edit_contexts[at:at] = [context] * len(inserted_chars)
                inserted_by_event[at:at] = [event_id] * len(inserted_chars)
                inserted_units = sum(2 if ord(ch) > 0xFFFF else 1 for ch in inserted_chars)
                cursor_units += inserted_units
                utf16_length += inserted_units
                has_nonbmp = has_nonbmp or inserted_units != len(inserted_chars)
            else:
                # Attribute-only ops do not change text.
                continue
        if changed_ai:
            tainted_edit_events += 1
            # Text typed in the same transaction as an edit to generated material has
            # ambiguous authorship; preserve it in the sample but exclude it from scoring.
            if source == "user":
                for i, event_num in enumerate(inserted_by_event):
                    if event_num == event_id:
                        labels[i] = None
                        edit_contexts[i] = next(iter(edited_contexts), None)

        # The official interface replays the event stream by applying each textDelta
        # to the current Quill document, then restores the logged cursor. Validate that
        # every recorded cursor remains a legal offset into the reconstructed document.
        if "currentCursor" in ev:
            current_cursor = int(ev["currentCursor"])
            if not 0 <= current_cursor <= utf16_length:
                raise ValueError(f"logged cursor {current_cursor} is outside UTF-16 document length {utf16_length}")

    if (len(chars) != len(labels) or len(chars) != len(origins)
            or len(chars) != len(edit_contexts) or len(chars) != len(inserted_by_event)):
        raise AssertionError("replay arrays lost alignment")
    if tainted_origins:
        labels = [None if o in tainted_origins else y for o, y in zip(origins, labels)]
    return "".join(chars), labels, tainted_edit_events


def replay_text_independently(events: list[dict]) -> str:
    """Plain text-only Quill Delta replay, kept independent of authorship labels."""
    init = next(e for e in events if e["eventName"] == "system-initialize")
    chars = list(init["currentDoc"])
    utf16_length = sum(2 if ord(ch) > 0xFFFF else 1 for ch in chars)
    has_nonbmp = utf16_length != len(chars)
    for ev in events:
        if ev.get("eventName") not in ("text-insert", "text-delete"):
            continue
        cursor_units = 0
        for op in ev["textDelta"].get("ops", []):
            if "retain" in op:
                cursor_units += int(op["retain"])
            elif "delete" in op:
                start = char_index_at_utf16(chars, cursor_units, assume_bmp=not has_nonbmp)
                delete_units = int(op["delete"])
                if not has_nonbmp:
                    end = start + delete_units
                    if end > len(chars):
                        raise ValueError("delete runs past document end")
                else:
                    remaining = delete_units
                    end = start
                    while remaining and end < len(chars):
                        width = 2 if ord(chars[end]) > 0xFFFF else 1
                        if width > remaining:
                            raise ValueError("delete splits a UTF-16 surrogate pair")
                        remaining -= width
                        end += 1
                    if remaining:
                        raise ValueError("delete runs past document end")
                del chars[start:end]
                utf16_length -= delete_units
            elif "insert" in op:
                at = char_index_at_utf16(chars, cursor_units, assume_bmp=not has_nonbmp)
                value = str(op["insert"])
                chars[at:at] = list(value)
                inserted_units = sum(2 if ord(ch) > 0xFFFF else 1 for ch in value)
                cursor_units += inserted_units
                utf16_length += inserted_units
                has_nonbmp = has_nonbmp or inserted_units != len(value)
    return "".join(chars)


def spans_from_labels(labels: list[int | None]) -> list[dict]:
    if not labels:
        return []
    out = []
    start = 0
    label = labels[0]
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != label:
            out.append({"start": start, "end": i, "label": -100 if label is None else label})
            if i < len(labels):
                start, label = i, labels[i]
    return out


def load_metadata(paths: list[Path]) -> dict[str, dict]:
    by_session = {}
    for path in paths:
        name = path.name.lower()
        writing_type = "creative" if "creative" in name else "argumentative" if "argument" in name else "unknown"
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row["session_id"].strip()
                if not sid:
                    continue
                row["_writing_type"] = writing_type
                by_session[sid] = row
    return by_session


def choose_workers(metadata: dict[str, dict], seed: int, target: int, max_sessions_per_worker: int):
    by_worker: dict[str, list[str]] = {}
    for sid, row in metadata.items():
        by_worker.setdefault(row["worker_id"], []).append(sid)
    eligible = [w for w, sids in by_worker.items() if len(sids) <= max_sessions_per_worker]
    random.Random(seed).shuffle(eligible)
    chosen: list[str] = []
    n = 0
    for worker in eligible:
        count = len(by_worker[worker])
        if n >= target:
            break
        if n + count <= target + 40:
            chosen.append(worker)
            n += count
    return set(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=Path("/mnt/f/pangram-at-home/data/coauthor.zip"))
    ap.add_argument("--metadata", type=Path, nargs="+", default=[
        Path("/mnt/f/pangram-at-home/data/coauthor_creative.csv"),
        Path("/mnt/f/pangram-at-home/data/coauthor_argumentative.csv"),
    ])
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/f/pangram-at-home/data/span_realistic_eval_v1"))
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--target-sessions", type=int, default=200)
    ap.add_argument("--max-sessions-per-worker", type=int, default=25)
    ap.add_argument("--min-scored-chars", type=int, default=100)
    ap.add_argument("--min-chars-per-class", type=int, default=50)
    args = ap.parse_args()
    metadata = load_metadata(args.metadata)
    selected_workers = choose_workers(metadata, args.seed, args.target_sessions, args.max_sessions_per_worker)
    selected_sids = {sid for sid, row in metadata.items() if row["worker_id"] in selected_workers}

    rows = []
    audit = Counter()
    observed_workers = set()
    source_session_sha256 = {}
    with zipfile.ZipFile(args.archive) as zf:
        for name in zf.namelist():
            sid = Path(name).stem
            if not name.endswith(".jsonl") or sid not in selected_sids:
                continue
            payload = zf.read(name)
            source_session_sha256[sid] = hashlib.sha256(payload).hexdigest()
            events = [json.loads(line) for line in payload.splitlines()]
            init_count = sum(e.get("eventName") == "system-initialize" for e in events)
            if init_count != 1:
                audit["excluded_invalid_initialize_count"] += 1
                continue
            audit["source_sessions_parsed"] += 1
            selects = sum(e.get("eventName") == "suggestion-select" for e in events)
            api_insertions = sum(e.get("eventName") == "text-insert" and e.get("eventSource") == "api"
                                 for e in events)
            audit["suggestion_select_events"] += selects
            audit["api_text_insert_events"] += api_insertions
            pending_selects = 0
            selection_pairing_ok = True
            for event in events:
                if event.get("eventName") == "suggestion-select":
                    pending_selects += 1
                elif event.get("eventName") == "text-insert" and event.get("eventSource") == "api":
                    if pending_selects == 0:
                        selection_pairing_ok = False
                        break
                    pending_selects -= 1
            if not selection_pairing_ok or pending_selects or selects != api_insertions:
                audit["excluded_unpaired_suggestion_events"] += 1
                continue
            final_text, labels, edited_events = apply_events(events)
            if final_text != replay_text_independently(events):
                raise AssertionError(f"label replay changed the final text for {sid}")
            row_meta = metadata[sid]
            observed_workers.add(row_meta["worker_id"])
            counts = Counter(labels)
            if counts[0] + counts[1] < args.min_scored_chars:
                audit["excluded_too_few_scored_chars"] += 1
                continue
            if counts[0] < args.min_chars_per_class:
                audit["excluded_human_below_class_minimum"] += 1
                continue
            if counts[1] < args.min_chars_per_class:
                audit["excluded_ai_below_class_minimum"] += 1
                continue
            writing_type = row_meta["_writing_type"]
            rows.append({
                "id": f"coauthor_{sid}",
                "text": final_text,
                "spans": spans_from_labels(labels),
                "kind": "mixed",
                "construction": "observed_human_ai_cowriting",
                "source": "CoAuthor (Lee et al., CHI 2022)",
                "domain": writing_type,
                "source_ids": [sid],
                "source_groups": ["coauthor_writer_" + worker_hash(row_meta["worker_id"])],
                "scored_char_counts": {"human": counts[0], "ai": counts[1], "ignored": counts[None]},
                "provenance": {
                    "human_source": "text-insert events eventSource=user",
                    "ai_source": "text-insert events eventSource=api",
                    "ignored_source": "prompt/starter text and AI suggestions touched by later user edits",
                    "edited_ai_event_count": edited_events,
                },
            })
            audit["included_mixed_sessions"] += 1
            audit["scored_human_chars"] += counts[0]
            audit["scored_ai_chars"] += counts[1]
            audit["ignored_chars"] += counts[None]

    if not rows:
        raise RuntimeError("No mixed, adequately long sessions survived provenance filtering")
    if audit["suggestion_select_events"] != audit["api_text_insert_events"]:
        raise RuntimeError("selected suggestion and API insertion event counts differ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "test.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "CoAuthor realistic mixed-authorship span evaluation v1",
        "seed": args.seed,
        "target_sessions": args.target_sessions,
        "minimum_scored_chars_per_session": args.min_scored_chars,
        "minimum_chars_per_authorship_class": args.min_chars_per_class,
        "selected_worker_count": len(selected_workers),
        "selected_session_count": len(selected_sids),
        "selected_session_ids": sorted(selected_sids),
        "selected_writer_groups": sorted("coauthor_writer_" + worker_hash(w) for w in selected_workers),
        "included_session_ids": sorted(row["source_ids"][0] for row in rows),
        "observed_worker_count": len(observed_workers),
        "counts": dict(audit),
        "label_ids": {"human": 0, "ai": 1, "ignored": -100},
        "offset_unit": "Unicode codepoints; exclusive end",
        "source_archive": str(args.archive),
        "source_archive_sha256": sha256_file(args.archive),
        "selected_source_session_sha256": source_session_sha256,
        "selected_source_files_missing": sorted(selected_sids - set(source_session_sha256)),
        "source_metadata": [str(p) for p in args.metadata],
        "source_metadata_sha256": {str(p): sha256_file(p) for p in args.metadata},
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "citation": "Mina Lee, Percy Liang, and Qian Yang. 2022. CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities. CHI 2022.",
        "source_home": "https://coauthor.stanford.edu/",
        "source_archive_url": "https://drive.google.com/file/d/1C9FCCsyY-5I7mcBHi-__R7lxHkGX_-9Q/view?usp=sharing",
        "source_metadata_url": "https://docs.google.com/spreadsheets/d/1O3EXJm52TQHfFSbzVGZmNIzzdu5ow6IjnOBrGTUY02o/edit?usp=sharing",
        "license_note": "The official CoAuthor landing page links the downloadable archive and the paper encourages research reuse, but neither states an explicit dataset license. Keep raw and derived text private/external for this evaluation; do not publish or redistribute without rights review. The interface repository's MIT license covers interface code only.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
