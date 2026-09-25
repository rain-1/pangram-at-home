"""Extract a bounded pre-2023 Stack Exchange human-labeled Q&A audit set.

Supply Posts.xml from an official Stack Exchange historical data dump. This
does not download the multi-gigabyte archive. The resulting rows are for local
research evaluation; pre-2023 provenance is not proof that every post was
written by a human or never edited later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup


CUTOFF = "2022-12-31T23:59:59Z"
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_body(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    # Code is a very strong authorship/domain cue and often includes copied code.
    for node in soup.select("pre, code, script, style"):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    # Avoid retaining direct contact details in a public-post corpus.
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    return text


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("posts_xml", type=Path, help="Posts.xml extracted from an official historical dump")
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/data/stackoverflow_human_eval_v1"))
    parser.add_argument("--snapshot", required=True, help="Dump release label/date, e.g. 'October 2022'")
    parser.add_argument("--site", default="stackoverflow.com", help="Stack Exchange host, e.g. writers.stackexchange.com")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=700)
    parser.add_argument("--users-xml", type=Path, help="Optional Users.xml from the same dump, used to exclude obvious bot/community accounts")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if not (args.site == "stackoverflow.com" or re.fullmatch(r"[a-z0-9-]+\.stackexchange\.com", args.site)):
        parser.error("--site must be a Stack Exchange or Stack Overflow host")
    site = args.site
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if not args.posts_xml.is_file():
        parser.error(f"Posts.xml not found: {args.posts_xml}")

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    bot_ids: set[str] = set()
    if args.users_xml:
        for _, user in ET.iterparse(args.users_xml, events=("end",)):
            if user.tag == "row":
                name = user.attrib.get("DisplayName", "").lower()
                if "bot" in name or "community" in name:
                    bot_ids.add(user.attrib.get("Id", ""))
                user.clear()
    rows = []
    eligible_count = 0
    processed_posts = 0
    root = None
    cutoff = datetime.fromisoformat(CUTOFF.replace("Z", "+00:00"))
    # Streaming parse keeps memory use bounded for the full Posts.xml file.
    for event, element in ET.iterparse(args.posts_xml, events=("start", "end")):
        if event == "start" and root is None:
            root = element
            continue
        if event == "start":
            continue
        if element.tag != "row":
            continue
        processed_posts += 1
        if processed_posts % 1000 == 0 and root is not None:
            root.clear()
        a = element.attrib
        if a.get("PostTypeId") not in {"1", "2"}:
            element.clear()
            continue
        created = a.get("CreationDate", "")
        try:
            created_dt = parse_utc(created)
        except ValueError:
            element.clear()
            continue
        # Include only rows whose latest edit is also no later than cutoff.
        edited = a.get("LastEditDate") or created
        try:
            edited_dt = parse_utc(edited)
        except ValueError:
            element.clear()
            continue
        if (created_dt > cutoff or edited_dt > cutoff or a.get("OwnerUserId") is None
                or a.get("OwnerUserId") in bot_ids):
            element.clear()
            continue
        text = clean_body(a.get("Body", ""))
        words = len(text.split())
        if not args.min_words <= words <= args.max_words or "[EMAIL]" in text:
            element.clear()
            if root is not None:
                root.clear()
            continue
        # URLs remain in source attribution, but are removed from the body to
        # reduce accidental personal links and external quoted material.
        text = re.sub(r"https?://\S+", "[URL]", text)
        post_id = a.get("Id")
        post_type = "question" if a.get("PostTypeId") == "1" else "answer"
        owner_id = a.get("OwnerUserId")
        if created_dt < datetime(2011, 4, 8, tzinfo=timezone.utc):
            license_name = "CC BY-SA 2.5"
        elif created_dt < datetime(2018, 5, 2, tzinfo=timezone.utc):
            license_name = "CC BY-SA 3.0"
        else:
            license_name = "CC BY-SA 4.0"
        row = {
            "text_id": f"{site}:{post_id}", "text": text, "label": 0,
            "source": site, "domain": "social_qa",
            "source_id": post_id, "group_id": f"{site}:{a.get('ParentId') or post_id}",
            "post_type": post_type, "created_at": created, "last_edit_at": edited,
            "title": a.get("Title", ""), "owner_user_id": owner_id, "score": int(a.get("Score", "0")),
            "license": license_name,
            "attribution": f"Stack Exchange contributor {owner_id}, post {post_id}",
            "license_url": f"https://{site}/help/licensing",
            "author_url": f"https://{site}/users/{owner_id}",
            "canonical_url": f"https://{site}/{'questions' if post_type == 'question' else 'a'}/{post_id}",
            "raw_sha256": sha((a.get("Body", "")).encode("utf-8")),
            "clean_sha256": sha(text.encode("utf-8")),
            "extraction_method": "BeautifulSoup HTML text; code/pre removed; contact email excluded; URLs redacted",
        }
        eligible_count += 1
        if len(rows) < args.max_rows:
            rows.append(row)
        else:
            replacement = rng.randrange(eligible_count)
            if replacement < args.max_rows:
                rows[replacement] = row
        element.clear()
        if root is not None:
            root.clear()

    if not rows:
        raise SystemExit("No rows met the filters; check dump, site, and word limits")
    out = args.output / "human_eval.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd")
    manifest = {
        "source": f"official Stack Exchange historical data dump; {site} Posts.xml",
        "source_url": "https://meta.stackexchange.com/questions/224873/all-stack-exchange-data-dump-releases",
        "source_file": args.posts_xml.name,
        "source_file_sha256": file_sha(args.posts_xml),
        "dump_snapshot": args.snapshot,
        "cutoff": CUTOFF,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "eligible_rows_seen": eligible_count, "max_rows": args.max_rows, "sample_seed": args.seed,
        "rights": "CC BY-SA; contribution license/version depends on date. Attribution and share-alike apply. Evaluation-only local research slice; not cleared for commercial training.",
        "rights_url": f"https://{site}/help/licensing",
        "human_label_caveat": "Creation and edit dates precede cutoff, but dumps do not prove human authorship; automated, copied, or assisted posts may remain.",
        "filters": {"post_types": ["question", "answer"], "owned_posts_only": True,
                    "creation_and_last_edit_before_cutoff": True, "email_containing_rows_excluded": True,
                    "obvious_bot_names_excluded": bool(args.users_xml),
                    "code_removed": True, "min_words": args.min_words, "max_words": args.max_words},
        "output_sha256": sha(out.read_bytes()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
