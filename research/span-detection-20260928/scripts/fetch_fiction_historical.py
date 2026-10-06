"""Fetch a bounded, author-diverse historical fiction pilot from Standard Ebooks.

The exact Standard Ebooks Git commits were verified to predate 2023. Text is
stored outside the repository; this script refuses to overwrite an existing
records.jsonl. Publication years refer to the original works, while source
revision dates refer to the public-domain ebook text snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


CUTOFF = "2022-12-31T23:59:59Z"
STANDARD_EBOOKS = "https://standardebooks.org"
GITHUB_API = "https://api.github.com/repos/standardebooks"

# Selected for one work per author, distinct from the ten works/authors already
# used in standard_ebooks_v1. Commit SHAs were verified through GitHub's public
# commits endpoint with `until=2022-12-31T23:59:59Z`.
WORKS = [
    {"repo": "f-scott-fitzgerald_the-great-gatsby", "commit": "122f83b8a0f29faa0513ec9ea4a75b34370375ca", "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "author_id": "f-scott-fitzgerald", "year": 1925, "genre": "modernist literary fiction"},
    {"repo": "charlotte-perkins-gilman_herland", "commit": "06edc0da0ec27c8ee40fd3ba391aec2a7f059679", "title": "Herland", "author": "Charlotte Perkins Gilman", "author_id": "charlotte-perkins-gilman", "year": 1915, "genre": "utopian speculative fiction"},
    {"repo": "stephen-crane_the-red-badge-of-courage", "commit": "715709b9c944657195738866f575e3fe2343c954", "title": "The Red Badge of Courage", "author": "Stephen Crane", "author_id": "stephen-crane", "year": 1895, "genre": "war fiction"},
    {"repo": "upton-sinclair_the-jungle", "commit": "32695122feed71de0991195f4cef7ab8983f0325", "title": "The Jungle", "author": "Upton Sinclair", "author_id": "upton-sinclair", "year": 1906, "genre": "social realist fiction"},
    {"repo": "g-k-chesterton_the-man-who-was-thursday", "commit": "7525180d1ef19be5578139f148273b144f94ad68", "title": "The Man Who Was Thursday", "author": "G. K. Chesterton", "author_id": "g-k-chesterton", "year": 1908, "genre": "fantasy mystery / philosophical thriller"},
    {"repo": "james-weldon-johnson_the-autobiography-of-an-ex-colored-man", "commit": "fe423d6aca55fa0889322e6212fb322b867335a6", "title": "The Autobiography of an Ex-Colored Man", "author": "James Weldon Johnson", "author_id": "james-weldon-johnson", "year": 1912, "genre": "African American literary fiction"},
    {"repo": "l-frank-baum_the-wonderful-wizard-of-oz", "commit": "3a49758023a73418d5dee1185fafaa3870ae6cec", "title": "The Wonderful Wizard of Oz", "author": "L. Frank Baum", "author_id": "l-frank-baum", "year": 1900, "genre": "children's fantasy"},
    {"repo": "mark-twain_a-connecticut-yankee-in-king-arthurs-court", "commit": "8c3251d53a4d75358cdf39832a501c8370a56dff", "title": "A Connecticut Yankee in King Arthur's Court", "author": "Mark Twain", "author_id": "mark-twain", "year": 1889, "genre": "satire / time-slip fiction"},
    {"repo": "mary-shelley_frankenstein", "commit": "21261379058381567bdd473b0368c440338ee279", "title": "Frankenstein; or, The Modern Prometheus", "author": "Mary Shelley", "author_id": "mary-shelley", "year": 1818, "genre": "gothic / speculative fiction"},
    {"repo": "virginia-woolf_mrs-dalloway", "commit": "0de73d9a62b20d33803b86ef5b4c548c5bf9d5a9", "title": "Mrs Dalloway", "author": "Virginia Woolf", "author_id": "virginia-woolf", "year": 1925, "genre": "modernist literary fiction"},
    {"repo": "edith-wharton_the-age-of-innocence", "commit": "5638e11601f2bd683e6ce04e1d77eebef7eead98", "title": "The Age of Innocence", "author": "Edith Wharton", "author_id": "edith-wharton", "year": 1920, "genre": "literary / social realism"},
    {"repo": "willa-cather_death-comes-for-the-archbishop", "commit": "dd92c6b3fd2174031015d937cbc536bc51784309", "title": "Death Comes for the Archbishop", "author": "Willa Cather", "author_id": "willa-cather", "year": 1927, "genre": "historical literary fiction"},
]

EXCLUDED_FILE_PARTS = (
    "colophon", "imprint", "titlepage", "uncopyright", "halftitle", "dedication",
    "foreword", "introduction", "preface", "afterword", "endnote", "footnote",
    "toc", "table-of-contents", "copyright", "back-cover", "loi",
)
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def own_text(node: Tag) -> str:
    pieces: list[str] = []
    def visit(child: Any) -> None:
        if isinstance(child, NavigableString):
            pieces.append(str(child))
        elif isinstance(child, Tag):
            if child.name in BLOCK_TAGS:
                return
            for sub in child.children:
                visit(sub)
    for child in node.children:
        visit(child)
    return re.sub(r"\s+", " ", " ".join(pieces)).strip()


def parse_book(archive_bytes: bytes, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tar = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz")
    members = {m.name: m for m in tar.getmembers() if m.isfile()}
    opf_name = next(n for n in members if n.endswith("/src/epub/content.opf"))
    prefix = opf_name.rsplit("/src/epub/content.opf", 1)[0]
    opf = BeautifulSoup(tar.extractfile(members[opf_name]).read(), "xml")
    creator = opf.find("dc:creator")
    title = opf.find("dc:title")
    se_date = opf.find("dc:date")
    rights = opf.find("dc:rights")
    source_urls = [x.get_text(" ", strip=True) for x in opf.find_all("dc:source")]
    subjects = [x.get_text(" ", strip=True) for x in opf.find_all("dc:subject")]
    if creator is None or creator.get_text(strip=True) != spec["author"]:
        raise ValueError(f"author mismatch in {spec['repo']}: {creator}")
    if title is None:
        raise ValueError(f"no title in {spec['repo']}")
    # Respect EPUB spine order; do not alphabetically scramble chapter files.
    id_to_href = {x.get("id"): x.get("href") for x in opf.find_all("item") if x.get("id") and x.get("href")}
    text_parts = []
    used_files = []
    for itemref in opf.find_all("itemref"):
        href = id_to_href.get(itemref.get("idref"), "")
        if not href.endswith(".xhtml") or "/text/" not in f"/{href}":
            continue
        if any(token in Path(href).name.lower() for token in EXCLUDED_FILE_PARTS):
            continue
        member_name = f"{prefix}/src/epub/{unquote(href)}"
        member = members.get(member_name)
        if member is None:
            # Some packages encode relative href paths with a leading ./.
            member_name = member_name.replace("/text/./", "/text/")
            member = members.get(member_name)
        if member is None:
            continue
        doc = BeautifulSoup(tar.extractfile(member).read(), "xml")
        body = doc.find("body")
        if body is None:
            continue
        for node in body.select("nav, aside, header, footer, script, style"):
            node.decompose()
        for block in body.find_all(list(BLOCK_TAGS)):
            text = own_text(block)
            if text:
                text_parts.append(text)
        used_files.append(Path(href).name)
    text = "\n\n".join(text_parts).strip()
    if len(text.split()) < 5000:
        raise ValueError(f"unexpectedly short whole-book text for {spec['repo']}: {len(text.split())}")
    metadata = {
        "ebook_title": title.get_text(" ", strip=True),
        "ebook_author": creator.get_text(" ", strip=True),
        "ebook_edition_date": se_date.get_text(" ", strip=True) if se_date else None,
        "ebook_rights_statement": rights.get_text(" ", strip=True) if rights else None,
        "original_edition_sources": source_urls,
        "subjects": subjects,
        "included_text_files": used_files,
    }
    return text, metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("/mnt/f/pangram-at-home/data/fiction_candidates_v1/historical"))
    ap.add_argument("--existing-manifest", type=Path, default=Path("/mnt/f/pangram-at-home/data/standard_ebooks_v1/manifest.json"))
    args = ap.parse_args()
    records_path = args.outdir / "records.jsonl"
    if records_path.exists():
        ap.error(f"refusing to overwrite existing corpus: {records_path}; choose a fresh --outdir")
    if args.existing_manifest.exists():
        old = json.loads(args.existing_manifest.read_text())
        old_books = {x.get("book") for x in old.get("books", [])}
        old_authors = {b.split("_", 1)[0] for b in old_books if b}
        overlap = [w["repo"] for w in WORKS if w["repo"] in old_books or w["author_id"] in old_authors]
        if overlap:
            ap.error(f"selected work/author overlaps existing Standard Ebooks pool: {overlap}")

    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "pangram-at-home-historical-fiction-research/1.0"
    records = []
    for spec in WORKS:
        repo, commit = spec["repo"], spec["commit"]
        archive_path = raw_dir / f"{repo}-{commit[:12]}.tar.gz"
        if archive_path.exists():
            archive_bytes = archive_path.read_bytes()
        else:
            url = f"{GITHUB_API}/{repo}/tarball/{commit}"
            response = session.get(url, timeout=120)
            response.raise_for_status()
            archive_bytes = response.content
            archive_path.write_bytes(archive_bytes)
        # Use the commit API once to verify the pinned source revision metadata.
        api = f"{GITHUB_API}/{repo}/commits/{commit}"
        commit_response = session.get(api, timeout=30)
        commit_response.raise_for_status()
        commit_info = commit_response.json()
        commit_date = commit_info["commit"]["committer"]["date"]
        if commit_date > CUTOFF:
            raise ValueError(f"source revision is after cutoff: {repo} {commit_date}")
        text, book_meta = parse_book(archive_bytes, spec)
        record = {
            "text": text,
            "document_id": f"standardebooks:{repo}",
            "source": "standard_ebooks_github",
            "source_id": repo,
            "work_id": repo,
            "title": spec["title"],
            "author_ids": [spec["author_id"]],
            "author_names": [spec["author"]],
            "publication_date": f"{spec['year']}",
            "publication_date_basis": "original first-publication year for the historical book; full text retained as a single work",
            "genre": spec["genre"],
            "source_revision": commit,
            "source_revision_date": commit_date,
            "human_origin_status": "historical_human_work_pre_2023_versioned_source",
            "human_origin_evidence": "Known historical book by the named author; Standard Ebooks repository snapshot is pinned to a commit dated before 2023. The source dates the version, not every historical composition/editing step.",
            "rights_status": "Standard Ebooks states source texts are believed public domain in the United States; the ebook edition/editorial work is dedicated CC0. Verify jurisdiction separately.",
            "rights_url": "https://standardebooks.org/about/standard-ebooks-and-the-public-domain",
            "canonical_url": f"{STANDARD_EBOOKS}/ebooks/{repo.replace('_', '/')}",
            "source_url": f"https://github.com/standardebooks/{repo}/tree/{commit}",
            "raw_archive_path": str(archive_path.relative_to(args.outdir)),
            "source_sha256": sha(archive_bytes),
            "clean_sha256": sha(text.encode("utf-8")),
            "group_id": f"historical-fiction:{spec['author_id']}:{repo}",
            "word_count": len(text.split()),
            "source_metadata": book_meta,
        }
        records.append(record)
        print(f"{spec['title']}: {record['word_count']} words (one work group)", flush=True)
        time.sleep(0.1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": "fiction_candidates_v1_historical",
        "record_count": len(records),
        "independent_work_count": len({r["work_id"] for r in records}),
        "independent_author_count": len({a for r in records for a in r["author_ids"]}),
        "records_path": "records.jsonl",
        "raw_archive_directory": "raw/",
        "source_cutoff": CUTOFF,
        "total_words": sum(r["word_count"] for r in records),
        "counts_by_original_publication_decade": {},
        "counts_by_genre": {},
        "rights_note": "Research candidate; Standard Ebooks says it believes source texts are US public domain and dedicates its ebook edition contributions to CC0. Jurisdiction-specific rights still need checking; do not publish raw corpus until the release plan is reviewed.",
        "provenance_note": "All records are one whole work each, from a pinned pre-2023 Standard Ebooks GitHub snapshot. The ten previously used Standard Ebooks works and their authors were excluded. Historical publication date and public byline support human-origin attribution but do not certify every later transcription/editing step.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "works": [{k: r[k] for k in ("document_id", "title", "author_ids", "publication_date", "genre", "source_revision", "source_revision_date", "word_count", "source_sha256", "clean_sha256")} for r in records],
    }
    decade_counts: dict[str, int] = {}
    genre_counts: dict[str, int] = {}
    for r in records:
        decade = str(int(r["publication_date"][:4]) // 10 * 10)
        decade_counts[decade] = decade_counts.get(decade, 0) + 1
        genre_counts[r["genre"]] = genre_counts.get(r["genre"], 0) + 1
    manifest["counts_by_original_publication_decade"] = decade_counts
    manifest["counts_by_genre"] = genre_counts
    manifest["records_sha256"] = sha(records_path.read_bytes())
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} full works / {manifest['total_words']} words to {records_path}")


if __name__ == "__main__":
    main()
