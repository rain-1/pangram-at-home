"""Fetch pre-2023 Standard Ebooks snapshots and extract fiction prose."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup


BOOKS = [
    "james-joyce_dubliners",
    "robert-louis-stevenson_the-strange-case-of-dr-jekyll-and-mr-hyde",
    "arthur-conan-doyle_the-lost-world",
    "bram-stoker_dracula",
    "edgar-allan-poe_the-narrative-of-arthur-gordon-pym-of-nantucket",
    "edgar-rice-burroughs_a-princess-of-mars",
    "henry-james_the-turn-of-the-screw",
    "h-g-wells_the-time-machine",
    "jack-london_the-call-of-the-wild",
    "joseph-conrad_heart-of-darkness",
]
CUTOFF = "2022-12-31T23:59:59Z"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/data/standard_ebooks_v1"))
    args = parser.parse_args()
    root = args.output
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "pangram-at-home-research/0.1"
    rows = []
    manifest = {"source": "Standard Ebooks official GitHub", "cutoff": CUTOFF,
                "rights": "Public domain in US per Standard Ebooks; check other jurisdictions",
                "rights_url": "https://standardebooks.org/about/standard-ebooks-and-the-public-domain",
                "books": []}
    for book in BOOKS:
        api = f"https://api.github.com/repos/standardebooks/{book}/commits"
        response = session.get(api, params={"until": CUTOFF, "per_page": 1}, timeout=30)
        response.raise_for_status()
        commits = response.json()
        if not commits:
            raise ValueError(f"No pre-2023 snapshot: {book}")
        commit = commits[0]["sha"]
        date = commits[0]["commit"]["committer"]["date"]
        archive_url = f"https://api.github.com/repos/standardebooks/{book}/tarball/{commit}"
        archive_path = raw / f"{book}-{commit[:12]}.tar.gz"
        if archive_path.exists():
            archive_bytes = archive_path.read_bytes()
        else:
            archive = session.get(archive_url, timeout=120)
            archive.raise_for_status()
            archive_bytes = archive.content
            archive_path.write_bytes(archive_bytes)
        book_rows = 0
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile() or "/src/epub/text/" not in member.name or not member.name.endswith(".xhtml"):
                    continue
                chapter = member.name.rsplit("/", 1)[-1]
                if any(token in chapter for token in ("colophon", "imprint", "titlepage", "uncopyright", "halftitle", "dedication")):
                    continue
                contents = tar.extractfile(member).read()
                soup = BeautifulSoup(contents, "html.parser")
                body = soup.find("body")
                if body is None:
                    continue
                paragraphs = []
                for p in body.find_all("p"):
                    if p.find_parent(["nav", "aside", "header"]):
                        continue
                    paragraph = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                    if len(paragraph.split()) >= 15:
                        paragraphs.append(paragraph)
                # Keep nearby paragraphs in one sample; all chunks from a book share group_id.
                chunk = []
                count = 0
                for paragraph in paragraphs + [""]:
                    words = len(paragraph.split())
                    if chunk and (count + words > 380 or not paragraph):
                        text = "\n\n".join(chunk)
                        if count >= 100:
                            number = book_rows
                            rows.append({"text_id": f"standardebooks:{book}:{number}", "text": text,
                                         "label": 0, "source": "standard_ebooks", "domain": "creative_writing",
                                         "source_id": book, "group_id": f"standardebooks:{book}",
                                         "chapter": chapter, "snapshot_commit": commit,
                                         "snapshot_date": date, "raw_sha256": sha(contents),
                                         "clean_sha256": sha(text.encode()),
                                         "canonical_url": f"https://github.com/standardebooks/{book}/tree/{commit}"})
                            book_rows += 1
                        chunk, count = [], 0
                    if paragraph:
                        chunk.append(paragraph)
                        count += words
        manifest["books"].append({"book": book, "commit": commit, "snapshot_date": date,
                                  "archive_sha256": sha(archive_bytes), "rows": book_rows})
        print(book, book_rows, flush=True)
    out = root / "human.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd")
    manifest["rows"] = len(rows)
    manifest["output_sha256"] = sha(out.read_bytes())
    manifest["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
