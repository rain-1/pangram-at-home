"""Fetch dated Federal Reserve Beige Book prose for professional-domain auditing.

The Beige Book is a Board of Governors publication. The source pages are archived
by release year; this script keeps only releases dated no later than 2022-12-31.
Raw HTML and extracted passages are written outside Git by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup


BASE = "https://www.federalreserve.gov"
YEARS = range(2019, 2023)
CUTOFF = "2022-12-31"
RIGHTS_URL = f"{BASE}/disclaimer.htm"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/data/federal_reserve_beige_book_v1"))
    args = parser.parse_args()
    root = args.output
    raw_dir = root / "raw_html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "pangram-at-home-research/0.1 (dated public document collection)"
    records = []
    report_meta = []

    for year in YEARS:
        archive_url = f"{BASE}/monetarypolicy/beigebook{year}.htm"
        response = session.get(archive_url, timeout=45)
        response.raise_for_status()
        archive_bytes = response.content
        (raw_dir / f"archive_{year}.html").write_bytes(archive_bytes)
        soup = BeautifulSoup(archive_bytes, "html.parser")
        release_urls = []
        for anchor in soup.find_all("a", href=True):
            href = urljoin(archive_url, anchor["href"])
            if re.search(rf"/monetarypolicy/beigebook\d{{6,8}}\.htm$", href):
                if href not in release_urls:
                    release_urls.append(href)

        for report_url in release_urls:
            report_response = session.get(report_url, timeout=45)
            report_response.raise_for_status()
            html = report_response.content
            page = BeautifulSoup(html, "html.parser")
            title = clean(page.title.get_text(" ", strip=True)) if page.title else ""
            date_match = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})$", title)
            if not date_match:
                raise ValueError(f"Cannot read release date from title {title!r} at {report_url}")
            date = datetime.strptime(date_match.group(1), "%B %d, %Y").date().isoformat()
            if date > CUTOFF:
                continue
            path = raw_dir / f"beigebook_{date}.html"
            path.write_bytes(html)

            # Keep the report narrative. Drop navigation, headers, footers, tables,
            # and links' labels where the same text is duplicated in page chrome.
            for node in page.select("header, footer, nav, script, style, table, .social, .printOnly"):
                node.decompose()
            main = page.find("main") or page.find(id="article") or page.find(id="content") or page.body
            if main is None:
                continue
            paragraphs = []
            for node in main.find_all(["p", "li"]):
                paragraph = clean(node.get_text(" ", strip=True))
                words = paragraph.split()
                if len(words) >= 20 and not paragraph.lower().startswith(("for media", "contact", "last update")):
                    paragraphs.append(paragraph)

            chunks: list[str] = []
            chunk_words = 0
            report_rows = 0
            for paragraph in paragraphs + [""]:
                size = len(paragraph.split())
                if chunks and (chunk_words + size > 360 or not paragraph):
                    text = "\n\n".join(chunks)
                    if chunk_words >= 90:
                        records.append({
                            "text_id": f"beigebook:{date}:{report_rows}",
                            "text": text,
                            "label": 0,
                            "source": "federal_reserve_beige_book",
                            "domain": "professional_finance",
                            "source_id": f"beigebook:{date}",
                            "group_id": f"federal_reserve_beige_book:{date}",
                            "original_publication_date": date,
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "canonical_url": report_url,
                            "source_version": sha(html),
                            "license": "Board website public domain unless otherwise indicated; cite Board",
                            "rights_url": RIGHTS_URL,
                            "attribution": "Board of Governors of the Federal Reserve System",
                            "raw_sha256": sha(html),
                            "clean_sha256": sha(text.encode("utf-8")),
                            "extraction_method": "HTML paragraph extraction; navigation/tables removed",
                        })
                        report_rows += 1
                    chunks, chunk_words = [], 0
                if paragraph:
                    chunks.append(paragraph)
                    chunk_words += size
            report_meta.append({"date": date, "url": report_url, "html_sha256": sha(html), "passages": report_rows})
            print(f"{date}: {report_rows} passages", flush=True)

    out = root / "human.parquet"
    pq.write_table(pa.Table.from_pylist(records), out, compression="zstd")
    manifest = {
        "source": "Federal Reserve Board Beige Book archives",
        "category": "professional_finance",
        "cutoff": CUTOFF,
        "rights": "Board disclaimer states that, unless otherwise indicated, information on the Board website is public domain and may be copied/distributed without permission; cite the Board. It also cautions that use may implicate privately owned rights. This collection excludes tables/graphics and is suitable as a candidate source pending item-level audit.",
        "rights_url": RIGHTS_URL,
        "archive_years": list(YEARS),
        "reports": report_meta,
        "rows": len(records),
        "output_sha256": sha(out.read_bytes()),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(records)} passages to {out}")


if __name__ == "__main__":
    main()
