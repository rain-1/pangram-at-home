"""Fetch a dated, item-licensed PMC paper pilot through NCBI E-utilities.

This fetches bounded batches; it does not mirror the multi-million-article PMC
collection. Raw XML and parsed documents are stored outside Git.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from lxml import etree

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "pangram-at-home/0.1 (research corpus pilot)"
YEARS = list(range(2011, 2023))
CC_BY = re.compile(r"creativecommons\.org/licenses/by/4\.0")
CC_ZERO = re.compile(r"creativecommons\.org/publicdomain/zero/1\.0")


def get(session: requests.Session, path: str, params: dict) -> bytes:
    for attempt in range(6):
        response = session.get(f"{BASE}/{path}", params=params, timeout=90)
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(min(2 ** attempt, 30))
            continue
        response.raise_for_status()
        time.sleep(0.38)  # Below NCBI's unauthenticated three requests/second limit.
        return response.content
    raise RuntimeError(f"NCBI request failed after retries: {path}")


def clean(value: str) -> str:
    return " ".join(value.split())


def text(element) -> str:
    return clean(" ".join(element.itertext())) if element is not None else ""


def first(element, path: str) -> str:
    return text(element.find(path))


def article_date(meta) -> str:
    for path in ["./pub-date[@pub-type='epub']", "./pub-date[@publication-format='electronic']", "./pub-date", "./history/date[@date-type='accepted']"]:
        node = meta.find(path)
        if node is None:
            continue
        year = first(node, "./year")
        if not year.isdigit():
            continue
        month = first(node, "./month")
        day = first(node, "./day")
        return f"{int(year):04d}-{int(month) if month.isdigit() else 1:02d}-{int(day) if day.isdigit() else 1:02d}"
    return ""


def parse_article(article) -> dict | None:
    meta = article.find("./front/article-meta")
    if meta is None or article.get("article-type") == "preprint":
        return None
    license_node = meta.find("./permissions/license")
    if license_node is None:
        return None
    license_url = license_node.get("{http://www.w3.org/1999/xlink}href", "")
    if not license_url:
        references = license_node.xpath(".//*[local-name()='license_ref']")
        if references:
            license_url = text(references[0])
    if CC_BY.search(license_url):
        license_name = "CC BY 4.0"
    elif CC_ZERO.search(license_url):
        license_name = "CC0 1.0"
    else:
        return None
    date = article_date(meta)
    if not date or date > "2022-12-31":
        return None
    ids = {x.get("pub-id-type"): text(x) for x in meta.findall("./article-id")}
    pmcid = ids.get("pmcid", "") or ids.get("pmc", "")
    if not pmcid:
        return None
    abstract = text(meta.find("./abstract"))
    body = article.find("./body")
    paragraphs = []
    if body is not None:
        for node in body.iter("p"):
            if any(parent.tag in {"fig", "table-wrap", "supplementary-material"} for parent in node.iterancestors()):
                continue
            paragraph = text(node)
            if len(paragraph) >= 80:
                paragraphs.append(paragraph)
    full_text = "\n\n".join(paragraphs)
    if len(abstract) < 300 or len(full_text) < 1000:
        return None
    title = text(meta.find("./title-group/article-title"))
    journal = first(article, "./front/journal-meta/journal-title-group/journal-title")
    authors = []
    for contrib in meta.findall("./contrib-group/contrib[@contrib-type='author']"):
        surname = first(contrib, "./name/surname")
        given = first(contrib, "./name/given-names")
        if surname:
            authors.append(clean(f"{given} {surname}"))
    return {
        "source": "pmc_oa",
        "source_id": f"PMC{pmcid.lstrip('PMC')}",
        "canonical_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid.lstrip('PMC')}/",
        "doi": ids.get("doi", ""),
        "title": title,
        "authors": authors,
        "journal": journal,
        "publication_date": date,
        "license": license_name,
        "license_url": license_url,
        "abstract": abstract,
        "body": full_text,
        "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
        "body_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--per-year", type=int, default=100)
    args = parser.parse_args()
    if args.per_year < 1 or args.per_year > 500:
        raise SystemExit("--per-year must be 1..500")
    out = args.root / "data" / "pmc_pilot_v1"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    all_docs = []
    for year in YEARS:
        query = (
            '(cc by license[filter] OR cc0 license[filter]) '
            f'AND {year}/01/01:{year}/12/31[Publication Date] '
            'NOT preprint[All Fields]'
        )
        seed_params = {"db": "pmc", "term": query, "retmode": "json", "retmax": 0}
        result = json.loads(get(session, "esearch.fcgi", seed_params))["esearchresult"]
        count = int(result["count"])
        count_to_fetch = min(count, max(args.per_year * 3, 150))
        # ESearch exposes only the first 10k records for this route. Select a
        # deterministic offset inside that window and save the IDs.
        max_start = max(0, min(count - count_to_fetch, 10000 - count_to_fetch))
        start = (year * 7919) % (max_start + 1) if max_start else 0
        ids = json.loads(get(session, "esearch.fcgi", {
            **seed_params, "retstart": start, "retmax": count_to_fetch,
        }))["esearchresult"]["idlist"]
        docs = []
        doc_ids = set()
        for offset in range(0, len(ids), 40):
            chunk = ids[offset:offset + 40]
            chunk_hash = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:12]
            raw_path = raw / f"{year}_{offset:04d}_{chunk_hash}.xml.gz"
            if raw_path.exists():
                payload = gzip.decompress(raw_path.read_bytes())
            else:
                payload = get(session, "efetch.fcgi", {
                    "db": "pmc", "id": ",".join(chunk), "retmode": "xml",
                })
                raw_path.write_bytes(gzip.compress(payload))
            root = etree.fromstring(payload)
            for article in root.findall(".//article"):
                doc = parse_article(article)
                if doc is not None and doc["source_id"] not in doc_ids:
                    docs.append(doc)
                    doc_ids.add(doc["source_id"])
            if len(docs) >= args.per_year:
                break
        all_docs.extend(docs[:args.per_year])
        print(f"{year}: selected {min(len(docs), args.per_year)} / {count} search matches", flush=True)
    path = out / "documents.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for doc in all_docs:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")
    manifest = {
        "source": "NCBI PMC E-utilities",
        "query_years": YEARS,
        "per_year_target": args.per_year,
        "selected": len(all_docs),
        "rights": "per-article CC BY 4.0 or CC0 1.0 verified from JATS XML",
        "documents_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_xml_dir": str(raw),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
