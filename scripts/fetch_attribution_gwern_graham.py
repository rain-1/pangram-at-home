"""Collect bounded official Gwern and Paul Graham essays for private attribution.

Use whole essays as groups. This collector never calls model APIs and never
publishes text in Git. Dates are provenance evidence, not absolute proof of
unaided human authorship.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString
import pyarrow.parquet as pq
import requests


GWERN_INDEX = "https://gwern.net/index"
GRAHAM_INDEX = "https://www.paulgraham.com/articles.html"
USER_AGENT = "pangram-at-home-research/0.1 (bounded private authorship audit)"
DATE_LIMIT = "2022-12-31"
MONTHS = "January February March April May June July August September October November December".split()
GWERN_TEXT_QUARANTINE = {
    "https://gwern.net/cyoa": "AI Dungeon model-output examples may remain in prose extraction",
    "https://gwern.net/gpt-3-nonfiction": "GPT-3 nonfiction demonstration includes model output",
    "https://gwern.net/gpt-2-preference-learning": "text-generation demonstration may include model output",
    "https://gwern.net/rnn-metadata": "RNN author-style demonstration may include generated text",
    "https://gwern.net/lorem": "placeholder and word-list content is not author prose",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str) -> bytes:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError(f"Page unexpectedly large: {url}")
    return response.content


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, width: int = 24) -> set[bytes]:
    words = re.findall(r"\b\w+\b", text.casefold())
    return {hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words) - width + 1))}


def parent_fingerprints(root: Path) -> tuple[set[str], set[bytes]]:
    hashes, grams = set(), set()
    for split in ("train", "val", "test"):
        path = root / "diverse_pyramid_v1" / f"{split}_full.parquet"
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            hashes.add(sha(text.encode()))
            grams.update(shingles(text))
    return hashes, grams


def gwern_links(html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for node in soup.select("#markdownBody a[href]"):
        url = urljoin(GWERN_INDEX, node["href"]).split("#", 1)[0]
        parts = urlparse(url)
        if parts.netloc != "gwern.net" or not parts.path or parts.path in ("/", "/index"):
            continue
        if parts.path.startswith(("/doc/", "/static/", "/tag/", "/blog/", "/fiction/")):
            continue
        if re.search(r"\.(?:pdf|png|jpg|html|txt|csv|zip|gz)$", parts.path):
            continue
        links.add(url)
    return sorted(links)


def gwern_meta(url: str) -> dict | None:
    try:
        raw = get(url + ".md")
    except (requests.RequestException, ValueError):
        return None
    head = raw.decode("utf-8", errors="replace").split("\n...\n", 1)[0][:10000]

    def field(name: str) -> str | None:
        match = re.search(rf"(?m)^{name}:\s*(.+)$", head)
        return match.group(1).strip().strip('"\'') if match else None

    created, modified = field("created"), field("modified")
    if not created or not modified or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", created) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", modified):
        return None
    return {"url": url, "created": created, "modified": modified,
            "title": field("title"), "author": field("author"), "status": field("status"),
            "markdown_sha256": sha(raw)}


def gwern_body(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("#markdownBody")
    if body is None:
        return ""
    for node in body.select("blockquote, pre, code, table, figure, aside, nav, noscript, "
                            ".footnotes, .sidenote, .admonition, .annotation, "
                            "#backlinks-section, #similars-section, #link-bibliography-section, "
                            "#external-links, #see-also, #references"):
        node.decompose()
    paragraphs = []
    for node in body.find_all("p"):
        if node.find_parent(["blockquote", "pre", "table", "figure", "aside", "nav"]):
            continue
        paragraph = clean(node.get_text(" ", strip=True))
        if len(paragraph.split()) >= 8:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def graham_links(html: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    seen, results = set(), []
    for node in soup.find_all("a", href=True):
        url = urljoin(GRAHAM_INDEX, node["href"])
        parts = urlparse(url)
        if parts.netloc not in ("paulgraham.com", "www.paulgraham.com") or not parts.path.endswith(".html"):
            continue
        if url in seen or parts.path in ("/articles.html", "/index.html"):
            continue
        title = clean(node.get_text(" ", strip=True))
        if not title:
            continue
        seen.add(url)
        results.append((url, title))
    return results


def graham_body(html: bytes) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")
    fonts = [node for node in soup.find_all("font") if node.get("size") == "2"]
    if not fonts:
        return None, ""
    font = max(fonts, key=lambda node: len(node.get_text(" ", strip=True)))
    fragment = BeautifulSoup(str(font), "html.parser")
    for node in fragment.select("blockquote, pre, code, table, script, style"):
        node.decompose()
    for br in fragment.find_all("br"):
        br.replace_with(NavigableString("\n"))
    raw = fragment.get_text("", strip=False)
    sections = [clean(part) for part in re.split(r"\n\s*\n", raw) if clean(part)]
    if not sections:
        return None, ""
    date_text = sections[0]
    match = re.match(rf"^({'|'.join(MONTHS)})\s+(?:(\d{{1,2}}),?\s+)?(\d{{4}})$", date_text)
    if not match:
        return None, ""
    month, day, year = match.group(1), match.group(2), match.group(3)
    date = datetime.strptime(f"{month} {day or '1'} {year}", "%B %d %Y").date().isoformat()
    prose = []
    for section in sections[1:]:
        if section in {"Notes", "Thanks", "Thanks to", "Note", "Footnotes"} or section.startswith(("Notes [", "Thanks to ")):
            break
        if len(section.split()) >= 8:
            prose.append(section)
    return date, "\n\n".join(prose)


def graham_snapshot(url: str) -> tuple[bytes, str, str] | None:
    """Get the archived official page as it stood no later than 2022."""
    target = "https://web.archive.org/web/20221231id_/" + url
    try:
        response = requests.get(target, headers={"User-Agent": USER_AGENT}, timeout=25)
        response.raise_for_status()
        match = re.search(r"/web/(\d{14})id_/", response.url)
        if not match or match.group(1)[:8] > "20221231" or len(response.content) > 2_000_000:
            return None
        return response.content, response.url, match.group(1)[:8]
    except requests.RequestException:
        return None


def build(args) -> None:
    output = args.output
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    gwern_index = get(GWERN_INDEX)
    graham_index = get(GRAHAM_INDEX)
    gwern_urls = gwern_links(gwern_index)
    graham_urls = graham_links(graham_index)
    with ThreadPoolExecutor(max_workers=2) as pool:
        gwern_metadata = list(pool.map(gwern_meta, gwern_urls))
    metadata = [row for row in gwern_metadata if row]
    gwern_eligible = [row for row in metadata if row["created"] <= DATE_LIMIT and
                      row["modified"] <= DATE_LIMIT and
                      (not row["author"] or row["author"].casefold() == "gwern") and
                      row["status"] == "finished"]
    gwern_eligible.sort(key=lambda row: (row["created"] >= "2020-01-01", row["created"]), reverse=True)
    records = []
    gwern_failures = []
    for row in gwern_eligible:
        if sum(record["author_id"] == "gwern" for record in records) >= args.per_author:
            break
        try:
            html = get(row["url"])
            text = gwern_body(html)
        except (requests.RequestException, ValueError) as error:
            gwern_failures.append({"url": row["url"], "reason": type(error).__name__})
            continue
        if len(text.split()) < args.min_words:
            gwern_failures.append({"url": row["url"], "reason": "too_short_after_cleaning"})
            continue
        document_id = "gwern:" + row["url"].removeprefix("https://gwern.net/")
        quarantine_reason = GWERN_TEXT_QUARANTINE.get(row["url"])
        records.append({"text": text, "author_id": "gwern", "document_id": document_id,
                        "group_id": document_id, "title": row["title"],
                        "canonical_url": row["url"], "publication_date": row["created"],
                        "publication_date_precision": "day",
                        "source_modified_date": row["modified"],
                        "last_modified_date": row["modified"],
                        "attribution_eligible": quarantine_reason is None,
                        "strict_human_candidate": quarantine_reason is None,
                        "quarantine_reason": quarantine_reason,
                        "human_origin_status": "official_gwern_page_created_and_modified_before_2023; no independent human-only proof",
                        "byline_status": "official_gwern_site; explicit author field Gwern or site-level Gwern attribution",
                        "rights_status": "Gwern.net states CC0; third-party quotation and external material removed where structurally marked",
                        "rights_evidence_url": "https://gwern.net/about#license",
                        "source_sha256": sha(html), "source_markdown_sha256": row["markdown_sha256"],
                        "clean_sha256": sha(text.encode()), "source_revision": row["modified"],
                        "source": "gwern.net", "domain": "essay", "publication_period": "2020-2022" if row["created"] >= "2020-01-01" else "pre-2020",
                        "split": "unsplit_private_attribution"})

    graham_metadata = []
    for url, title in graham_urls:
        try:
            html = get(url)
            date, text = graham_body(html)
        except (requests.RequestException, ValueError):
            continue
        if not date or date > DATE_LIMIT or len(text.split()) < args.min_words:
            continue
        graham_metadata.append((date, url, title, html, text))
        if len(graham_metadata) >= args.max_graham_scan:
            break
    graham_metadata.sort(key=lambda item: (item[0] >= "2020-01-01", item[0]), reverse=True)
    selected_graham = graham_metadata[:args.per_author]
    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = list(pool.map(lambda item: graham_snapshot(item[1]), selected_graham))
    for (date, url, title, html, text), snapshot in zip(selected_graham, snapshots):
        snapshot_url = None
        snapshot_date = None
        current_sha = sha(html)
        if snapshot is not None:
            archived_html, archived_url, archived_date = snapshot
            archived_page_date, archived_text = graham_body(archived_html)
            if archived_page_date == date and len(archived_text.split()) >= args.min_words:
                html, text = archived_html, archived_text
                snapshot_url, snapshot_date = archived_url, archived_date
        document_id = "paul_graham:" + url.rsplit("/", 1)[-1].removesuffix(".html")
        records.append({"text": text, "author_id": "paul_graham", "document_id": document_id,
                        "group_id": document_id, "title": title,
                        "canonical_url": url, "publication_date": date,
                        "publication_date_precision": "month; day set to 01 for ISO compatibility",
                        "source_modified_date": None,
                        "last_modified_date": None,
                        "source_snapshot_url": snapshot_url, "source_snapshot_date": snapshot_date,
                        "attribution_eligible": True,
                        "strict_human_candidate": snapshot_date is not None,
                        "quarantine_reason": None,
                        "human_origin_status": ("official_page_archived_before_2023; no independent human-only proof"
                                                if snapshot_date else
                                                "official_byline_and_page_date_before_2023; current_html_revision_date_unavailable"),
                        "byline_status": "official_paul_graham_essay_index_and_site_biography",
                        "rights_status": "copyright retained; FAQ discourages mirroring; private research only",
                        "rights_evidence_url": "https://www.paulgraham.com/gfaq.html",
                        "source_sha256": sha(html), "source_current_html_sha256": current_sha,
                        "source_markdown_sha256": None,
                        "clean_sha256": sha(text.encode()), "source_revision": snapshot_date,
                        "source": "paulgraham.com", "domain": "essay",
                        "publication_period": "2020-2022" if date >= "2020-01-01" else "pre-2020",
                        "split": "unsplit_private_attribution"})

    if Counter(row["author_id"] for row in records) != {"gwern": args.per_author, "paul_graham": args.per_author}:
        raise RuntimeError(f"Could not collect target per author: {Counter(row['author_id'] for row in records)}")
    if len({row["document_id"] for row in records}) != len(records) or len({row["clean_sha256"] for row in records}) != len(records):
        raise RuntimeError("Duplicate attribution work or cleaned text")
    parent_hashes, parent_grams = parent_fingerprints(args.data_root)
    for row in records:
        row["detector_exact_text_overlap"] = row["clean_sha256"] in parent_hashes
        row["detector_24word_shingle_overlap_count"] = len(shingles(row["text"]) & parent_grams)

    output.mkdir(parents=True)
    path = output / "records.jsonl"
    with path.open("w") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"role": "private whole-essay writer attribution; never detector training",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_indexes": {GWERN_INDEX: sha(gwern_index), GRAHAM_INDEX: sha(graham_index)},
                "records_sha256": sha(path.read_bytes()), "records": len(records),
                "authors": dict(Counter(row["author_id"] for row in records)),
                "attribution_eligible": dict(Counter(row["author_id"] for row in records if row["attribution_eligible"])),
                "periods": dict(Counter(f"{row['author_id']}:{row['publication_period']}" for row in records)),
                "strict_human_candidates": dict(Counter(row["author_id"] for row in records if row["strict_human_candidate"])),
                "gwern_index_candidates": len(gwern_urls),
                "gwern_metadata_with_dates": len(metadata),
                "gwern_created_and_modified_pre2023_finished_eligible": len(gwern_eligible),
                "gwern_recent_or_revised_quarantine_count": sum(
                    bool(row["created"] > DATE_LIMIT or row["modified"] > DATE_LIMIT)
                    for row in metadata),
                "gwern_model_output_quarantined": [row["document_id"] for row in records
                                                   if row["author_id"] == "gwern" and row["quarantine_reason"]],
                "gwern_fetch_failures": gwern_failures,
                "graham_index_candidates": len(graham_urls),
                "graham_date_eligible_scanned": len(graham_metadata),
                "overlap_audit": "exact cleaned SHA-256 and normalized 24-word shingles against diverse train/val/test parent; per-record counts retained",
                "overlap_records": sum(bool(row["detector_exact_text_overlap"] or row["detector_24word_shingle_overlap_count"]) for row in records),
                "human_origin_caveat": "Original site dates and bylines support provenance; they do not prove absence of AI-assisted revisions or copied passages.",
                "rights": "Gwern official site says CC0; Graham asks readers to link rather than mirror. Keep all collected full text private and outside Git.",
                "selection": {"per_author": args.per_author, "min_words": args.min_words,
                              "preference": "2020-2022 first; older dated essays fill quota; post-2022 or later-revised Gwern pages excluded"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("records", "authors", "periods", "overlap_records",
                                               "gwern_recent_or_revised_quarantine_count")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/f/pangram-at-home/data"))
    parser.add_argument("--output", type=Path,
                        default=Path("/mnt/f/pangram-at-home/data/attribution_authors_v1/gwern_graham"))
    parser.add_argument("--per-author", type=int, default=80)
    parser.add_argument("--min-words", type=int, default=300)
    parser.add_argument("--max-graham-scan", type=int, default=150)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
