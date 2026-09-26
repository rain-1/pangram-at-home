"""Acquire a bounded, archived original-online-fiction research candidate.

Full copyrighted story text stays outside Git. Run with an empty output path.
The Internet Archive snapshot timestamp is version evidence, not a guarantee
that every sentence was unaided human writing.
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
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pyarrow.parquet as pq
import requests


BASE = "https://beneath-ceaseless-skies.com"
YEARS = (2021, 2020, 2019, 2018)
AGENT = "pangram-at-home-research/0.1 (bounded private fiction audit)"
ARCHIVE_CUTOFF = "20221231"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 30) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": AGENT}, timeout=timeout)
    response.raise_for_status()
    if len(response.content) > 3_000_000:
        raise ValueError("Unexpectedly large page")
    return response


def normalized_shingles(text: str, width: int = 24) -> set[bytes]:
    words = re.findall(r"\b\w+\b", text.casefold())
    return {hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words) - width + 1))}


def parent_fingerprints(data_root: Path) -> tuple[set[str], set[bytes]]:
    hashes, shingles = set(), set()
    for split in ("train", "val", "test"):
        table = pq.read_table(data_root / "diverse_pyramid_v1" / f"{split}_full.parquet", columns=["text"])
        for text in table.column("text").to_pylist():
            hashes.add(digest(text.encode()))
            shingles.update(normalized_shingles(text))
    return hashes, shingles


def issue_links() -> tuple[list[str], dict[str, str]]:
    """Only story URLs present on the publisher's pre-2022 issue-year pages."""
    links, page_hashes = set(), {}
    for year in YEARS:
        for page in range(1, 8):
            url = f"{BASE}/issues/{year}/" + (f"page/{page}/" if page > 1 else "")
            try:
                raw = fetch(url).content
            except (requests.RequestException, ValueError):
                break
            soup = BeautifulSoup(raw, "html.parser")
            page_hashes[url] = digest(raw)
            for node in soup.select('a[href*="/stories/"]'):
                target = node.get("href", "").split("#", 1)[0]
                parsed = urlparse(target)
                if parsed.hostname in ("beneath-ceaseless-skies.com", "www.beneath-ceaseless-skies.com") and parsed.path.startswith("/stories/"):
                    links.add(BASE + parsed.path.rstrip("/") + "/")
            next_url = f"{BASE}/issues/{year}/page/{page + 1}/"
            if not any(node.get("href", "").rstrip("/") == next_url.rstrip("/") for node in soup.select("a[href]")):
                break
    return sorted(links), page_hashes


def parse_story(raw: bytes) -> dict | None:
    soup = BeautifulSoup(raw, "html.parser")
    container = soup.select_one(".right-content")
    body = soup.select_one(".bcs-story-content")
    if container is None or body is None:
        return None
    title_node = container.select_one(".post-title")
    author_node = container.select_one(".post-author")
    author_link = author_node.select_one("a[href]") if author_node else None
    issue_node = container.select_one(".issue-title")
    if title_node is None or author_link is None or issue_node is None:
        return None
    # Copy the node to remove nested byline/date before reading the title.
    title_copy = BeautifulSoup(str(title_node), "html.parser")
    for node in title_copy.select(".post-author, .issue-title"):
        node.decompose()
    title = title_copy.get_text(" ", strip=True)
    author = author_link.get_text(" ", strip=True)
    author_url = author_link["href"].split("#", 1)[0]
    issue = issue_node.get_text(" ", strip=True)
    match = re.search(r"Issue\s+#(\d+),\s+([A-Za-z]+\s+\d{1,2},\s+20\d{2})", issue)
    if not match:
        return None
    date = datetime.strptime(match.group(2), "%B %d, %Y").date().isoformat()
    for node in body.select("script, style, nav, aside, figure, blockquote, .wp-caption"):
        node.decompose()
    paragraphs = []
    for node in body.select("p"):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text and not re.fullmatch(r"[\W_]+", text):
            paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    return {"title": title, "author": author, "author_url": author_url,
            "issue_id": match.group(1), "publication_date": date, "text": text}


def acquire_story(url: str, try_archive: bool) -> tuple[dict | None, str]:
    target = f"https://web.archive.org/web/{ARCHIVE_CUTOFF}id_/{url}"
    archive_failure = "not_attempted"
    if try_archive:
        try:
            response = fetch(target, timeout=20)
            match = re.search(r"/web/(\d{14})id_/", response.url)
            if match and match.group(1)[:8] <= ARCHIVE_CUTOFF:
                parsed = parse_story(response.content)
                if parsed and len(parsed["text"].split()) >= 450:
                    parsed.update({"canonical_url": url, "archive_url": response.url,
                                   "archive_date": match.group(1)[:8],
                                   "source_sha256": digest(response.content)})
                    return parsed, "archived"
                archive_failure = "unparseable_or_short_snapshot"
            else:
                archive_failure = "missing_pre2023_snapshot"
        except (requests.RequestException, ValueError):
            archive_failure = "snapshot_fetch_failed"
    try:
        response = fetch(url)
        parsed = parse_story(response.content)
        if not parsed:
            return None, "unparseable_current_page"
        if parsed["publication_date"] > ARCHIVE_CUTOFF[:4] + "-" + ARCHIVE_CUTOFF[4:6] + "-" + ARCHIVE_CUTOFF[6:]:
            return None, "post2022_publication"
        if len(parsed["text"].split()) < 450:
            return None, "short_or_nonprose"
        parsed.update({"canonical_url": url, "archive_url": None,
                       "archive_date": None, "source_sha256": digest(response.content),
                       "archive_failure": archive_failure})
        return parsed, "current_page"
    except (requests.RequestException, ValueError):
        return None, "current_fetch_failed"


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    urls, index_hashes = issue_links()
    # Work over complete stories, not snippets or arbitrary paragraph chunks.
    # Archived and current page versions retain separate provenance flags.
    records, exclusions = [], Counter()
    seen_author = Counter()
    if args.seed_records:
        for line in args.seed_records.read_text().splitlines():
            record = json.loads(line)
            record.setdefault("source_version_status", "archived_pre2023")
            record.setdefault("strict_human_candidate", True)
            records.append(record)
            seen_author[record["author_ids"][0].removeprefix("bcs_author:")] += 1
    seen_urls = {row["canonical_url"] for row in records}
    discovered_url_count = len(urls)
    urls = [url for url in urls if url not in seen_urls]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = ((url, index < args.archive_attempts) for index, url in enumerate(urls))
        for parsed, reason in pool.map(lambda task: acquire_story(*task), tasks):
            if parsed is None:
                exclusions[reason] += 1
                continue
            author_slug = urlparse(parsed["author_url"]).path.rstrip("/").rsplit("/", 1)[-1]
            if seen_author[author_slug] >= args.max_per_author:
                exclusions["author_cap"] += 1
                continue
            work_slug = urlparse(parsed["canonical_url"]).path.rstrip("/").rsplit("/", 1)[-1]
            work_id = "bcs:" + work_slug
            text = parsed.pop("text")
            version_status = "archived_pre2023" if parsed["archive_date"] else "current_revision_unverified"
            records.append({"text": text, "source": "beneath_ceaseless_skies",
                            "source_id": work_id, "work_id": work_id,
                            "group_id": work_id, "author_ids": ["bcs_author:" + author_slug],
                            "author_name": parsed.pop("author"),
                            "domain": "original_online_fiction", "label": 0,
                            "source_revision": parsed["archive_date"],
                            "source_version_status": version_status,
                            "strict_human_candidate": bool(parsed["archive_date"]),
                            "human_origin_status": ("publisher-bylined story with pre-2023 archived original page; no independent unaided-authorship proof"
                                                    if parsed["archive_date"] else
                                                    "publisher-bylined story dated before 2023; current revision date unavailable"),
                            "rights_status": "copyright retained by author/publisher; private research candidate only; training reuse needs rights review",
                            "clean_sha256": digest(text.encode()),
                            "word_count": len(text.split()), **parsed})
            seen_author[author_slug] += 1
            if len(records) >= args.target:
                break
    if len({row["work_id"] for row in records}) != len(records) or len({row["clean_sha256"] for row in records}) != len(records):
        raise RuntimeError("Duplicate work or cleaned text")
    parent_hashes, parent_shingles = parent_fingerprints(args.data_root)
    for row in records:
        row["detector_exact_text_overlap"] = row["clean_sha256"] in parent_hashes
        row["detector_24word_shingle_overlap_count"] = len(normalized_shingles(row["text"]) & parent_shingles)
    args.output.mkdir(parents=True)
    path = args.output / "records.jsonl"
    with path.open("w") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"role": "private candidate for original online human fiction; not active training or evaluation",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": BASE, "archive_cutoff": ARCHIVE_CUTOFF, "index_hashes": index_hashes,
                "target": args.target, "max_per_author": args.max_per_author,
                "source_story_urls": discovered_url_count, "records": len(records),
                "authors": len(seen_author), "years": dict(Counter(row["publication_date"][:4] for row in records)),
                "source_versions": dict(Counter(row.get("source_version_status", "archived_pre2023") for row in records)),
                "exclusions": dict(exclusions),
                "overlap_records": sum(bool(row["detector_exact_text_overlap"] or row["detector_24word_shingle_overlap_count"]) for row in records),
                "overlap_scope": "exact cleaned SHA-256 and normalized 24-word shingles against diverse_pyramid_v1 train/val/test full text",
                "records_sha256": digest(path.read_bytes()),
                "human_origin_caveat": "Archived publisher pages give stronger version provenance; current pages only carry pre-2023 publication dates. Neither proves absence of all human-AI collaboration or quotation.",
                "rights_caveat": "Full text is copyrighted and retained privately. Public availability is not a training license."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("source_story_urls", "records", "authors", "years", "exclusions", "overlap_records", "records_sha256")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/f/pangram-at-home/data"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/data/fiction_candidates_v1/original_online"))
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--max-per-author", type=int, default=3)
    parser.add_argument("--archive-attempts", type=int, default=25)
    parser.add_argument("--seed-records", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
