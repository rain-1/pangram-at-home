"""Build a bounded fanfiction provenance pilot for private research review.

PAN19 and PAN20 come from official benchmark archives. AO3 text comes only
from a tiny, pinned 2020-labelled third-party slice; AO3 requests fetch work
metadata, not chapter text. The output needs overlap, grouping, and rights
review before inclusion in the detector pipeline or public release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import time
import zipfile

from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import requests


PAN_URL = "https://zenodo.org/api/records/20141612/files/pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23.zip/content"
PAN_RECORD = "https://zenodo.org/records/20141612"
PAN20_URL = "https://zenodo.org/api/records/5106099/files/pan20-authorship-verification-training-small.zip/content"
PAN20_RECORD = "https://zenodo.org/records/5106099"
PAN20_MD5 = "884de9d75561bd1a6b71002753253f31"
AO3_REPO = "ray0rf1re/AO3-2020"
AO3_REV = "5fa81ed59f6d75bd2aa8a593d56813316b1301c3"
AO3_FILE = "data/tokens_250K/train-00000.parquet"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def normalized_hash(text: str) -> str:
    return sha(" ".join(text.casefold().split()))


def download_pan(path: Path) -> None:
    if path.exists():
        return
    response = requests.get(PAN_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def download_pan20(path: Path) -> None:
    if path.exists():
        return
    temp = path.with_suffix(".zip.part")
    checksum = hashlib.md5()
    with requests.get(PAN20_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temp.open("wb") as file:
            for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                if block:
                    file.write(block)
                    checksum.update(block)
    if checksum.hexdigest() != PAN20_MD5:
        temp.unlink(missing_ok=True)
        raise ValueError("PAN20 archive MD5 mismatch")
    temp.replace(path)


def pan20_records(archive: Path, max_docs: int) -> tuple[list[dict], dict]:
    """Deduplicate repeated pair members and retain stable PAN author IDs."""
    archive_sha256 = sha(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        data_name = next(name for name in z.namelist() if name.endswith("training-small.jsonl"))
        truth_name = next(name for name in z.namelist() if name.endswith("training-small-truth.jsonl"))
        truth = {row["id"]: row["authors"] for row in
                 (json.loads(line) for line in z.open(truth_name))}
        # Streaming top-K by deterministic rank bounds retained text in memory.
        # Metadata for every distinct exact text detects conflicting author IDs.
        metadata = {}
        selected = {}
        heap = []
        pair_count = 0
        for line in z.open(data_name):
            pair = json.loads(line)
            pair_count += 1
            authors = truth.get(pair["id"])
            if not authors or len(authors) != 2:
                continue
            fandoms = json.loads(pair["fandoms"]) if isinstance(pair["fandoms"], str) else pair["fandoms"]
            for index, text in enumerate(pair["pair"]):
                if not text or len(text.split()) < 250:
                    continue
                content_hash = sha(text)
                item = metadata.setdefault(content_hash,
                                           {"authors": set(), "fandoms": set(),
                                            "pair_ids": [], "occurrences": 0})
                item["authors"].add(str(authors[index]))
                item["fandoms"].add(fandoms[index])
                item["occurrences"] += 1
                if len(item["pair_ids"]) < 3:
                    item["pair_ids"].append(pair["id"])
                rank = int(sha("pan20-pilot:" + content_hash), 16)
                if content_hash in selected:
                    continue
                if len(selected) < max_docs * 3:
                    selected[content_hash] = (rank, text)
                    heapq.heappush(heap, (-rank, content_hash))
                elif rank < -heap[0][0]:
                    _, worst = heapq.heapreplace(heap, (-rank, content_hash))
                    selected.pop(worst)
                    selected[content_hash] = (rank, text)
    rows = []
    conflicts = sum(len(item["authors"]) != 1 for item in metadata.values())
    author_counts = Counter()
    fandom_counts = Counter()
    for content_hash, (_, text) in sorted(selected.items(), key=lambda item: item[1][0]):
        info = metadata[content_hash]
        if len(info["authors"]) != 1:
            continue
        author = next(iter(info["authors"]))
        fandom = sorted(info["fandoms"])[0] if info["fandoms"] else None
        if author_counts[author] >= 4 or fandom_counts[fandom] >= 30:
            continue
        author_counts[author] += 1
        fandom_counts[fandom] += 1
        rows.append({"text": text, "source": "PAN20 fanfiction authorship verification small",
                     "source_id": "pan20:" + content_hash, "source_pair_ids": info["pair_ids"],
                     "work_id": None,
                     "work_id_status": "Original fanfiction work ID unavailable; source text is a roughly 21k-character excerpt",
                     "author_ids": ["pan20:" + author], "author_id_scope": "PAN20 archive stable author ID",
                     "fandom_id": fandom, "chapter_id": None,
                     "publication_date": None, "collection_date": "2020-03-19 (benchmark release)",
                     "source_revision": "Zenodo 5106099 v0.0.1; archive SHA-256 " + archive_sha256,
                     "canonical_url": PAN20_RECORD,
                     "human_origin_status": "Fanfiction benchmark published in 2020 with source author IDs; individual authorship not independently verified",
                     "rights_status": "Official research benchmark; source fanfic authors retain rights; public model/data reuse not cleared",
                     "source_sha256": content_hash, "clean_sha256": content_hash,
                     "normalized_text_sha256": normalized_hash(text),
                     "selection_status": "private_discriminative_research_candidate; author-disjoint_split_required"})
        if len(rows) >= max_docs:
            break
    return rows, {"archive_sha256": archive_sha256, "pairs": pair_count,
                  "distinct_exact_texts": len(metadata), "conflicting_author_texts": conflicts,
                  "selected": len(rows), "authors": len(author_counts), "fandoms": len(fandom_counts)}


def pan_records(archive: Path, max_docs: int) -> tuple[list[dict], dict]:
    candidates = []
    with zipfile.ZipFile(archive) as z:
        base = z.namelist()[0]
        collection = json.loads(z.read(base + "collection-info.json"))
        english = {row["problem-name"] for row in collection if row["language"] == "en"}
        fandoms = {}
        for problem in english:
            items = json.loads(z.read(f"{base}{problem}/fandom-info.json"))
            fandoms.update({f"{problem}/{x['true-author']}/{x['known-text']}": x["fandom"]
                            for x in items})
        for name in z.namelist():
            parts = name.removeprefix(base).split("/")
            if len(parts) != 3 or parts[0] not in english or not parts[1].startswith("candidate"):
                continue
            if not parts[2].endswith(".txt"):
                continue
            text = z.read(name).decode("utf-8").strip()
            if len(text.split()) < 250:
                continue
            key = f"{parts[0]}/{parts[1]}/{parts[2][:-4]}"
            candidates.append((sha("pan19-pilot:" + key), name, text,
                               f"{parts[0]}:{parts[1]}", fandoms.get(key)))
    candidates.sort()
    rows = []
    seen = set()
    author_counts = Counter()
    fandom_counts = Counter()
    for _, name, text, author, fandom in candidates:
        h = normalized_hash(text)
        if h in seen or author_counts[author] >= 8 or fandom_counts[fandom] >= 20:
            continue
        seen.add(h)
        author_counts[author] += 1
        fandom_counts[fandom] += 1
        source_id = "pan19:" + name.removeprefix(base)
        rows.append({"text": text, "source": "PAN19 fanfiction authorship training archive",
                     "source_id": source_id, "work_id": None,
                     "work_id_status": "Original work ID unavailable; archive text path is not a work ID",
                     "author_ids": ["pan19:" + author], "author_id_scope": "within PAN problem only",
                     "fandom_id": fandom, "chapter_id": None,
                     "publication_date": None, "collection_date": "2019-01-23",
                     "source_revision": "Zenodo record 20141612; archive SHA-256 " + sha(archive.read_bytes()),
                     "canonical_url": PAN_RECORD,
                     "human_origin_status": "Fanfiction benchmark published in 2019; per-document authorship not independently verified",
                     "rights_status": "Open research archive; individual fanfic reuse rights not verified; candidate only",
                     "source_sha256": sha(text), "clean_sha256": sha(text),
                     "normalized_text_sha256": h,
                     "selection_status": "provenance_candidate_not_training_eligible"})
        if len(rows) >= max_docs:
            break
    return rows, {"archive_sha256": sha(archive.read_bytes()), "english_problems": len(english),
                  "eligible_texts": len(candidates), "selected": len(rows),
                  "authors_local_to_problem": len(author_counts), "fandoms": len(fandom_counts)}


def work_metadata(work_id: int, session: requests.Session) -> dict | None:
    url = f"https://archiveofourown.org/works/{work_id}"
    try:
        response = session.get(url, timeout=20, allow_redirects=True)
    except requests.RequestException:
        return None
    if response.status_code != 200 or "archiveofourown.org" not in response.url:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    def value(selector: str) -> str | None:
        item = soup.select_one(selector)
        return item.get_text(" ", strip=True) if item else None
    byline = soup.select("h3.byline a[href^='/users/']")
    authors = sorted({"ao3:" + sha(link.get("href", ""))[:20] for link in byline})
    return {"title": value("h2.title"), "author_ids": authors,
            "published": value("dd.published"), "last_updated": value("dd.status"),
            "rating": value("dd.rating"), "language": value("dd.language"),
            "chapters": value("dd.chapters"), "canonical_url": url,
            "metadata_url": response.url}


def ao3_records(root: Path, max_pages: int, max_chapters_per_work: int,
                pause_seconds: float) -> tuple[list[dict], dict]:
    file = Path(hf_hub_download(repo_id=AO3_REPO, repo_type="dataset",
                                revision=AO3_REV, filename=AO3_FILE))
    raw = pq.read_table(file).to_pylist()
    by_work = defaultdict(list)
    for row in raw:
        by_work[int(row["storyId"])].append(row)
    work_ids = sorted(by_work, key=lambda x: sha(f"ao3-pilot:{x}"))[:max_pages]
    cache = root / "ao3_work_metadata.json"
    metadata = json.loads(cache.read_text()) if cache.exists() else {}
    session = requests.Session()
    session.headers.update({"User-Agent": "PangramResearch/0.1 (limited public-work metadata audit)"})
    requests_made = 0
    for work_id in work_ids:
        if str(work_id) in metadata:
            continue
        details = work_metadata(work_id, session)
        metadata[str(work_id)] = details
        cache.write_text(json.dumps(metadata, indent=2) + "\n")
        requests_made += 1
        if details is None:
            break  # Stop on a block/error; do not attempt a bypass.
        time.sleep(pause_seconds)
    rows = []
    reasons = Counter()
    for work_id in work_ids:
        details = metadata.get(str(work_id))
        if not details:
            reasons["metadata_unavailable_or_unattempted"] += 1
            continue
        try:
            published = date.fromisoformat(details["published"])
            last = date.fromisoformat(details["last_updated"] or details["published"])
        except (TypeError, ValueError):
            reasons["undated"] += 1
            continue
        if published.year > 2022 or last.year > 2022:
            reasons["post_2022_or_updated"] += 1
            continue
        if details["rating"] not in ("General Audiences", "Teen And Up Audiences"):
            reasons["rating_not_in_pilot"] += 1
            continue
        if details["language"] != "English" or not details["author_ids"]:
            reasons["language_or_author_unavailable"] += 1
            continue
        for chapter in sorted(by_work[work_id], key=lambda x: x["idx"])[:max_chapters_per_work]:
            text = chapter["text"].strip()
            if len(text.split()) < 250:
                continue
            rows.append({"text": text, "source": AO3_REPO,
                         "source_id": f"ao3_2020:{work_id}:{chapter['idx']}",
                         "work_id": f"ao3:{work_id}", "author_ids": details["author_ids"],
                         "author_id_scope": "Stable hash of public AO3 byline URL; current metadata",
                         "chapter_id": None, "source_chapter_index": chapter["idx"],
                         "chapter_id_status": "Source idx is an index, not a verified AO3 chapter ID",
                         "title": details["title"], "rating": details["rating"],
                         "publication_date": details["published"],
                         "last_updated_date": details["last_updated"],
                         "collection_date": "2020 (corpus card claim; chapter snapshot date unverified)",
                         "source_revision": AO3_REV,
                         "canonical_url": details["canonical_url"],
                         "human_origin_status": "Pre-2023 dated public work and claimed 2020 corpus cutoff; individual authorship and later revisions unverified",
                         "rights_status": "Uploader CC BY-NC claim is not individual-author permission; candidate only",
                         "source_sha256": sha(chapter["textRaw"]), "clean_sha256": sha(text),
                         "normalized_text_sha256": normalized_hash(text),
                         "selection_status": "provenance_candidate_not_training_eligible"})
    return rows, {"source_file": AO3_FILE, "source_file_sha256": sha(file.read_bytes()),
                  "source_revision": AO3_REV, "source_rows": len(raw),
                  "source_works": len(by_work), "work_pages_selected": len(work_ids),
                  "metadata_requests_made": requests_made, "pilot_works": len({r["work_id"] for r in rows}),
                  "pilot_chapters": len(rows), "exclusions": dict(reasons)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--pan20-max-docs", type=int, default=500)
    parser.add_argument("--pan19-max-docs", type=int, default=100)
    parser.add_argument("--ao3-max-work-pages", type=int, default=18)
    parser.add_argument("--ao3-max-chapters-per-work", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    out = args.root / "data/fiction_candidates_v1/fanfiction"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "records.jsonl"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    archive = out / "pan19_source.zip"
    pan20_archive = out / "pan20_training_small.zip"
    download_pan20(pan20_archive)
    pan20, pan20_meta = pan20_records(pan20_archive, args.pan20_max_docs)
    download_pan(archive)
    pan, pan_meta = pan_records(archive, args.pan19_max_docs)
    ao3, ao3_meta = ao3_records(out, args.ao3_max_work_pages,
                                 args.ao3_max_chapters_per_work, args.pause_seconds)
    rows = pan20 + pan + ao3
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {"role": "fanfiction provenance pilot; private research candidates require overlap and grouping audit before detector use",
                "split_status": "none; original work IDs absent for PAN excerpts and AO3 rights unresolved",
                "records": len(rows), "sources": dict(Counter(row["source"] for row in rows)),
                "records_sha256": sha(destination.read_bytes()),
                "pan20": pan20_meta, "pan19": pan_meta, "ao3_2020": ao3_meta}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "sources": manifest["sources"]}, indent=2))


if __name__ == "__main__":
    main()
