#!/usr/bin/env python3
"""Acquire public, whole-essay author-attribution candidates from ACX and LessWrong.

All corpus material is written outside this repository. Public availability is not
treated as a redistribution license. Byline/date evidence is recorded separately
from human-origin confidence; this script does not certify biological authorship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

ACX = "https://www.astralcodexten.com"
LW = "https://www.lesswrong.com"
AUTHOR_IDS = {"scott-alexander": "public-byline:Scott Alexander",
              "eliezer-yudkowsky": "public-byline:Eliezer Yudkowsky"}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pangram-attribution-research/1.0 (public pages; contact: research project)"})


def get(url: str, **kwargs: Any) -> requests.Response:
    for attempt in range(5):
        try:
            r = SESSION.get(url, timeout=40, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_html(html: str) -> str:
    # The HTMLParser backend retains invalid nested <p> elements. Older
    # LessWrong posts contain unclosed paragraph tags; iterating those parents
    # and children separately duplicates most of a document many times. lxml
    # applies normal HTML tree repair before extraction.
    soup = BeautifulSoup(html or "", "lxml")
    for node in soup.select("script,style,nav,footer,header,form,button,svg,figure,iframe,video,audio,.captioned-image-container,.subscription-widget-wrap,.paywall"):
        node.decompose()
    for node in soup.select("blockquote,pre,code, .footnote, .footnotes"):
        node.decompose()
    chunks = []
    block_tags = {"p", "h1", "h2", "h3", "li"}

    def own_text(node: Tag) -> str:
        pieces: list[str] = []
        def visit(child: Any) -> None:
            if isinstance(child, NavigableString):
                pieces.append(str(child))
            elif isinstance(child, Tag):
                if child.name in block_tags:
                    return
                for grandchild in child.children:
                    visit(grandchild)
        for child in node.children:
            visit(child)
        return " ".join(" ".join(pieces).split())

    for node in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        txt = own_text(node)
        if not txt:
            continue
        if txt.lower() in {"subscribe", "share", "comments", "leave a comment"}:
            continue
        chunks.append(txt)
    # De-duplicate only immediate repeated paragraph strings from layout artifacts.
    out = []
    for x in chunks:
        if not out or x != out[-1]:
            out.append(x)
    return "\n\n".join(out).strip()


def date_year(value: str) -> int:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).year
    except Exception:
        return 0


def acx_archive() -> list[dict[str, Any]]:
    out, seen = [], set()
    # ACX began in 2021. Continue to the end of its public archive; stop once
    # the archive has crossed into 2020, retaining 2021+ entries only.
    for offset in range(0, 3200, 40):
        batch = get(f"{ACX}/api/v1/archive", params={"sort": "new", "offset": offset}).json()
        if not batch:
            break
        for item in batch:
            if item.get("slug") and item["slug"] not in seen:
                seen.add(item["slug"])
                out.append(item)
        if date_year(batch[-1].get("post_date", "")) < 2021:
            break
    return out


def fetch_acx(max_docs: int, outdir: Path) -> list[dict[str, Any]]:
    archive = acx_archive()
    # Main essays only: exclude open threads, roundups/link posts, contests,
    # metaposts, and posts with an explicitly different byline.
    rejected = re.compile(r"^(open thread|hidden open thread|links for|links [—-]|your book review|finalist #|meetup|apply for|weekly|monthly|prediction contest|highlights? from (the )?comments|comments on |acx grants?:? project updates)", re.I)
    reject_ai_experiment = re.compile(r"(?:can this ai save|ai wrote|written by ai|llm[- ]written|ai[- ]generated|chatgpt wrote)", re.I)
    candidates = []
    for p in archive:
        date = p.get("post_date", "")
        title = p.get("title", "").strip()
        if not (2021 <= date_year(date) <= 2022) or not title or rejected.search(title) or reject_ai_experiment.search(title):
            continue
        # Skip non-essay post types and posts not explicitly marked public.
        if p.get("type") not in (None, "newsletter"):
            continue
        candidates.append(p)

    rows = []
    raw_dir = outdir / "raw" / "astral_codex_ten"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for meta in candidates:
        if len(rows) >= max_docs:
            break
        slug = meta["slug"]
        url = f"{ACX}/api/v1/posts/{slug}"
        try:
            doc = get(url).json()
        except Exception as e:
            print(f"ACX fetch failed {slug}: {e}")
            continue
        bylines = [b.get("name", "") for b in doc.get("publishedBylines", [])]
        if bylines != ["Scott Alexander"]:
            continue
        if doc.get("audience") != "everyone":
            continue
        # Guest/collection posts are excluded even if Scott appears as platform
        # publisher; an explicit guest byline displaces the author attribution.
        subtitle = doc.get("subtitle") or ""
        if re.search(r"\b(guest post|by [A-Z])", subtitle, re.I):
            continue
        body = doc.get("body_html") or ""
        clean = clean_html(body)
        if len(clean.split()) < 150:
            continue
        raw_name = f"{slug}.html"
        (raw_dir / raw_name).write_bytes(body.encode("utf-8"))
        canonical = doc.get("canonical_url") or f"{ACX}/p/{slug}"
        pubdate = doc.get("post_date", meta.get("post_date"))
        modified = doc.get("updated_at")
        unchanged_evidence = bool(modified and modified[:4] <= "2022")
        rows.append({
            "text": clean,
            "author_id": "scott-alexander",
            "document_id": f"acx:{doc.get('id', slug)}",
            "title": doc.get("title", meta.get("title", "")),
            "canonical_url": canonical,
            "publication_date": pubdate,
            "source_modified_date": modified,
            "attribution_eligible": True,
            "strict_human_candidate": bool(date_year(pubdate or "") <= 2022 and unchanged_evidence),
            "human_origin_status": ("pre_2023_byline_and_no_observed_later_modification_still_not_proof" if unchanged_evidence else "pre_2023_byline_current_text_revision_history_uncertain"),
            "human_origin_evidence": "Astral Codex Ten public archive/API gives dated publication and Scott Alexander byline; no document-level authorship/revision audit was exposed.",
            "rights_status": "copyrighted_public_access_no_explicit_reuse_license_found_private_research_only",
            "source_sha256": sha(body), "clean_sha256": sha(clean),
            "group_id": "content-sha256:" + sha(clean),
            "source": "Astral Codex Ten public archive API and public post API",
            "source_metadata": {"api_url": url, "archive_slug": slug, "updated_at": modified,
                                "published_bylines": bylines, "audience": doc.get("audience"),
                                "type": doc.get("type"), "raw_file": str((raw_dir / raw_name).relative_to(outdir))},
        })
        time.sleep(.05)
    return rows


def lw_posts(user_id: str) -> list[dict[str, Any]]:
    query = '''query { posts(input: {terms: {view: "userPosts", userId: "%s", limit: 1000, offset: 0}}) {
      results { _id title pageUrl postedAt coauthorUserIds coauthors { _id username displayName }
        contents { html } }
    } }''' % user_id
    r = SESSION.post(f"{LW}/graphql", json={"query": query}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(json.dumps(j["errors"])[:1000])
    return j["data"]["posts"]["results"]


def fetch_lw(outdir: Path, max_docs: int, old_fallback: int) -> list[dict[str, Any]]:
    uid = "nmk3nLpQE89dMRzzN"  # extracted from LessWrong's official profile page
    posts = lw_posts(uid)
    recent = [p for p in posts if 2020 <= date_year(p.get("postedAt", "")) <= 2022]
    older = [p for p in posts if date_year(p.get("postedAt", "")) < 2020]
    # Recent dates first; optionally add older, explicitly separated records if
    # the requested bounded count is not reached.
    ordered = recent + older[:old_fallback]
    raw_dir = outdir / "raw" / "lesswrong"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in ordered:
        if len(rows) >= max_docs:
            break
        html = ((p.get("contents") or {}).get("html") or "")
        clean = clean_html(html)
        # Avoid short AMA/announcement/quick-take-like items; retain whole essays.
        if len(clean.split()) < 150:
            continue
        coauthors = [a.get("displayName") for a in (p.get("coauthors") or []) if a.get("displayName")]
        if coauthors:
            continue
        year = date_year(p.get("postedAt", ""))
        raw_name = f"{p['_id']}.html"
        (raw_dir / raw_name).write_bytes(html.encode("utf-8"))
        canonical = p.get("pageUrl") or f"{LW}/posts/{p['_id']}"
        rows.append({
            "text": clean,
            "author_id": "eliezer-yudkowsky",
            "document_id": f"lesswrong:{p['_id']}",
            "title": p.get("title", ""), "canonical_url": canonical,
            "publication_date": p.get("postedAt"),
            "source_modified_date": p.get("updatedAt"),
            "attribution_eligible": True,
            "strict_human_candidate": False,
            "human_origin_status": ("pre_2023_byline_current_text_revision_history_unavailable" if year >= 2020 else "older_byline_human_likely_date_stratum_revision_history_unavailable"),
            "human_origin_evidence": "LessWrong public profile and GraphQL post record supply author account, publication timestamp, and full post content; this is platform/byline provenance, not a forensic authorship audit.",
            "rights_status": "copyrighted_public_access_no_post_specific_reuse_license_found_private_research_only",
            "source_sha256": sha(html), "clean_sha256": sha(clean),
            "group_id": "content-sha256:" + sha(clean),
            "source": "LessWrong official profile and public GraphQL API",
            "source_metadata": {"profile_url": f"{LW}/users/eliezer-yudkowsky", "graphql_url": f"{LW}/graphql",
                                "post_id": p["_id"], "coauthors": coauthors,
                                "date_stratum": "2020-2022" if year >= 2020 else "pre-2020 older fallback",
                                "raw_file": str((raw_dir / raw_name).relative_to(outdir))},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("/mnt/f/pangram-at-home/data/attribution_authors_v1/scott_eliezer"))
    ap.add_argument("--scott-max", type=int, default=100)
    ap.add_argument("--eliezer-max", type=int, default=60)
    ap.add_argument("--eliezer-older-fallback", type=int, default=31)
    args = ap.parse_args()
    if (args.outdir / "records.jsonl").exists():
        ap.error(f"refusing to overwrite existing corpus: {args.outdir / 'records.jsonl'}; choose a fresh --outdir")
    args.outdir.mkdir(parents=True, exist_ok=True)
    scott = fetch_acx(args.scott_max, args.outdir)
    eliezer = fetch_lw(args.outdir, args.eliezer_max, args.eliezer_older_fallback)
    records = scott + eliezer
    # Keep duplicate documents as records but assign shared content-based groups.
    with (args.outdir / "records.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": "attribution_authors_v1_scott_eliezer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_path": "records.jsonl", "raw_html_directory": "raw/",
        "record_count": len(records),
        "counts_by_author": dict(Counter(r["author_id"] for r in records)),
        "counts_by_status": dict(Counter(r["human_origin_status"] for r in records)),
        "counts_by_year": dict(Counter(r["publication_date"][:4] for r in records if r.get("publication_date"))),
        "strict_human_candidate_count": sum(bool(r.get("strict_human_candidate")) for r in records),
        "minimum_clean_word_count": min((len(r["text"].split()) for r in records), default=0),
        "rights": "No explicit corpus reuse license was identified in the source pages/API. Kept outside Git for private research; do not redistribute texts absent permission.",
        "provenance_limitations": [
            "A public byline/account and publication timestamp establish platform attribution, not identity verification or keystroke-level human authorship.",
            "The platforms expose current page text and an updated_at value where available; revision history was not audited, so later edits cannot be excluded.",
            "Posts with explicit guest bylines/coauthors and short non-essay items are omitted. Exact-clean-text hashes group duplicates; semantic/cross-post duplicates require further review.",
            "Posts summarizing or compiling other people's updates/comments are excluded when recognizable from title/content; residual embedded source/quotation material may remain.",
            "Legacy Slate Star Codex returned HTTP 403 during this acquisition, so no SSC originals were fetched. ACX is used for Scott's publicly accessible 2021-22 work.",
        ],
        "sources": [
            {"name": "Astral Codex Ten archive/API", "archive": f"{ACX}/archive?sort=new", "data_endpoint": f"{ACX}/api/v1/archive?sort=new&offset=...", "post_endpoint": f"{ACX}/api/v1/posts/<slug>"},
            {"name": "LessWrong Eliezer profile", "profile": f"{LW}/users/eliezer-yudkowsky", "api": f"{LW}/graphql"},
        ],
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("record_count", "counts_by_author", "counts_by_year", "counts_by_status")}, indent=2))


if __name__ == "__main__":
    main()
