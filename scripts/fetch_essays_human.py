"""Fetch the official PERSUADE 2.0 essay file and make a text-only eval corpus.

The source contains essays from grades 6–12 and detailed student attributes.
The default path streams the official CSV to a temporary file, exports only
essay text and opaque IDs, then deletes the source CSV. Treat the output as
restricted research data; this is not a training-data exporter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests


FILE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"
SOURCE = "https://github.com/scrosseye/persuade_corpus_2.0"
LICENSE = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
RIGHTS_EVIDENCE = "https://github.com/scrosseye/persuade_corpus_2.0#readme"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(path: Path) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "pangram-at-home-research/0.1"
    url = "https://drive.google.com/uc?export=download&id=" + FILE_ID
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()
    # Large Drive files require a confirmation form. Follow its explicit token.
    if "text/html" in response.headers.get("Content-Type", ""):
        html = response.text
        match = re.search(r'name="confirm" value="([^"]+)"', html)
        if not match:
            raise RuntimeError("Google Drive did not provide a download confirmation")
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params={"id": FILE_ID, "export": "download", "confirm": match.group(1)},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise RuntimeError("Expected CSV bytes; received an HTML page")
    with path.open("wb") as out:
        for block in response.iter_content(1024 * 1024):
            if block:
                out.write(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/data/persuade_essays_v1"))
    parser.add_argument("--input-csv", type=Path, help="Use an already downloaded official PERSUADE 2.0 training CSV")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    temp_path = args.input_csv
    delete_temp = False
    if temp_path is None:
        fd, name = tempfile.mkstemp(prefix="persuade_source_", suffix=".csv", dir=root)
        Path(name).unlink(missing_ok=True)
        import os

        os.close(fd)
        temp_path = Path(name)
        delete_temp = True
        print("Downloading official source CSV (about 588 MB)...", flush=True)
        download(temp_path)

    rows: list[dict] = []
    seen: set[str] = set()
    headers: list[str] = []
    try:
        with temp_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            headers = reader.fieldnames or []
            id_col = next((x for x in ("essay_id_comp", "essay_id", "id") if x in headers), None)
            text_col = next((x for x in ("full_text", "essay_text", "text") if x in headers), None)
            if not id_col or not text_col:
                raise RuntimeError(f"Cannot locate essay ID/text columns; headers={headers}")
            for record in reader:
                essay_id = (record.get(id_col) or "").strip()
                text = re.sub(r"\s+", " ", (record.get(text_col) or "")).strip()
                if not essay_id or not text or essay_id in seen:
                    continue
                seen.add(essay_id)
                raw = text.encode("utf-8")
                rows.append({
                    "text_id": f"persuade:{essay_id}",
                    "text": text,
                    "label": 0,
                    "source": "persuade_2.0",
                    "source_id": essay_id,
                    "group_id": f"persuade:{essay_id}",
                    "domain": "student_argumentative_essay",
                    "collection_period": "Feedback Prize 2021–2022; essays predate ChatGPT",
                    "license": LICENSE,
                    "canonical_url": SOURCE,
                    "raw_sha256": sha(raw),
                    "clean_sha256": sha(raw),
                    "word_count": len(text.split()),
                })
        out = root / "human_eval.parquet"
        pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd")
        manifest = {
            "source": SOURCE,
            "source_file_id": FILE_ID,
            "source_file": "persuade_corpus_2.0_train.csv",
            "source_release": "PERSUADE 2.0; official authors' repository",
            "source_release_date": "2024",
            "collection_period": "Feedback Prize 2021–2022; source consists of grade 6–12 student writing",
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "rights_evidence_url": RIGHTS_EVIDENCE,
            "use": "Restricted noncommercial research evaluation only; excluded from all training",
            "privacy": "Source has minors' work and student attributes. Export omits all attributes and annotations, keeps only essay text and opaque source ID. Original CSV deleted after export when downloaded by this script. Output remains sensitive student writing; restrict access, do not publish, and honor takedown requests.",
            "rows": len(rows),
            "unique_source_ids": len(seen),
            "source_headers_seen": headers,
            "word_count_min": min((r["word_count"] for r in rows), default=0),
            "word_count_median": sorted(r["word_count"] for r in rows)[len(rows) // 2] if rows else 0,
            "word_count_max": max((r["word_count"] for r in rows), default=0),
            "output_sha256": sha(out.read_bytes()),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {len(rows)} unique essays to {out}", flush=True)
    finally:
        if delete_temp:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
