"""Extract pre-2023 CC BY 4.0 abstracts from official ACL Anthology metadata."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

from lxml import etree

REPO = "https://github.com/acl-org/acl-anthology.git"
LICENSE_URL = "https://aclanthology.org/faq/copyright/"


def text(node) -> str:
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    repo = args.root / "reference" / "acl-anthology"
    xml_dir = repo / "data" / "xml"
    if not xml_dir.exists():
        raise SystemExit(f"Clone {REPO} to {repo} and sparse-checkout data first")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    output = args.root / "data" / "acl_abstracts_v1"
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for file in sorted(xml_dir.glob("*.xml")):
        if not file.name[:4].isdigit() or not (2016 <= int(file.name[:4]) <= 2022):
            continue
        root = etree.parse(str(file)).getroot()
        for volume in root.findall("./volume"):
            volume_id = volume.get("id", "")
            venue = text(volume.find("./meta/venue"))
            for paper in volume.findall("./paper"):
                abstract = text(paper.find("./abstract"))
                title = text(paper.find("./title"))
                if len(abstract) < 400 or len(abstract.split()) < 80 or not title:
                    continue
                paper_id = f"{file.stem}-{volume_id}.{paper.get('id')}"
                authors = []
                for author in paper.findall("./author"):
                    name = " ".join(filter(None, [text(author.find("./first")), text(author.find("./last"))]))
                    if name:
                        authors.append(name)
                documents.append({
                    "source": "acl_anthology",
                    "source_id": paper_id,
                    "canonical_url": f"https://aclanthology.org/{paper_id}/",
                    "year": int(file.name[:4]),
                    "venue": venue,
                    "volume": volume_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
                    "license": "CC BY 4.0",
                    "license_evidence_url": LICENSE_URL,
                })
    documents.sort(key=lambda d: d["source_id"])
    path = output / "documents.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for doc in documents:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    manifest = {
        "source": REPO,
        "revision": commit,
        "years": [2016, 2022],
        "count": len(documents),
        "by_year": dict(sorted(collections.Counter(d["year"] for d in documents).items())),
        "license_evidence_url": LICENSE_URL,
        "documents_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
