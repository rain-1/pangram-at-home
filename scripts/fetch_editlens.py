"""Download the gated, noncommercial EditLens research benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import snapshot_download

REPO = "pangram/editlens_iclr"
REVISION = "34ac1ade5a814c4cea098ef70328d92880d1f1c0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or dotenv_values(".env").get("HF_API_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN or HF_API_TOKEN in .env is required for gated EditLens access")
    path = args.root / "data" / "editlens_iclr"
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        token=token,
        local_dir=path,
        max_workers=4,
    )
    print(path)


if __name__ == "__main__":
    main()
