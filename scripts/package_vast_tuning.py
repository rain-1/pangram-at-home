"""Build a private Vast upload containing code plus train/validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home")
CODE = ["requirements.txt", "requirements-tune.txt", "scripts/train_segment_lora.py",
        "scripts/tune_diverse_ray.py", "scripts/bootstrap_vast_tuning.sh"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "packages/vast_tuning_v1.tar.gz")
    args = parser.parse_args()
    files = [(REPO / name, f"pangram-at-home/{name}") for name in CODE]
    datasets = [ROOT / "data/diverse_pyramid_v1"] + sorted((ROOT / "data").glob("diverse_ablation_v1_*"))
    for folder in datasets:
        for name in ("train_full.parquet", "val_full.parquet", "manifest.json"):
            path = folder / name
            if path.exists():
                files.append((path, f"pangram-data/data/{folder.name}/{name}"))
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "files": {archive: sha256(path) for path, archive in files},
                "model": {"repo": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
                "test_sets_included": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as tar:
        for path, archive in files:
            tar.add(path, arcname=archive)
        encoded = json.dumps(manifest, indent=2).encode()
        import io
        info = tarfile.TarInfo("pangram-at-home/package_manifest.json")
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
