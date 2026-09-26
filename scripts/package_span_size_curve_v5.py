"""Package audited size-curve data, code, and initialization adapter for Vast."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]


def main():
    files: list[tuple[Path, str]] = []
    code = (
        "requirements-span.txt", "scripts/bootstrap_span_vast.sh",
        "scripts/launch_span_size_curve_v5.sh", "scripts/run_span_size_curve_v5.py",
        "scripts/train_token_lora.py", "scripts/train_segment_lora.py",
        "scripts/span_data.py", "scripts/span_metrics.py", "scripts/evaluate_span_pilot.py",
    )
    files.extend((REPO / name, "pangram-at-home/" + name) for name in code)
    curve = ROOT / "data/span_size_curve_v5"
    files.append((curve / "manifest.json", "pangram-data/data/span_size_curve_v5/manifest.json"))
    for size in (5000, 10000, 20000):
        folder = curve / f"size_{size}"
        for name in ("train.jsonl", "val.jsonl", "test_llmtrace.jsonl", "manifest.json"):
            path = folder / name
            files.append((path, "pangram-data/" + str(path.relative_to(ROOT))))
    datasets = {
        "span_human_eval_v2": ("calibration.jsonl", "test.jsonl", "manifest.json"),
        "span_training_v4": ("val.jsonl", "manifest.json"),
        "span_realistic_eval_v1": ("test.jsonl", "manifest.json"),
        "span_sources_v5/normalized_aitdna_real": ("locked_test.jsonl", "manifest.json"),
    }
    for folder, names in datasets.items():
        for name in names:
            path = ROOT / "data" / folder / name
            files.append((path, "pangram-data/" + str(path.relative_to(ROOT))))
    adapter = ROOT / "runs/vast_hpo_selected_v3/best_adapter"
    for path in adapter.iterdir():
        if path.is_file():
            files.append((path, "pangram-data/" + str(path.relative_to(ROOT))))
    manifest = {
        "purpose": "private nested span-data size curve and frozen evaluation",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "files": {},
    }
    output = ROOT / "packages/span_size_curve_v5.tar.gz"
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path, name in files:
            data = path.read_bytes()
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("pangram-at-home/span_package_manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(output, output.stat().st_size, "bytes", len(files), "files")


if __name__ == "__main__":
    main()
