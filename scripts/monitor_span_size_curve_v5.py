"""Collect a finished Vast size-curve run, verify hashes, then destroy rental."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


RUNS = ("qwen3_token_repeat2_v5_5k_e1", "qwen3_token_repeat2_v5_10k_e1",
        "qwen3_token_repeat2_v5_20k_e1", "qwen3_token_repeat2_v5_5k_e4")
EVALS = ("v5_human_calibration", "v5_llmtrace_heldout",
         "v5_prior_synthetic_val", "v5_human_locked_test", "v5_aitdna", "v5_coauthor")
ROOT = Path("/mnt/f/pangram-at-home")
VAST = "/home/ubuntu/.venvs/pangram-vast-cli/bin/vastai"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--hourly", type=float, default=2.02)
    args = p.parse_args()
    started = time.time()
    destination = ROOT / "vast_span_size_curve_v5"
    destination.mkdir(exist_ok=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=10", "-p", str(args.port), f"root@{args.host}"]
    remote = "/workspace/pangram-data/"
    while True:
        elapsed = (time.time() - started) / 3600
        if elapsed > args.hours or elapsed * args.hourly > 12:
            raise RuntimeError("Rental cap reached; instance preserved for inspection")
        result = subprocess.run(ssh + ["cat " + remote + "span_size_curve_v5_status.json"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            try:
                state = json.loads(result.stdout)
            except json.JSONDecodeError:
                state = {}
            (destination / "remote_status.json").write_text(json.dumps(state, indent=2) + "\n")
            print(round(elapsed, 2), state.get("phase"), state.get("runs"), flush=True)
            if state.get("phase") in ("complete", "partial_failure", "failed") and state.get("export"):
                break
        time.sleep(60)
    expected = state["export"]["sha256"]
    archive = destination / "span_size_curve_v5_export.tar.gz"
    for attempt in range(5):
        copy = subprocess.run(["rsync", "-a", "--partial", "--append-verify", "--quiet",
                               "-e", f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p {args.port}",
                               f"root@{args.host}:{remote}span_size_curve_v5_export.tar.gz",
                               str(archive)], capture_output=True, text=True, timeout=1800)
        if copy.returncode == 0:
            break
        if attempt == 4:
            raise RuntimeError("Export transfer failed; instance preserved: " + copy.stderr)
        time.sleep(10)
    if sha(archive) != expected:
        raise RuntimeError("Export hash mismatch; instance preserved")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or "span_size_curve_v5_export_manifest.json" not in names:
            raise RuntimeError("Malformed archive; instance preserved")
        manifest = json.load(tar.extractfile("span_size_curve_v5_export_manifest.json"))
        if set(manifest["files"]) != set(names) - {"span_size_curve_v5_export_manifest.json"}:
            raise RuntimeError("Manifest mismatch; instance preserved")
        for member in members:
            path = Path(member.name)
            if member.name.startswith("/") or ".." in path.parts or not member.isfile():
                raise RuntimeError("Unsafe archive; instance preserved")
            if member.name.startswith("runs/") and path.parts[1] not in RUNS:
                raise RuntimeError("Unexpected run; instance preserved")
            if not member.name.startswith("runs/") and member.name not in (
                    "span_size_curve_v5_export_manifest.json", "span_size_curve_v5_status_snapshot.json",
                    "span_size_curve_v5_bootstrap.log") and not member.name.endswith(".train.log"):
                raise RuntimeError("Unexpected path; instance preserved")
            if member.name == "span_size_curve_v5_export_manifest.json":
                continue
            data = tar.extractfile(member).read()
            meta = manifest["files"][member.name]
            if len(data) != meta["bytes"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise RuntimeError("Artifact hash mismatch; instance preserved")
            target = ROOT / member.name
            if target.exists() and sha(target) != meta["sha256"]:
                raise RuntimeError(f"Existing artifact differs: {target}; instance preserved")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if state["phase"] != "complete":
        raise RuntimeError("Partial/failed run collected; instance preserved for diagnosis")
    for run in RUNS:
        folder = ROOT / "runs" / run
        needed = [folder / "best_adapter/adapter_model.safetensors", folder / "train_summary.json"]
        needed += [folder / f"{name}.json" for name in EVALS]
        if any(not path.is_file() for path in needed):
            raise RuntimeError(f"Incomplete run {run}; instance preserved")
    removed = subprocess.run([VAST, "destroy", "instance", str(args.instance), "--raw"],
                             capture_output=True, text=True, timeout=60)
    if removed.returncode:
        raise RuntimeError("Verified export, but destroy failed: " + removed.stderr)
    print("Verified all four runs and destroyed instance", args.instance, flush=True)


if __name__ == "__main__":
    main()
