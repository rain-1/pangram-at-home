"""Collect private Vast sweep artifacts and enforce an initial cost cap."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--max-cost", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=16)
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/pangram-at-home/vast_results_v1"))
    parser.add_argument("--cli", default="/home/ubuntu/.venvs/pangram-vast-cli/bin/vastai")
    parser.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519"))
    args = parser.parse_args()
    if not os.getenv("VAST_API_KEY"):
        raise SystemExit("VAST_API_KEY is required in the environment")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=10", "-i", args.key, "-p", str(args.port), f"root@{args.host}"]
    scp = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-i", args.key, "-P", str(args.port)]
    status = "unknown"
    archive_copied = False
    try:
        while True:
            elapsed_hours = (time.monotonic() - started) / 3600
            estimated_cost = elapsed_hours * args.hourly_rate
            if elapsed_hours >= args.max_hours or estimated_cost >= args.max_cost:
                status = "budget_cap"
                print("Stopping at cost/time cap:", round(estimated_cost, 2), "USD", flush=True)
                break
            probe = run(ssh + ["test -f /workspace/pangram-data/HPO_DONE && echo done || "
                               "(test -f /workspace/pangram-data/HPO_FAILED && echo failed) || echo running"])
            if probe.returncode == 0 and probe.stdout.strip() in {"done", "failed"}:
                status = probe.stdout.strip()
                print("Remote status:", status, "estimated cost:", round(estimated_cost, 2), flush=True)
                break
            if int(elapsed_hours * 60) % 30 == 0:
                print("Still running; elapsed", round(elapsed_hours, 2), "hours; estimated cost",
                      round(estimated_cost, 2), "USD", flush=True)
            time.sleep(60)
        for name in ("hpo_export_v1.tar.gz", "hpo_driver.log", "ablation_driver.log",
                     "hpo_selection.log", "hpo_export.log"):
            source = f"root@{args.host}:/workspace/pangram-data/{name}"
            for attempt in range(3):
                copied = run(scp + [source, str(args.output / name)])
                if copied.returncode == 0:
                    print("Collected", name, flush=True)
                    if name == "hpo_export_v1.tar.gz":
                        archive_copied = True
                    break
                if attempt < 2:
                    time.sleep(10)
        (args.output / "monitor_status.json").write_text(json.dumps({
            "instance": args.instance, "status": status,
            "elapsed_hours": (time.monotonic() - started) / 3600,
            "estimated_cost_usd": (time.monotonic() - started) / 3600 * args.hourly_rate,
        }, indent=2) + "\n")
    finally:
        # Keep the instance disk for debugging or partial recovery if the sweep
        # fails or reaches the cap before the export has been copied.
        action = "destroy" if status == "done" and archive_copied else "stop"
        command = ([args.cli, "--raw", "destroy", "instance", str(args.instance), "-y"]
                   if action == "destroy" else
                   [args.cli, "--raw", "stop", "instance", str(args.instance)])
        result = run(command)
        print(action, "response:", result.stdout.strip()[:500],
              "exit", result.returncode, flush=True)


if __name__ == "__main__":
    main()
