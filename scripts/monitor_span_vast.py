"""Durable collector and rental closer for the specific Vast Repeat2 v4 job.

Run in a persistent local session after the remote runner starts. The instance
is destroyed only after the export has been copied, verified, and installed.
At a time/cost cap, collect what is available and stop (preserve) the rental.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

from dotenv import load_dotenv


REMOTE = "/workspace/pangram-data"
STATUS_NAME = "span_v4_status.json"
ARCHIVE_NAME = "span_v4_export.tar.gz"
MANIFEST_NAME = "span_v4_export_manifest.json"
OLD = "qwen3_token_repeat2_v3_pilot1"
NEW = "qwen3_token_repeat2_v4_pilot1"
TERMINAL = {"complete", "failed"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as file:
        json.dump(data, file, indent=2)
        file.write("\n")
        temp = Path(file.name)
    os.replace(temp, path)


def call(args: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def safe_member(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Unsafe archive member: {name!r}")
    allowed_run = len(path.parts) >= 3 and path.parts[0] == "runs" and path.parts[1] in (OLD, NEW)
    allowed_report = (len(path.parts) == 3 and path.parts[:2] == ("reports", "span_v4")
                      and path.suffix in {".md", ".json", ".pdf", ".log"})
    if name not in (MANIFEST_NAME, STATUS_NAME, "span_v4_train.log",
                    "span_v4_status_snapshot.json") and not (allowed_run or allowed_report):
        raise ValueError(f"Unexpected archive member: {name!r}")
    if any(part.startswith(".") or part.lower() in {".env", "secrets", "credentials"}
           for part in path.parts):
        raise ValueError(f"Sensitive or hidden archive path: {name!r}")
    return path


def verify_and_extract(archive: Path, destination: Path, expected_sha: str,
                       external_manifest: dict | None = None) -> dict:
    actual = digest(archive)
    if actual != expected_sha:
        raise ValueError("Archive SHA-256 disagrees with remote status")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            safe_member(member.name)
            if not member.isfile() or member.size > 20 * 1024**3:
                raise ValueError(f"Unexpected member type/size: {member.name}")
        names = {member.name for member in members}
        if len(names) != len(members) or MANIFEST_NAME not in names:
            raise ValueError("Duplicate archive paths or missing manifest")
        embedded = json.load(tar.extractfile(MANIFEST_NAME))
        if external_manifest is not None and embedded != external_manifest:
            raise ValueError("Embedded manifest differs from separate remote manifest")
        listed = embedded.get("files", {})
        if set(listed) != names - {MANIFEST_NAME}:
            raise ValueError("Archive files and manifest entries differ")
        for name, meta in listed.items():
            safe_member(name)
            source = tar.extractfile(name)
            if source is None:
                raise ValueError(f"Missing archive file: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            total = 0
            with target.open("wb") as out:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
                    total += len(chunk)
                    out.write(chunk)
            if total != meta["bytes"] or h.hexdigest() != meta["sha256"]:
                raise ValueError(f"File hash/size mismatch: {name}")
        (destination / MANIFEST_NAME).write_text(json.dumps(embedded, indent=2) + "\n")
    return {"sha256": actual, "bytes": archive.stat().st_size, "files": len(listed)}


def install_runs(extracted: Path, root: Path, *, require_complete: bool) -> list[str]:
    if require_complete:
        required = [f"runs/{NEW}/best_adapter/adapter_model.safetensors",
                    f"runs/{NEW}/run_config.json",
                    f"runs/{NEW}/train_summary.json"]
        required += [f"runs/{model}/{name}.json" for model in (OLD, NEW)
                     for name in ("v4_human_calibration", "v4_synthetic_val",
                                  "v4_prior_synthetic_val", "v4_human_locked_test",
                                  "v4_realistic_locked_test")]
        required.append("span_v4_status_snapshot.json")
        if any(not (extracted / name).is_file() for name in required):
            raise ValueError("Successful export lacks v4 adapter or frozen evaluations")
    installed = []

    def relocate(value):
        if isinstance(value, str) and (value == REMOTE or value.startswith(REMOTE + "/")):
            return str(root) + value[len(REMOTE):]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value

    for source in (extracted / "runs").rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(extracted)
        run_name = relative.parts[1]
        filename = relative.parts[-1]
        # Keep old v3 run config, metrics, summary, and adapter unchanged.
        if run_name == OLD and not filename.startswith("v4_"):
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if run_name == NEW and filename == "run_config.json":
            original = target.with_name("run_config.remote.json")
            if original.exists() and digest(original) != digest(source):
                raise ValueError("Differing remote run-config backup already exists")
            if not original.exists():
                shutil.copy2(source, original)
            relocated = relocate(json.loads(source.read_text()))
            if target.exists():
                if json.loads(target.read_text()) != relocated:
                    raise ValueError("Refusing to overwrite differing local run configuration")
                continue
            atomic_json(target, relocated)
            installed.append(str(relative))
            continue
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError(f"Refusing to overwrite differing local artifact: {relative}")
            continue
        shutil.copy2(source, target)
        installed.append(str(relative))
    return installed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path,
                        default=Path("/mnt/f/pangram-at-home/vast_span_v4/instance.json"))
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    parser.add_argument("--cli", default="/home/ubuntu/.venvs/pangram-vast-cli/bin/vastai")
    parser.add_argument("--local-root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    args = parser.parse_args()
    if args.interval_seconds < 5:
        parser.error("Polling interval must be at least five seconds")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    if not os.getenv("VAST_API_KEY"):
        raise SystemExit("VAST_API_KEY is missing")
    state = json.loads(args.state.read_text())
    instance = int(state["creation"]["new_contract"])
    connection = state["connection"]
    host = (state.get("direct_host") or connection.get("direct_host") or
            connection.get("ssh_host") or connection.get("public_ipaddr"))
    port = int(state.get("direct_port") or connection.get("direct_port") or
               connection["ssh_port"])
    hourly = float(state.get("billed_dph_total") or state["offer"]["dph_total"])
    created = float(state["created_at_unix"])
    max_hours = float(state.get("max_rental_hours", 8))
    max_cost = float(state.get("budget_dollars", 5))
    output = args.state.parent
    progress_file = output / "monitor_status.json"
    progress = {"instance": instance, "state": "monitoring", "remote_phase": None,
                "started_at_utc": datetime.now(timezone.utc).isoformat()}

    def save() -> None:
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_hours"] = round((time.time() - created) / 3600, 3)
        progress["estimated_rental_usd"] = round(progress["elapsed_hours"] * hourly, 3)
        atomic_json(progress_file, progress)

    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
           "-o", "ServerAliveCountMax=2", "-i", str(args.key), "-p", str(port),
           f"root@{host}"]
    rsync = ["rsync", "-a", "--partial", "--append-verify", "--quiet",
             "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
             f"-o ConnectTimeout=10 -i {args.key} -p {port}"]

    def remote_json(filename: str) -> dict | None:
        try:
            result = call(ssh + [f"cat {REMOTE}/{filename}"], timeout=30)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

    def transfer(filename: str, required: bool = True, attempts: int = 5,
                 timeout: int = 1800) -> Path | None:
        target = output / filename
        for attempt in range(attempts):
            try:
                result = call(rsync + [f"root@{host}:{REMOTE}/{filename}", str(target)],
                              timeout=timeout)
                if result.returncode == 0:
                    return target
            except subprocess.TimeoutExpired:
                pass
            if attempt < attempts - 1:
                time.sleep(10)
        if required:
            raise RuntimeError(f"Could not copy {filename}; preserving instance")
        return None

    def vast_action(action: str) -> None:
        command = [args.cli, "--raw", action, "instance", str(instance)]
        if action == "destroy":
            command.append("-y")
        result = call(command, timeout=90)
        if result.returncode:
            raise RuntimeError(f"Vast {action} failed; exit {result.returncode}")
        progress["vast_action"] = action
        save()
        print(f"Vast instance {instance}: {action} succeeded", flush=True)

    save()
    last_phase = None
    last_update = 0.0
    try:
        while True:
            elapsed = (time.time() - created) / 3600
            cost = elapsed * hourly
            if elapsed >= max_hours or cost >= max_cost:
                progress["state"] = "rental_cap_collecting"
                save()
                print(f"Rental cap reached ({elapsed:.2f} h, ~${cost:.2f}); collecting available files", flush=True)
                collected = []
                # Bound all transfer attempts so the cap cannot turn into hours
                # of additional rental. Partial rsync data remains for recovery.
                for filename in (STATUS_NAME, MANIFEST_NAME, ARCHIVE_NAME,
                                 "span_v4_train.log", "span_v4_driver.log"):
                    if transfer(filename, required=False, attempts=1, timeout=15):
                        collected.append(filename)
                progress["collected_at_cap"] = collected
                if {STATUS_NAME, MANIFEST_NAME, ARCHIVE_NAME} <= set(collected):
                    try:
                        terminal_status = json.loads((output / STATUS_NAME).read_text())
                        if terminal_status.get("phase") in TERMINAL:
                            export = terminal_status.get("export", {})
                            info = verify_and_extract(output / ARCHIVE_NAME, output / "export",
                                                      export["archive_sha256"],
                                                      json.loads((output / MANIFEST_NAME).read_text()))
                            installed = install_runs(output / "export", args.local_root,
                                                     require_complete=terminal_status["phase"] == "complete")
                            progress.update(export=info, installed_files=installed)
                    except (KeyError, ValueError, OSError, tarfile.TarError) as error:
                        progress["cap_export_check"] = f"unverified: {error}"
                progress["state"] = "rental_cap_stopping"
                save()
                vast_action("stop")
                progress["state"] = "stopped_for_recovery"
                save()
                return
            remote_status = remote_json(STATUS_NAME)
            if remote_status:
                phase = remote_status.get("phase", "unknown")
                progress["remote_phase"] = phase
                if phase != last_phase or time.time() - last_update >= 300:
                    print(f"Remote phase: {phase}; elapsed {elapsed:.2f} h; ~${cost:.2f}", flush=True)
                    last_phase, last_update = phase, time.time()
                    save()
                if phase in TERMINAL:
                    export = remote_status.get("export", {})
                    if not export.get("archive_sha256"):
                        raise RuntimeError("Terminal status lacks export hash; preserving instance")
                    try:
                        remote_sum = call(ssh + [f"sha256sum {REMOTE}/{ARCHIVE_NAME}"], timeout=60)
                    except subprocess.TimeoutExpired:
                        time.sleep(args.interval_seconds)
                        continue
                    if (remote_sum.returncode or not remote_sum.stdout.split() or
                            remote_sum.stdout.split()[0] != export["archive_sha256"]):
                        raise RuntimeError("Remote archive hash differs from terminal status; preserving instance")
                    progress["state"] = "collecting"
                    save()
                    transfer(STATUS_NAME)
                    manifest_path = transfer(MANIFEST_NAME)
                    archive_path = transfer(ARCHIVE_NAME)
                    transfer("span_v4_train.log", required=False)
                    transfer("span_v4_driver.log", required=False, attempts=1, timeout=20)
                    embedded = json.loads(manifest_path.read_text())
                    extracted = output / "export"
                    info = verify_and_extract(archive_path, extracted,
                                              export["archive_sha256"], embedded)
                    installed = install_runs(extracted, args.local_root,
                                             require_complete=(phase == "complete"))
                    progress.update(state="verified", export=info, installed_files=installed,
                                    remote_phase=phase)
                    save()
                    vast_action("destroy")
                    progress["state"] = "closed_complete" if phase == "complete" else "closed_failed"
                    save()
                    if phase == "complete":
                        try:
                            report = call([os.sys.executable,
                                           str(Path(__file__).resolve().parent / "report_span_v4.py"),
                                           "--root", str(args.local_root),
                                           "--out", str(Path(__file__).resolve().parent.parent / "reports")],
                                          timeout=180)
                            progress["report_generated"] = report.returncode == 0
                            if report.returncode:
                                progress["report_error"] = (report.stderr or report.stdout)[-500:]
                        except (OSError, subprocess.TimeoutExpired) as error:
                            progress["report_generated"] = False
                            progress["report_error"] = str(error)
                        save()
                        if progress["report_generated"]:
                            print("Generated span v4 comparison report", flush=True)
                    return
            time.sleep(args.interval_seconds)
    except Exception as error:
        progress["state"] = "error_instance_preserved"
        progress["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
