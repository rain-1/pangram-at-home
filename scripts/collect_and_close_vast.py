"""Verify the completed Vast sweep export on F:, then destroy its rental."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def call(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, **kwargs)


def checksum(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--instance",type=int,default=52593558)
    p.add_argument("--host",default="175.155.64.161")
    p.add_argument("--port",type=int,default=19970)
    p.add_argument("--output",type=Path,default=Path("/mnt/f/pangram-at-home/vast_results_v1"))
    p.add_argument("--max-wait-hours",type=float,default=3)
    args=p.parse_args()
    if not os.getenv("VAST_API_KEY"):
        raise SystemExit("VAST_API_KEY missing")
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/"collection_status.json"
    state={"instance":args.instance,"state":"waiting_for_sweep"}
    def save():state_path.write_text(json.dumps(state,indent=2)+"\n")
    ssh=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","-o","ServerAliveInterval=60",
         "-o","ServerAliveCountMax=3","-i",str(Path.home()/".ssh/id_ed25519"),"-p",str(args.port),f"root@{args.host}"]
    remote="/workspace/pangram-data"
    deadline=time.monotonic()+args.max_wait_hours*3600
    save()
    while True:
        probe=call(ssh+[f"test -f {remote}/HPO_DONE && echo done || (test -f {remote}/HPO_FAILED && echo failed || echo running)"])
        if probe.returncode==0 and probe.stdout.strip() in {"done","failed"}:
            if probe.stdout.strip()=="failed":
                state["state"]="remote_failed_kept_for_debugging";save()
                raise RuntimeError("Remote pipeline failed; instance kept for debugging")
            break
        if time.monotonic()>=deadline:
            state["state"]="wait_timeout_kept_for_debugging";save()
            raise TimeoutError("Sweep still running; instance kept for debugging")
        time.sleep(30)
    state["state"]="copying";save();print("Sweep done; copying export",flush=True)
    remote_hash=call(ssh+[f"sha256sum {remote}/hpo_export_v1.tar.gz"],check=True).stdout.split()[0]
    archive=args.output/"hpo_export_v1.tar.gz"
    src=f"root@{args.host}:{remote}/hpo_export_v1.tar.gz"
    rsync=["rsync","-a","--partial","--append-verify","--info=progress2",
           "-e",f"ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -i {Path.home() / '.ssh/id_ed25519'} -p {args.port}"]
    for attempt in range(1,9):
        result=subprocess.run(rsync+[src,str(archive)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        if result.returncode==0:break
        print("Transfer attempt",attempt,"failed:",result.stderr[-250:],flush=True)
        if attempt==8:raise RuntimeError("Could not copy archive; instance kept")
        time.sleep(15)
    local_hash=checksum(archive)
    if remote_hash!=local_hash:raise RuntimeError("Export SHA256 mismatch; instance kept")
    with tarfile.open(archive,"r:gz") as tar:
        names=set(tar.getnames())
        hpo={n for n in names if n.startswith("runs/hpo_diverse_v3_") and n.endswith("best_adapter/adapter_model.safetensors")}
        ablation={n for n in names if n.startswith("runs/ablation_diverse_v3_") and n.endswith("best_adapter/adapter_model.safetensors")}
        required={"results/hpo_diverse_v3_summary.json","results/hpo_diverse_v3_best_config.json",
                  "results/ablation_diverse_v3_summary.json"}
        if len(hpo)<3 or len(ablation)!=9 or not required<=names:
            raise RuntimeError(f"Incomplete export: {len(hpo)} HPO adapters, {len(ablation)} ablation adapters, missing {required-names}")
        summary=json.load(tar.extractfile("results/ablation_diverse_v3_summary.json"))
        hsummary=json.load(tar.extractfile("results/hpo_diverse_v3_summary.json"))
        if len(summary)!=9 or len(hsummary)!=24 or any(x["error"] for x in summary+hsummary):
            raise RuntimeError("Sweep summaries incomplete or contain trial failures")
    state.update(state="verified",archive_sha256=local_hash,archive_bytes=archive.stat().st_size,
                 hpo_adapters=len(hpo),ablation_adapters=len(ablation))
    save();print("Verified",len(hpo),"HPO and",len(ablation),"ablation adapters",flush=True)
    for name in ("hpo_driver.log","ablation_driver.log","hpo_selection.log","hpo_export.log"):
        source=f"root@{args.host}:{remote}/{name}"
        for attempt in range(3):
            result=subprocess.run(rsync+[source,str(args.output/name)],stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE,text=True)
            if result.returncode==0:break
            if attempt==2:raise RuntimeError(f"Could not copy {name}; instance kept")
            time.sleep(10)
    cli="/home/ubuntu/.venvs/pangram-vast-cli/bin/vastai"
    response=call([cli,"--raw","destroy","instance",str(args.instance),"-y"])
    if response.returncode:
        state["state"]="verified_but_close_failed";save()
        raise RuntimeError("Archive verified, but Vast destroy request failed: "+response.stderr[-300:])
    state["state"]="closed";state["closed_at_unix"]=time.time();save()
    print("Vast instance closed",args.instance,flush=True)


if __name__=="__main__":main()
