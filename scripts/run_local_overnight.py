"""Bounded local sequence Repeat2 comparison followed by a token pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    p.add_argument("--hours",type=float,default=10);p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();root=args.root
    lock=open("/tmp/pangram_local_overnight.lock","w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    best=json.loads((root/"vast_results_v1/live/hpo_diverse_v3_best_config.json").read_text())
    adapter=root/"runs"/best["source_run"]/"best_adapter"
    common=["--root",str(root),"--model",str(root/"models/Qwen3-1.7B"),
            "--learning-rate",str(best["learning_rate"]),"--lora-rank",str(best["lora_rank"]),
            "--lora-alpha",str(2*best["lora_rank"]),"--lora-dropout",str(best["lora_dropout"]),
            "--report-to","wandb"]
    jobs=[]
    for repeat,name in ((False,"qwen3_hpo_single_local_v1"),(True,"qwen3_hpo_repeat2_local_v1")):
        command=[sys.executable,"-u","scripts/train_segment_lora.py",*common,"--run-name",name,
            "--dataset-folder","diverse_pyramid_v1","--train-tier","full","--quantization","none",
            "--max-steps","3200","--eval-steps","400","--train-batch-size","1",
            "--gradient-accumulation-steps","8","--eval-batch-size","1","--disable-early-stopping",
            "--selection-metric","partial_auc_fpr_5pct","--metrics-jsonl",str(root/"runs"/name/"sweep_metrics.jsonl")]
        if repeat:command.append("--repeat2")
        jobs.append((name,command,"sequence"))
    name="qwen3_token_repeat2_pilot_v1"
    jobs.append((name,[sys.executable,"-u","scripts/train_token_lora.py",*common,"--run-name",name,
        "--init-adapter",str(adapter),"--max-steps","800","--eval-steps","200","--batch-size","1","--accumulation","8"],"token"))
    if args.dry_run:
        print(json.dumps(jobs,indent=2));return
    if not os.getenv("WANDB_API_KEY"):raise SystemExit("WANDB_API_KEY required")
    for name,_,_ in jobs:
        if (root/"runs"/name).exists():raise SystemExit(f"Run already exists: {name}")
    if not (adapter/"adapter_config.json").exists():raise SystemExit("Initialization adapter download incomplete")
    os.environ.update(WANDB_PROJECT="pangram-at-home",WANDB_RUN_GROUP="local_repeat2_span_v1",
        WANDB_DIR=str(root/"wandb"),TOKENIZERS_PARALLELISM="false",CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    (root/"wandb").mkdir(exist_ok=True)
    logdir=root/"overnight_repeat2_span_v1";logdir.mkdir(exist_ok=False)
    started=time.time();deadline=time.monotonic()+args.hours*3600
    status={"started":started,"max_hours":args.hours,"jobs":[],"state":"running"}
    def save():
        tmp=logdir/"status.tmp";tmp.write_text(json.dumps(status,indent=2)+"\n");tmp.replace(logdir/"status.json")
    def execute(command,name):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("Overnight wall-clock budget exhausted")
        with (logdir/f"{name}.log").open("w") as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=remaining)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if code:raise RuntimeError(f"{name} exited {code}; see its log")
    try:
        for name,command,task in jobs:
            remaining=deadline-time.monotonic()
            if remaining<600:raise TimeoutError("Insufficient time for another training stage")
            status["current_job"]=name;save();print("Starting",name,flush=True)
            execute(command+["--hours",str(max(.05,(remaining-300)/3600))],name)
            summary=json.loads((root/"runs"/name/"train_summary.json").read_text())
            status["jobs"].append({"name":name,"training":summary});save()
        # Paired passage models also provide coarse window-level localization baselines.
        for name,_,task in jobs:
            status["current_job"]=name+"_span_validation";save()
            execute([sys.executable,"-u","scripts/evaluate_span_pilot.py","--root",str(root),
                     "--run-name",name,"--task",task,"--report-to","wandb"],name+"_span_validation")
        status["state"]="complete"
    except Exception as exc:
        status["state"]="stopped";status["error"]=str(exc);raise
    finally:
        status["elapsed_hours"]=(time.time()-started)/3600;save()
        subprocess.run([sys.executable,"scripts/summarize_overnight.py","--root",str(root)],cwd=REPO)


if __name__=="__main__":main()
