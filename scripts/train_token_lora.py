"""Binary token localization with Repeat2 and second-copy-only supervision."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)
from span_data import SpanDataset
from train_segment_lora import DeadlineCallback, SavedEvalCallback, RunMetadataCallback


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    p.add_argument("--model",type=Path,default=Path("/mnt/f/pangram-at-home/models/Qwen3-1.7B"))
    p.add_argument("--init-adapter",type=Path)
    p.add_argument("--run-name",default="qwen3_token_repeat2_pilot_v1")
    p.add_argument("--max-steps",type=int,default=800);p.add_argument("--eval-steps",type=int,default=200)
    p.add_argument("--learning-rate",type=float,default=7.607757094022466e-5)
    p.add_argument("--lora-rank",type=int,default=32);p.add_argument("--lora-alpha",type=int,default=64)
    p.add_argument("--lora-dropout",type=float,default=.068837366330751)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accumulation",type=int,default=8)
    p.add_argument("--hours",type=float,default=3);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--report-to",choices=["none","wandb"],default="wandb")
    p.add_argument("--single-copy",action="store_true")
    args=p.parse_args();set_seed(args.seed);torch.set_num_threads(4)
    output=args.root/"runs"/args.run_name
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model);tokenizer.pad_token=tokenizer.eos_token
    folder=args.root/"data/span_pilot_v1"
    train=SpanDataset(folder/"train.jsonl",tokenizer,repeat2=not args.single_copy)
    val=SpanDataset(folder/"val.jsonl",tokenizer,repeat2=not args.single_copy)
    model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=2,dtype=torch.bfloat16,
        device_map={"":0},classifier_dropout=0.1,
        id2label={0:"human",1:"ai_generated"},label2id={"human":0,"ai_generated":1})
    model.config.use_cache=False;model.config.pad_token_id=tokenizer.pad_token_id
    model=get_peft_model(model,LoraConfig(task_type=TaskType.TOKEN_CLS,r=args.lora_rank,lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],modules_to_save=["score"]))
    init_sha=None
    if args.init_adapter:
        adapter_config=json.loads((args.init_adapter/"adapter_config.json").read_text())
        if adapter_config["r"]!=args.lora_rank or adapter_config["lora_alpha"]!=args.lora_alpha:
            raise ValueError("Initialization adapter rank/alpha must match")
        weights=args.init_adapter/"adapter_model.safetensors"
        # Retain the newly initialized token head (including its bias), and
        # transfer only the trained backbone adapters from sequence classification.
        state=get_peft_model_state_dict(model)
        transfer={k:v for k,v in load_file(weights).items() if "lora_" in k}
        if set(transfer) != {k for k in state if "lora_" in k}:
            raise ValueError("Source and target LoRA module keys differ")
        state.update(transfer)
        loaded=set_peft_model_state_dict(model,state)
        if any("lora_" in k for k in loaded.missing_keys) or loaded.unexpected_keys:
            raise RuntimeError(f"Adapter transfer mismatch: {loaded}")
        init_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    config={**vars(args),"task":"binary_token_classification","repeat2":not args.single_copy,
        "max_source_tokens":512,"stride":256,"first_copy_loss_masked":not args.single_copy,
        "classifier_dropout":0.1,"quantization":"none","effective_batch_size":args.batch_size*args.accumulation,
        "assisted_supported":False,"train_windows":len(train),"val_windows":len(val),
        "init_adapter_sha256":init_sha,"train_sha256":hashlib.sha256((folder/"train.jsonl").read_bytes()).hexdigest(),
        "val_sha256":hashlib.sha256((folder/"val.jsonl").read_bytes()).hexdigest()}
    config=json.loads(json.dumps(config,default=str))
    (output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    def metrics(pred):
        logits,labels=pred;mask=labels!=-100;scores=(logits[...,1]-logits[...,0])[mask];y=labels[mask]
        return {"roc_auc":roc_auc_score(y,scores),"partial_auc_fpr_5pct":roc_auc_score(y,scores,max_fpr=.05),
                "token_accuracy_at_zero":float(np.mean((scores>=0)==y))}
    training=TrainingArguments(output_dir=str(output),run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation,max_steps=args.max_steps,
        learning_rate=args.learning_rate,lr_scheduler_type="cosine",warmup_ratio=.05,weight_decay=.01,
        bf16=True,gradient_checkpointing=True,eval_strategy="steps",save_strategy="steps",
        eval_steps=args.eval_steps,save_steps=args.eval_steps,save_total_limit=2,load_best_model_at_end=True,
        metric_for_best_model="partial_auc_fpr_5pct",greater_is_better=True,logging_steps=20,
        report_to=args.report_to,seed=args.seed,remove_unused_columns=False)
    trainer=Trainer(model=model,args=training,train_dataset=train,eval_dataset=val,
        data_collator=DataCollatorForTokenClassification(tokenizer,pad_to_multiple_of=8,label_pad_token_id=-100),
        compute_metrics=metrics,callbacks=[DeadlineCallback(time.monotonic()+args.hours*3600),SavedEvalCallback(output/"sweep_metrics.jsonl"),RunMetadataCallback(config)])
    training_result=trainer.train();trainer.save_model(output/"best_adapter");tokenizer.save_pretrained(output/"best_adapter")
    (output/"train_summary.json").write_text(json.dumps({"best_metric":trainer.state.best_metric,
        "best_checkpoint":trainer.state.best_model_checkpoint,"global_step":trainer.state.global_step,
        "train_runtime_seconds":training_result.metrics.get("train_runtime"),
        "peak_allocated_gb":torch.cuda.max_memory_allocated()/2**30},indent=2)+"\n")
    if args.report_to=="wandb":
        import wandb
        if wandb.run is not None:wandb.finish()


if __name__=="__main__":main()
