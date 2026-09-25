"""Train a Pangram-4-inspired stage-1 segment classifier on the frozen mixed split.

Only binary whole-document labels are available, so this run trains a binary
last-token head. Tokenwise and edit-fraction objectives need separate labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class TextDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        columns = ["text", "label"]
        if "domain" in pq.read_schema(path).names:
            columns.append("domain")
        rows = pq.read_table(path, columns=columns).to_pydict()
        self.labels = rows["label"]
        self.domains = rows.get("domain")
        self.encodings = tokenizer(rows["text"], truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}


class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_save = True
            control.should_training_stop = True
        return control


class SavedEvalCallback(TrainerCallback):
    """Publish an evaluation only after its corresponding checkpoint is saved."""

    def __init__(self, path: Path):
        self.path = path
        self.latest_metrics = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.latest_metrics = dict(metrics or {})

    def on_save(self, args, state, control, **kwargs):
        if self.latest_metrics is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"step": state.global_step, **self.latest_metrics}) + "\n")
            self.latest_metrics = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--model", type=Path, default=Path("/mnt/f/pangram-at-home/models/Qwen3-1.7B"))
    parser.add_argument("--run-name", default="qwen3_17b_mixed_stage1_v1")
    parser.add_argument("--dataset-folder", default="mixed_pyramid_v1")
    parser.add_argument("--train-tier", choices=["tiny", "small", "medium", "full"], default="full")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--hours", type=float, default=9.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--selection-metric", choices=["roc_auc", "partial_auc_fpr_5pct"], default="roc_auc")
    parser.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    args = parser.parse_args()

    def finish_on_sigterm(signum, frame):
        """Let ASHA-pruned subprocesses close their W&B run cleanly."""
        if args.report_to == "wandb":
            import wandb
            if wandb.run is not None:
                wandb.run.summary["stopped_by_scheduler"] = True
                wandb.finish(exit_code=0)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, finish_on_sigterm)

    torch.set_num_threads(4)
    set_seed(args.seed)
    folder = args.root / "data" / args.dataset_folder
    output = args.root / "runs" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    if args.metrics_jsonl and args.metrics_jsonl.exists():
        raise SystemExit(f"Metrics path already exists: {args.metrics_jsonl}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    train = TextDataset(folder / f"train_{args.train_tier}.parquet", tokenizer, args.max_length)
    val = TextDataset(folder / "val_full.parquet", tokenizer, args.max_length)
    train_counts = np.bincount(train.labels, minlength=2).tolist()
    assert train_counts[0] == train_counts[1], train_counts
    quantization = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
                    if args.quantization == "nf4" else None)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if quantization is not None:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    ))
    model.print_trainable_parameters()
    config = {
        "base_model": str(args.model), "train_file": str(folder / f"train_{args.train_tier}.parquet"),
        "val_file": str(folder / "val_full.parquet"), "train_counts": train_counts,
        "max_length": args.max_length, "epochs": args.epochs, "hours": args.hours,
        "seed": args.seed, "task": "binary_segment_classification",
        "dataset_folder": args.dataset_folder, "learning_rate": args.learning_rate,
        "report_to": args.report_to, "eval_steps": args.eval_steps,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "weight_decay": 0.01,
        "warmup_ratio": 0.05, "lr_scheduler_type": "cosine",
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "selection_metric": args.selection_metric,
        "quantization": args.quantization,
        "train_sha256": file_sha256(folder / f"train_{args.train_tier}.parquet"),
        "val_sha256": file_sha256(folder / "val_full.parquet"),
        "dataset_manifest_sha256": file_sha256(folder / "manifest.json"),
        "label_0": "human", "label_1": "ai_generated",
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def metrics(pred):
        logits, labels = pred
        scores = logits[:, 1].astype(np.float32) - logits[:, 0].astype(np.float32)
        human_scores = np.sort(scores[labels == 0])[::-1]
        allowed = int(np.floor(.02 * len(human_scores)))
        threshold = np.nextafter(human_scores[allowed], np.inf)
        result = {"roc_auc": roc_auc_score(labels, scores),
                  "partial_auc_fpr_5pct": roc_auc_score(labels, scores, max_fpr=.05),
                  "ai_recall_at_fpr_2pct": float(np.mean(scores[labels == 1] >= threshold))}
        if val.domains is not None:
            domains = np.asarray(val.domains)
            result["worst_domain_ai_recall_at_fpr_2pct"] = float(min(
                np.mean(scores[(labels == 1) & (domains == domain)] >= threshold)
                for domain in set(domains)
            ))
        return result

    training_args = TrainingArguments(
        output_dir=str(output), run_name=args.run_name,
        per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps, num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,
        eval_strategy="steps", save_strategy="steps", eval_steps=args.eval_steps,
        save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model=args.selection_metric, greater_is_better=True,
        logging_steps=20, report_to=args.report_to, dataloader_num_workers=0,
        remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6), DeadlineCallback(time.monotonic() + args.hours * 3600)]
        + ([SavedEvalCallback(args.metrics_jsonl)] if args.metrics_jsonl else []),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output / "best_adapter")
    tokenizer.save_pretrained(output / "best_adapter")
    (output / "train_summary.json").write_text(json.dumps({"best_metric": trainer.state.best_metric, "best_checkpoint": trainer.state.best_model_checkpoint, "global_step": trainer.state.global_step}, indent=2) + "\n")


if __name__ == "__main__":
    main()
