"""One backward/optimizer step at the selected HPO microbatch size.

This checks memory fit only; it does not train or save a checkpoint.
Run with no other project GPU process active.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--repeat2", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    free_before, total = torch.cuda.mem_get_info()
    config = json.loads((args.root / "runs/vast_hpo_selected_v3/run_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2, dtype=torch.bfloat16, device_map={"": 0})
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    source = tokenizer("The author explains a detailed observation. " * 200,
                       truncation=True, max_length=512)["input_ids"]
    assert len(source) == 512
    tokens = source + source if args.repeat2 else source
    batch = {"input_ids": torch.tensor([tokens] * args.batch_size, device="cuda"),
             "attention_mask": torch.ones((args.batch_size, len(tokens)), dtype=torch.long, device="cuda"),
             "labels": torch.tensor([0, 1] * ((args.batch_size + 1) // 2), device="cuda")[:args.batch_size]}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    torch.cuda.reset_peak_memory_stats()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({"repeat2": args.repeat2, "microbatch": args.batch_size,
                      "tokens_per_example": len(tokens), "loss": float(loss.item()),
                      "gpu_total_gib": round(total / 2**30, 3),
                      "gpu_free_before_model_gib": round(free_before / 2**30, 3),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}))


if __name__ == "__main__":
    main()
