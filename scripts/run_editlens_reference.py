"""Evaluate the two gated EditLens reference checkpoints on our frozen splits.

Reference models and benchmark are CC BY-NC-SA; scores are research diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from run_baselines import metrics, threshold_for_fpr


class NormedLinear(torch.nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, values):
        return self.linear(self.norm(values))


def clean_text(value: str) -> str:
    # Follows the source repository's input cleaning for these prose splits.
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    paragraphs = [p for p in value.split("\n") if p.strip()]
    if len(paragraphs) > 1 and paragraphs[0].lstrip("*# ").startswith(
        ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")
    ):
        value = "\n".join(paragraphs[1:])
    return " ".join(value.lower().split())


def load_model(root: Path, kind: str):
    if kind == "roberta":
        path = root / "models" / "editlens_roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16).to("cuda")
    else:
        base = root / "models" / "Llama-3.2-3B"
        adapter = root / "models" / "editlens_Llama-3.2-3B"
        tokenizer = AutoTokenizer.from_pretrained(base)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        with safe_open(adapter / "adapter_model.safetensors", framework="pt") as file:
            n_buckets = file.get_slice("base_model.model.score.linear.weight").get_shape()[0]
        model = AutoModelForSequenceClassification.from_pretrained(
            base,
            num_labels=n_buckets,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.score = NormedLinear(model.config.hidden_size, n_buckets, device=model.device, dtype=torch.bfloat16)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def score(texts: list[str], tokenizer, model, batch_size: int) -> np.ndarray:
    outputs = []
    n_buckets = model.config.num_labels
    bucket_weights = torch.arange(n_buckets, device="cuda", dtype=torch.float32) / (n_buckets - 1)
    for start in range(0, len(texts), batch_size):
        batch = [clean_text(t) for t in texts[start:start + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=510, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            values = (torch.softmax(logits, dim=-1) * bucket_weights).sum(dim=-1)
        outputs.extend(values.cpu().tolist())
        if start % (batch_size * 50) == 0:
            print(f"scored {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(outputs)


def load(path: Path):
    data = pq.read_table(path, columns=["text", "label"]).to_pydict()
    return data["text"], np.asarray(data["label"], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--model", choices=["roberta", "llama"], required=True)
    parser.add_argument("--tier", choices=["small", "full"], default="small")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    tokenizer, model = load_model(args.root, args.model)
    batch_size = args.batch_size or (16 if args.model == "roberta" else 4)
    data = args.root / "data" / "editlens_pyramid_v1"
    val_text, val_label = load(data / f"val_{args.tier}.parquet")
    val_scores = score(val_text, tokenizer, model, batch_size)
    threshold = threshold_for_fpr(val_scores, val_label, 0.02)
    result = {
        "model": args.model, "tier": args.tier,
        "target_val_fpr": 0.02,
        "val": metrics(val_scores, val_label, threshold),
    }
    for split in ["test", "test_enron"]:
        test_text, test_label = load(data / f"{split}_{args.tier}.parquet")
        test_scores = score(test_text, tokenizer, model, batch_size)
        result[split] = metrics(test_scores, test_label, threshold)
    out = args.root / "results" / f"reference_{args.model}_{args.tier}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
