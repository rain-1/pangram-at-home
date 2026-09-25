"""Evaluate the two gated EditLens reference checkpoints on our frozen splits.

Reference models and benchmark are CC BY-NC-SA; scores are research diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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
    data = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
    return data["text"], np.asarray(data["label"], dtype=int), np.asarray(data["source"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--model", choices=["roberta", "llama"], required=True)
    parser.add_argument("--dataset", choices=["editlens", "pmc", "paper", "mixed", "diverse"], default="editlens")
    parser.add_argument("--tier", choices=["small", "full"], default="small")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--audit-acl-limit", type=int, default=0)
    parser.add_argument("--audit-pmc-body", action="store_true")
    parser.add_argument("--cross-test", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    tokenizer, model = load_model(args.root, args.model)
    batch_size = args.batch_size or (16 if args.model == "roberta" else 4)
    folder = {
        "editlens": "editlens_pyramid_v1",
        "pmc": "pmc_pyramid_v1",
        "paper": "paper_pyramid_v1",
        "mixed": "mixed_pyramid_v1",
        "diverse": "diverse_pyramid_v1",
    }[args.dataset]
    data = args.root / "data" / folder
    val_text, val_label, _ = load(data / f"val_{args.tier}.parquet")
    val_scores = score(val_text, tokenizer, model, batch_size)
    threshold = threshold_for_fpr(val_scores, val_label, 0.02)
    result = {
        "model": args.model, "tier": args.tier, "dataset": args.dataset,
        "target_val_fpr": 0.02,
        "val": metrics(val_scores, val_label, threshold),
    }
    for split in (["test", "test_enron"] if args.dataset == "editlens" else ["test"]):
        test_text, test_label, sources = load(data / f"{split}_{args.tier}.parquet")
        test_scores = score(test_text, tokenizer, model, batch_size)
        result[split] = metrics(test_scores, test_label, threshold)
        result[split]["by_source"] = {
            source: metrics(test_scores[sources == source], test_label[sources == source], threshold)
            for source in sorted(set(sources))
            if len(set(test_label[sources == source])) == 2
        }
        if args.dataset == "diverse":
            domains = np.asarray(pq.read_table(data / f"{split}_{args.tier}.parquet", columns=["domain"]).to_pydict()["domain"])
            result[split]["by_domain"] = {
                domain: metrics(test_scores[domains == domain], test_label[domains == domain], threshold)
                for domain in sorted(set(domains))
            }
    if args.dataset == "diverse":
        external_paths = {
            "raid_external": args.root / "data/raid_external_v1/frozen.parquet",
            "enron_external": args.root / "data/editlens_pyramid_v1/test_enron_full.parquet",
            "gpt4_ood": args.root / "data/mage_external_v1/frozen/gpt4_ood.parquet",
            "paraphrase": args.root / "data/mage_external_v1/frozen/paraphrase.parquet",
        }
        for name, path in external_paths.items():
            rows = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
            labels = np.asarray(rows["label"], dtype=int)
            sources = np.asarray(rows["source"])
            scores = score(rows["text"], tokenizer, model, batch_size)
            result[name] = metrics(scores, labels, threshold)
            result[name]["by_domain"] = {
                source_name: metrics(scores[sources == source_name], labels[sources == source_name], threshold)
                for source_name in sorted(set(sources))
            }
            print(name, result[name]["roc_auc"], flush=True)
        audit_paths = {
            "standard_ebooks_human": args.root / "data/standard_ebooks_v1/human.parquet",
            "persuade_essays_human": args.root / "data/persuade_essays_v1/human_eval.parquet",
            "federal_reserve_human": args.root / "data/federal_reserve_beige_book_v1/human.parquet",
            "stackexchange_writers_human": args.root / "data/stackexchange_writers_v1/human_eval.parquet",
            "pmc_full_body_human": args.root / "data/pmc_body_audit_v1/human_test.parquet",
        }
        for name, path in audit_paths.items():
            rows = pq.read_table(path, columns=["text", "text_id"]).to_pydict()
            texts = rows["text"]
            if name in {"persuade_essays_human", "stackexchange_writers_human"}:
                seed = "persuade-audit-v1:" if name == "persuade_essays_human" else "stackexchange-audit-v1:"
                indices = sorted(range(len(texts)), key=lambda i: hashlib.sha256((seed + rows["text_id"][i]).encode()).digest())[:1000]
                texts = [texts[i] for i in indices]
            scores = score(texts, tokenizer, model, batch_size)
            result[name] = {"rows": len(scores), "false_positives": int((scores >= threshold).sum()),
                            "fpr": float((scores >= threshold).mean())}
            print(name, result[name]["fpr"], flush=True)
    if args.audit_acl_limit:
        acl_path = args.root / "data" / "acl_abstracts_v1" / "documents.jsonl.gz"
        with gzip.open(acl_path, "rt", encoding="utf-8") as file:
            acl = [json.loads(line) for line in file]
        acl = [row for row in acl if 100 <= len(row["abstract"].split()) <= 350]
        acl.sort(key=lambda row: hashlib.sha256(("acl-audit-v1:" + row["source_id"]).encode()).digest())
        acl = acl[:args.audit_acl_limit]
        acl_scores = score([row["abstract"] for row in acl], tokenizer, model, batch_size)
        result["acl_human_audit"] = {
            "rows": len(acl),
            "false_positives": int((acl_scores >= threshold).sum()),
            "fpr": float((acl_scores >= threshold).mean()),
        }
    if args.audit_pmc_body:
        if args.dataset not in {"editlens", "pmc"}:
            raise SystemExit("PMC body audit is only work-disjoint from editlens/pmc training")
        body = pq.read_table(args.root / "data" / "pmc_body_audit_v1" / "human_test.parquet", columns=["text"]).to_pydict()["text"]
        body_scores = score(body, tokenizer, model, batch_size)
        result["pmc_body_human_audit"] = {
            "rows": len(body),
            "false_positives": int((body_scores >= threshold).sum()),
            "fpr": float((body_scores >= threshold).mean()),
        }
    if args.cross_test:
        if args.dataset != "paper":
            raise SystemExit("--cross-test requires --dataset paper")
        texts, labels, sources = load(args.root / "data" / "paper_cross_model_test_v1" / "test.parquet")
        scores = score(texts, tokenizer, model, batch_size)
        result["cross_model_test"] = metrics(scores, labels, threshold)
        result["cross_model_test"]["by_source"] = {
            source: metrics(scores[sources == source], labels[sources == source], threshold)
            for source in sorted(set(sources))
        }
    out = args.root / "results" / f"reference_{args.dataset}_{args.model}_{args.tier}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.dataset == "diverse":
        mirror = Path(__file__).resolve().parents[1] / "reports/metrics" / out.name
        mirror.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
