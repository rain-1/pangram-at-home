"""Generate title-matched AI abstracts from licensed pre-2023 papers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "Qwen2.5-0.5B-Instruct"),
    "smollm": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", "SmolLM2-1.7B-Instruct"),
}
PROMPT_VERSION = "paper-title-abstract-v1"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt(doc: dict) -> str:
    target = len(doc["abstract"].split())
    return (
        f"Write an original scientific research abstract of about {target} words. "
        "Use clear academic prose with motivation, methods, findings, and limitations. "
        "Write only the abstract, without a title, headings, or commentary. "
        "Do not quote or reproduce an existing abstract.\n\n"
        f"Research topic: {doc['title']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--source", choices=["pmc", "acl"], default="pmc")
    parser.add_argument("--model", choices=list(MODELS), default="qwen")
    parser.add_argument("--limit", type=int, default=0, help="For a small smoke run; 0 means all eligible papers")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    source_dir = args.root / "data" / ("pmc_pilot_v1" if args.source == "pmc" else "acl_abstracts_v1")
    input_path = source_dir / "documents.jsonl.gz"
    output_path = source_dir / f"generated_{args.model}.jsonl"
    model_name, model_revision, model_dir = MODELS[args.model]
    model_path = args.root / "models" / model_dir
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this generation run")
    docs = [json.loads(line) for line in gzip.open(input_path, "rt", encoding="utf-8")]
    docs = [d for d in docs if 120 <= len(d["abstract"].split()) <= 280 and d["title"]]
    docs.sort(key=lambda d: sha(f'{args.source}-selection-v1:{d["source_id"]}'))
    if args.source == "acl":
        by_venue = {}
        diverse = []
        for doc in docs:
            venue = doc["venue"]
            if by_venue.get(venue, 0) < 30:
                diverse.append(doc)
                by_venue[venue] = by_venue.get(venue, 0) + 1
        docs = diverse
    if args.limit:
        docs = docs[:args.limit]
    done = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            done.add(json.loads(line)["source_id"])
    docs = [d for d in docs if d["source_id"] not in done]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    for offset in range(0, len(docs), args.batch_size):
        batch = docs[offset:offset + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(doc)}],
                tokenize=False, add_generation_prompt=True,
            )
            for doc in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        seed = int(sha(f"{PROMPT_VERSION}:{args.model}:{offset}")[:8], 16)
        torch.manual_seed(seed)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=400, do_sample=True,
                temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        with output_path.open("a", encoding="utf-8") as file:
            for doc, generated in zip(batch, decoded):
                generated = re.sub(r"^\s*(abstract\s*[:\-]\s*)", "", generated.strip(), flags=re.I)
                record = {
                    "source_id": doc["source_id"],
                    "model": model_name,
                    "model_revision": model_revision,
                    "prompt_version": PROMPT_VERSION,
                    "prompt": prompt(doc),
                    "seed": seed,
                    "text": generated,
                    "text_sha256": sha(generated),
                    "human_abstract_sha256": doc["abstract_sha256"],
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"generated {offset + len(batch)} / {len(docs)} remaining", flush=True)
    print(output_path)


if __name__ == "__main__":
    main()
