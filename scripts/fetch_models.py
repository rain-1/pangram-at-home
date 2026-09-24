"""Download pinned local generators and gated reference checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import snapshot_download

MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "Qwen2.5-0.5B-Instruct"),
    "smollm": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", "SmolLM2-1.7B-Instruct"),
    "roberta": ("pangram/editlens_roberta-large", "f93e1ace74528cfb48f337ab2fe946fb71a728cb", "editlens_roberta-large"),
    "llama_adapter": ("pangram/editlens_Llama-3.2-3B", "b5f8044f631f5b455eafbcb569dcf175f2b0726d", "editlens_Llama-3.2-3B"),
    "llama_base": ("meta-llama/Llama-3.2-3B", "13afe5124825b4f3751f836b40dafda64c1ed062", "Llama-3.2-3B"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--which", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or dotenv_values(".env").get("HF_API_TOKEN")
    for which in args.which:
        repo, revision, local_name = MODELS[which]
        if which not in {"qwen", "smollm"} and not token:
            raise SystemExit(f"{which} needs HF_TOKEN or HF_API_TOKEN in .env")
        path = args.root / "models" / local_name
        snapshot_download(
            repo_id=repo, revision=revision, token=token,
            local_dir=path, max_workers=4,
            ignore_patterns=["original/*"] if which == "llama_base" else None,
            allow_patterns=[
                "config.json", "generation_config.json", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "merges.txt", "vocab.json",
            ] if which == "smollm" else None,
        )
        print(f"{which}: {path}")


if __name__ == "__main__":
    main()
