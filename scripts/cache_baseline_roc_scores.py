"""Reproduce raw held-out scores for the five baseline families used in ROC plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from run_baselines import fit_embedding, fit_ngrams


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--dataset", choices=["mixed", "paper"], required=True)
    parser.add_argument("--model", choices=["char", "char_full", "word", "embedding", "roberta", "llama"], required=True)
    args = parser.parse_args()
    data = args.root / "data" / f"{args.dataset}_pyramid_v1"
    train_tier = "full" if args.dataset == "paper" or args.model == "char_full" else "medium"
    train_path = data / f"train_{train_tier}.parquet"
    test_path = data / "test_full.parquet"
    cache = args.root / "runs" / "roc_cache_v1"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{args.dataset}_{args.model}.npz"
    metadata = cache / f"{args.dataset}_{args.model}.json"
    if output.exists():
        assert metadata.exists(), f"Missing metadata for {output}"
        old = json.loads(metadata.read_text())
        assert old["test_sha256"] == sha256(test_path)
        if args.model not in {"roberta", "llama"}:
            assert old["train_sha256"] == sha256(train_path)
        print(f"Using {output}")
        return
    test = pq.read_table(test_path, columns=["text", "label", "source", "text_id"]).to_pydict()
    if args.model in {"roberta", "llama"}:
        from run_editlens_reference import load_model, score
        tokenizer, model = load_model(args.root, args.model)
        scores = score(test["text"], tokenizer, model, 16 if args.model == "roberta" else 4)
    else:
        train = pq.read_table(train_path, columns=["text", "label"]).to_pydict()
        if args.model == "embedding":
            predict = fit_embedding(train["text"], np.asarray(train["label"]), args.root)
        else:
            predict = fit_ngrams("char" if args.model in {"char", "char_full"} else "word", train["text"], np.asarray(train["label"]))
        scores = predict(test["text"])
    np.savez_compressed(
        output, score=np.asarray(scores, dtype=np.float64),
        label=np.asarray(test["label"], dtype=np.int8),
        source=np.asarray(test["source"], dtype=str),
        text_id=np.asarray(test["text_id"], dtype=str),
    )
    metadata.write_text(json.dumps({
        "dataset": args.dataset, "model": args.model, "train_tier": train_tier if args.model not in {"roberta", "llama"} else None,
        "train_sha256": sha256(train_path) if args.model not in {"roberta", "llama"} else None,
        "test_sha256": sha256(test_path), "rows": len(scores),
    }, indent=2) + "\n")
    print(f"Wrote {output} ({len(scores)} scores)", flush=True)


if __name__ == "__main__":
    main()
