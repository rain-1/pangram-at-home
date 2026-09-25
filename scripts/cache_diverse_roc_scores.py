"""Recreate per-example baseline scores for ROC plots on frozen diverse tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from run_baselines import fit_embedding, fit_ngrams


ROOT = Path("/mnt/f/pangram-at-home")
REPO = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["char", "word", "embedding", "roberta", "llama"], required=True)
    args = parser.parse_args()
    name = args.model
    data = ROOT / "data/diverse_pyramid_v1"
    train_path = data / "train_full.parquet"
    test_paths = {"test": data / "test_full.parquet",
                  "raid_external": ROOT / "data/raid_external_v1/frozen.parquet"}
    if name in {"char", "word", "embedding"}:
        train = pq.read_table(train_path, columns=["text", "label"]).to_pydict()
        labels = np.asarray(train["label"], dtype=int)
        predict = (fit_embedding(train["text"], labels, ROOT) if name == "embedding"
                   else fit_ngrams(name, train["text"], labels))
        metrics_path = REPO / f"reports/metrics/baseline_diverse_{name}_full.json"
    else:
        import torch
        from run_editlens_reference import load_model, score

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for EditLens references")
        tokenizer, model = load_model(ROOT, name)
        batch_size = 16 if name == "roberta" else 4
        predict = lambda texts: score(texts, tokenizer, model, batch_size)
        metrics_path = REPO / f"reports/metrics/reference_diverse_{name}_full.json"
    expected = json.loads(metrics_path.read_text())
    out_dir = ROOT / "runs/diverse_roc_scores_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, path in test_paths.items():
        rows = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
        scores = np.asarray(predict(rows["text"]), dtype=np.float32)
        labels = np.asarray(rows["label"], dtype=np.int8)
        auc = float(roc_auc_score(labels, scores))
        published = expected[split]["roc_auc"]
        if abs(auc - published) > 0.001:
            raise RuntimeError(f"{name}/{split} AUROC {auc:.6f} differs from published {published:.6f}")
        output = out_dir / f"{name}_{split}.npz"
        np.savez_compressed(output, score=scores, label=labels,
                            source=np.asarray(rows["source"], dtype=str),
                            input_sha256=file_hash(path), published_auc=published)
        print(f"{name}/{split}: AUROC {auc:.6f}, cached {output}", flush=True)


if __name__ == "__main__":
    main()
