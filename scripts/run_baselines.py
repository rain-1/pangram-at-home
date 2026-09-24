"""Train simple binary detectors on the frozen EditLens research pyramid."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline


def load(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    data = pq.read_table(path, columns=["text", "label", "source"]).to_pydict()
    return data["text"], np.asarray(data["label"], dtype=int), data["source"]


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, target: float) -> float:
    human = np.sort(scores[labels == 0])[::-1]
    allowed = int(np.floor(target * len(human)))
    if allowed >= len(human):
        return float("-inf")
    return float(np.nextafter(human[allowed], np.inf))


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    human = labels == 0
    ai = labels == 1
    tp = int((pred & ai).sum())
    fp = int((pred & human).sum())
    tn = int((~pred & human).sum())
    fn = int((~pred & ai).sum())
    return {
        "rows": len(labels),
        "human": int(human.sum()),
        "ai": int(ai.sum()),
        "threshold": threshold,
        "fpr": fp / int(human.sum()),
        "tpr": tp / int(ai.sum()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def fit_ngrams(kind: str, texts: list[str], labels: np.ndarray):
    if kind == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=3,
            max_features=300_000, sublinear_tf=True, dtype=np.float32,
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=3,
            max_features=200_000, sublinear_tf=True, dtype=np.float32,
        )
    model = make_pipeline(vectorizer, LogisticRegression(C=4.0, max_iter=1000))
    model.fit(texts, labels)
    return lambda batch: model.predict_proba(batch)[:, 1]


def fit_embedding(texts: list[str], labels: np.ndarray, root: Path):
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name, cache_folder=str(root / "hf"))
    vector = encoder.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    classifier = LogisticRegression(C=4.0, max_iter=1000)
    classifier.fit(vector, labels)

    def predict(batch: list[str]) -> np.ndarray:
        encoded = encoder.encode(
            batch, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return classifier.predict_proba(encoded)[:, 1]

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--dataset", choices=["editlens", "pmc", "paper", "mixed"], default="editlens")
    parser.add_argument("--train-tier", default="medium")
    parser.add_argument("--model", choices=["char", "word", "embedding"], default="char")
    parser.add_argument("--audit-acl", action="store_true", help="Report FPR on pre-2023 ACL human abstracts")
    args = parser.parse_args()
    root = args.root
    folder = {
        "editlens": "editlens_pyramid_v1",
        "pmc": "pmc_pyramid_v1",
        "paper": "paper_pyramid_v1",
        "mixed": "mixed_pyramid_v1",
    }[args.dataset]
    data = root / "data" / folder
    manifest = json.loads((data / "manifest.json").read_text())
    if args.train_tier not in manifest["splits"]["train"]:
        raise SystemExit(f"Unknown train tier {args.train_tier}; choose from {list(manifest['splits']['train'])}")
    train_texts, train_labels, _ = load(data / f"train_{args.train_tier}.parquet")
    if args.model == "embedding":
        predict = fit_embedding(train_texts, train_labels, root)
    else:
        predict = fit_ngrams(args.model, train_texts, train_labels)
    val_texts, val_labels, _ = load(data / "val_full.parquet")
    val_scores = predict(val_texts)
    threshold = threshold_for_fpr(val_scores, val_labels, 0.02)
    result = {
        "model": args.model,
        "train_tier": args.train_tier,
        "train_rows": len(train_labels),
        "dataset": manifest.get("dataset", manifest.get("source")),
        "revision": manifest.get("revision"),
        "target_val_fpr": 0.02,
        "val": metrics(val_scores, val_labels, threshold),
    }
    for split in (["test", "test_enron"] if args.dataset == "editlens" else ["test"]):
        texts, labels, sources = load(data / f"{split}_full.parquet")
        scores = predict(texts)
        result[split] = metrics(scores, labels, threshold)
        result[split]["by_source"] = {
            source: metrics(scores[np.asarray(sources) == source],
                            labels[np.asarray(sources) == source], threshold)
            for source in sorted(set(sources))
            if len(set(labels[np.asarray(sources) == source])) == 2
        }
    if args.audit_acl:
        acl_path = root / "data" / "acl_abstracts_v1" / "documents.jsonl.gz"
        with gzip.open(acl_path, "rt", encoding="utf-8") as file:
            acl = [json.loads(line) for line in file]
        acl = [row for row in acl if 100 <= len(row["abstract"].split()) <= 350]
        acl_scores = predict([row["abstract"] for row in acl])
        predicted = acl_scores >= threshold
        result["acl_human_audit"] = {
            "rows": len(acl),
            "false_positives": int(predicted.sum()),
            "fpr": float(predicted.mean()),
            "by_year": {
                str(year): {
                    "rows": sum(row["year"] == year for row in acl),
                    "fpr": float(predicted[np.asarray([row["year"] == year for row in acl])].mean()),
                }
                for year in sorted({row["year"] for row in acl})
            },
        }
    path = root / "results" / f"baseline_{args.dataset}_{args.model}_{args.train_tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
