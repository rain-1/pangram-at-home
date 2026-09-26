"""Train simple binary detectors on the frozen data pyramids."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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
    # Advance in the score array's dtype. A float64-only increment rounds back
    # to the same float32 score during comparison and admits one extra human.
    return float(np.nextafter(human[allowed], np.array(np.inf, dtype=human.dtype)))


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
    parser.add_argument("--dataset", choices=["editlens", "pmc", "paper", "mixed", "diverse"], default="editlens")
    parser.add_argument("--train-tier", default="medium")
    parser.add_argument("--model", choices=["char", "word", "embedding"], default="char")
    parser.add_argument("--audit-acl", action="store_true", help="Report FPR on pre-2023 ACL human abstracts")
    parser.add_argument("--audit-pmc-body", action="store_true", help="Report FPR on held-out PMC full-text chunks")
    parser.add_argument("--cross-test", action="store_true", help="Score paper test with held-out generator swaps")
    args = parser.parse_args()
    root = args.root
    folder = {
        "editlens": "editlens_pyramid_v1",
        "pmc": "pmc_pyramid_v1",
        "paper": "paper_pyramid_v1",
        "mixed": "mixed_pyramid_v1",
        "diverse": "diverse_pyramid_v1",
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
        if args.dataset == "diverse":
            domains = np.asarray(pq.read_table(data / f"{split}_full.parquet", columns=["domain"]).to_pydict()["domain"])
            result[split]["by_domain"] = {
                domain: metrics(scores[domains == domain], labels[domains == domain], threshold)
                for domain in sorted(set(domains))
            }
    if args.dataset == "diverse":
        external = root / "data/mage_external_v1/frozen"
        for split in ("gpt4_ood", "paraphrase"):
            rows = pq.read_table(external / f"{split}.parquet", columns=["text", "label", "source"]).to_pydict()
            labels = np.asarray(rows["label"], dtype=int)
            scores = predict(rows["text"])
            domains = np.asarray(rows["source"])
            result[split] = metrics(scores, labels, threshold)
            result[split]["by_domain"] = {
                domain: metrics(scores[domains == domain], labels[domains == domain], threshold)
                for domain in sorted(set(domains))
            }
        books = pq.read_table(root / "data/standard_ebooks_v1/human.parquet", columns=["text", "source_id"]).to_pydict()
        book_scores = predict(books["text"])
        result["standard_ebooks_human"] = {
            "rows": len(book_scores), "false_positives": int((book_scores >= threshold).sum()),
            "fpr": float((book_scores >= threshold).mean()),
            "by_book": {book: float((book_scores[np.asarray(books["source_id"]) == book] >= threshold).mean())
                        for book in sorted(set(books["source_id"]))},
        }
        essay_path = root / "data/persuade_essays_v1/human_eval.parquet"
        if essay_path.exists():
            essays = pq.read_table(essay_path, columns=["text_id", "text"]).to_pydict()
            indices = sorted(range(len(essays["text"])),
                             key=lambda i: hashlib.sha256(("persuade-audit-v1:" + essays["text_id"][i]).encode()).digest())[:1000]
            essay_scores = predict([essays["text"][i] for i in indices])
            result["persuade_essays_human"] = {
                "rows": len(essay_scores), "false_positives": int((essay_scores >= threshold).sum()),
                "fpr": float((essay_scores >= threshold).mean()),
            }
        finance_path = root / "data/federal_reserve_beige_book_v1/human.parquet"
        if finance_path.exists():
            finance = pq.read_table(finance_path, columns=["text", "source_id"]).to_pydict()
            finance_scores = predict(finance["text"])
            result["federal_reserve_human"] = {
                "rows": len(finance_scores), "false_positives": int((finance_scores >= threshold).sum()),
                "fpr": float((finance_scores >= threshold).mean()),
                "by_release": {release: float((finance_scores[np.asarray(finance["source_id"]) == release] >= threshold).mean())
                               for release in sorted(set(finance["source_id"]))},
            }
        social_path = root / "data/stackexchange_writers_v1/human_eval.parquet"
        if social_path.exists():
            social = pq.read_table(social_path, columns=["text_id", "text"]).to_pydict()
            indices = sorted(range(len(social["text"])),
                             key=lambda i: hashlib.sha256(("stackexchange-audit-v1:" + social["text_id"][i]).encode()).digest())[:1000]
            social_scores = predict([social["text"][i] for i in indices])
            result["stackexchange_writers_human"] = {
                "rows": len(social_scores), "false_positives": int((social_scores >= threshold).sum()),
                "fpr": float((social_scores >= threshold).mean()),
            }
        paper_body_path = root / "data/pmc_body_audit_v1/human_test.parquet"
        if paper_body_path.exists():
            bodies = pq.read_table(paper_body_path, columns=["text", "source_id"]).to_pydict()
            body_scores = predict(bodies["text"])
            result["pmc_full_body_human"] = {
                "rows": len(body_scores), "papers": len(set(bodies["source_id"])),
                "false_positives": int((body_scores >= threshold).sum()),
                "fpr": float((body_scores >= threshold).mean()),
            }
        enron_path = root / "data/editlens_pyramid_v1/test_enron_full.parquet"
        if enron_path.exists():
            enron = pq.read_table(enron_path, columns=["text", "label"]).to_pydict()
            enron_labels = np.asarray(enron["label"], dtype=int)
            enron_scores = predict(enron["text"])
            result["enron_external"] = metrics(enron_scores, enron_labels, threshold)
        raid_path = root / "data/raid_external_v1/frozen.parquet"
        if raid_path.exists():
            raid = pq.read_table(raid_path, columns=["text", "label", "source", "generator"]).to_pydict()
            raid_labels = np.asarray(raid["label"], dtype=int)
            raid_scores = predict(raid["text"])
            raid_domains = np.asarray(raid["source"])
            result["raid_external"] = metrics(raid_scores, raid_labels, threshold)
            result["raid_external"]["by_domain"] = {
                domain: metrics(raid_scores[raid_domains == domain], raid_labels[raid_domains == domain], threshold)
                for domain in sorted(set(raid_domains))
            }
    if args.audit_acl:
        acl_path = root / "data" / "acl_abstracts_v1" / "documents.jsonl.gz"
        with gzip.open(acl_path, "rt", encoding="utf-8") as file:
            acl = [json.loads(line) for line in file]
        acl = [row for row in acl if 100 <= len(row["abstract"].split()) <= 350]
        if args.dataset in {"paper", "mixed"}:
            used_ids = set()
            paper_dir = root / "data" / "paper_pyramid_v1"
            for split in ["train", "val", "test"]:
                used_ids.update(
                    row["source_id"] for row in pq.read_table(
                        paper_dir / f"{split}_full.parquet",
                        columns=["source", "source_id"],
                    ).to_pylist()
                    if row["source"] == "acl_anthology"
                )
            acl = [row for row in acl if row["source_id"] not in used_ids]
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
    if args.audit_pmc_body:
        if args.dataset not in {"editlens", "pmc"}:
            raise SystemExit("PMC body audit is only work-disjoint from editlens/pmc training")
        body = pq.read_table(root / "data" / "pmc_body_audit_v1" / "human_test.parquet", columns=["text"]).to_pydict()["text"]
        body_scores = predict(body)
        result["pmc_body_human_audit"] = {
            "rows": len(body),
            "false_positives": int((body_scores >= threshold).sum()),
            "fpr": float((body_scores >= threshold).mean()),
        }
    if args.cross_test:
        if args.dataset != "paper":
            raise SystemExit("--cross-test requires --dataset paper")
        texts, labels, sources = load(root / "data" / "paper_cross_model_test_v1" / "test.parquet")
        scores = predict(texts)
        result["cross_model_test"] = metrics(scores, labels, threshold)
        result["cross_model_test"]["by_source"] = {
            source: metrics(scores[np.asarray(sources) == source],
                            labels[np.asarray(sources) == source], threshold)
            for source in sorted(set(sources))
        }
    path = root / "results" / f"baseline_{args.dataset}_{args.model}_{args.train_tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
