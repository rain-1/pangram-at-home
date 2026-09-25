"""Measure near-duplicate train/evaluation text without exposing text itself."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    folder = Path("/mnt/f/pangram-at-home/data/diverse_pyramid_v1")
    train = pq.read_table(folder / "train_full.parquet", columns=["text", "text_id", "source"]).to_pydict()
    evaluation = {split: pq.read_table(folder / f"{split}_full.parquet", columns=["text", "text_id", "source"]).to_pydict()
                  for split in ("val", "test")}
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), min_df=2, max_features=250_000,
                                 dtype=np.float32, sublinear_tf=True)
    x = vectorizer.fit_transform(train["text"])
    report = {"method": "word 2-3 gram TF-IDF cosine; fit on train", "train_rows": len(train["text"]),
              "splits": {}}
    for split, rows in evaluation.items():
        y = vectorizer.transform(rows["text"])
        maximum = []
        matches = []
        for start in range(0, y.shape[0], 100):
            scores = (y[start:start + 100] @ x.T).tocsr()
            for offset in range(scores.shape[0]):
                row = scores.getrow(offset)
                if row.nnz:
                    index = row.data.argmax()
                    score = float(row.data[index])
                    train_index = int(row.indices[index])
                else:
                    score, train_index = 0.0, -1
                maximum.append(score)
                if score >= .7:
                    matches.append({"eval_id": rows["text_id"][start + offset],
                                    "eval_source": rows["source"][start + offset],
                                    "train_id": train["text_id"][train_index],
                                    "train_source": train["source"][train_index],
                                    "cosine": score})
        report["splits"][split] = {"rows": len(maximum), "max_cosine": max(maximum),
                                   "p99_cosine": float(np.quantile(maximum, .99)),
                                   "gte_0_7": sum(score >= .7 for score in maximum),
                                   "gte_0_8": sum(score >= .8 for score in maximum),
                                   "gte_0_9": sum(score >= .9 for score in maximum),
                                   "high_similarity_pairs": sorted(matches, key=lambda item: item["cosine"], reverse=True)}
    out = Path(__file__).resolve().parents[1] / "reports/metrics/diverse_similarity_v1.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
