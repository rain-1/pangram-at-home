"""Evaluate passage TF-IDF models as coarse sliding-window span baselines.

The character and word models are fitted on the frozen diverse passage train
split, then broadcast each 512-token window's log-odds across its tokens.
Overlapping window scores are averaged exactly as for the neural passage
baseline. Thresholds are independently calibrated to 5% document-any false
highlight on the existing pure-human calibration set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from run_baselines import fit_ngrams
from span_data import encode_document, window_starts
from span_metrics import calibrate_threshold, summarize_scores


SETS = {
    "human_calibration": "span_human_eval_v2/calibration.jsonl",
    "synthetic_v4_val": "span_training_v4/val.jsonl",
    "prior_synthetic_val": "span_pilot_v3/val.jsonl",
    "human_locked_test": "span_human_eval_v2/test.jsonl",
    "coauthor": "span_realistic_eval_v1/test.jsonl",
    "aitdna": "span_sources_v5/normalized_aitdna_real/locked_test.jsonl",
}


def score_rows(path: Path, tokenizer, predict, batch_size=256):
    rows = [json.loads(line) for line in path.open()]
    prepared = []
    window_texts = []
    assignments = []
    for doc_index, row in enumerate(rows):
        ids, offsets, labels = encode_document(row, tokenizer)
        prepared.append({"row": row, "label": np.asarray(labels, dtype=np.int16),
                         "score": np.zeros(len(ids), dtype=np.float64),
                         "counts": np.zeros(len(ids), dtype=np.int16)})
        for start in window_starts(len(ids), 512, 256):
            end = min(start + 512, len(ids))
            window_texts.append(row["text"][offsets[start][0]:offsets[end - 1][1]])
            assignments.append((doc_index, start, end))
    for base in range(0, len(window_texts), batch_size):
        probabilities = np.asarray(predict(window_texts[base:base + batch_size]), dtype=np.float64)
        margins = np.log(np.clip(probabilities, 1e-8, 1 - 1e-8) /
                         np.clip(1 - probabilities, 1e-8, 1 - 1e-8))
        for (index, start, end), score in zip(assignments[base:base + batch_size], margins):
            prepared[index]["score"][start:end] += float(score)
            prepared[index]["counts"][start:end] += 1
    for result in prepared:
        assert np.all(result["counts"] > 0)
        result["score"] /= result.pop("counts")
    return prepared


def cache_scores(path: Path, results) -> None:
    labels = [r["label"][r["label"] != -100] for r in results]
    scores = [r["score"][r["label"] != -100] for r in results]
    offsets = np.r_[0, np.cumsum([len(x) for x in labels])]
    np.savez_compressed(path, label=np.concatenate(labels).astype(np.int8),
                        score=np.concatenate(scores).astype(np.float32),
                        document_offsets=offsets,
                        document_ids=np.asarray([r["row"]["id"] for r in results]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--model", choices=("char", "word"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "runs" / f"span_{args.model}_window_baseline_v5"
    output.mkdir(parents=True, exist_ok=False)
    train_file = root / "data/diverse_pyramid_v1/train_full.parquet"
    table = pq.read_table(train_file, columns=["text", "label"]).to_pydict()
    predict = fit_ngrams(args.model, table["text"], np.asarray(table["label"], dtype=int))
    tokenizer = AutoTokenizer.from_pretrained(root / "models/Qwen3-1.7B")
    calibration = score_rows(root / "data" / SETS["human_calibration"], tokenizer, predict)
    threshold = calibrate_threshold(calibration, .05, "document")
    report = {"model": f"{args.model} TF-IDF passage classifier broadcast over 512/256 windows",
              "train_file": str(train_file), "train_rows": len(table["text"]),
              "threshold": threshold,
              "calibration": "5% document-any false highlight on pure-human calibration; frozen for all evaluation sets",
              "sets": {}}
    for name, relative in SETS.items():
        results = calibration if name == "human_calibration" else score_rows(root / "data" / relative, tokenizer, predict)
        report["sets"][name] = {"overall": summarize_scores(results, threshold),
                               "by_kind": {kind: summarize_scores([r for r in results if r["row"]["kind"] == kind], threshold)
                                           for kind in ("human", "ai", "mixed")},
                               "by_domain": {domain: summarize_scores([r for r in results if r["row"]["domain"] == domain], threshold)
                                             for domain in sorted({r["row"]["domain"] for r in results})}}
        cache_scores(output / f"{name}_scores.npz", results)
        print(name, report["sets"][name]["overall"]["ai_recall"],
              report["sets"][name]["overall"]["fpr"], flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
