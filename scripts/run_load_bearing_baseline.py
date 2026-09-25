"""Score frozen splits with Louis Abraham's GitHub PR vocabulary cluster model.

The model identifies the arriving PR-writing cluster, not AI authorship. Its
cluster-0 log odds are used as a fixed, label-free transfer baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from run_baselines import metrics, threshold_for_fpr


COMMIT = "45d353684c82ec59342cbcd96a85829d22acfd31"
MODEL_SHA256 = "22a9f53fdefa7c812e481c6d96f1011624765413567b0c16cf6598246bc76f8d"
URL_RE = re.compile(r"https?://[^\s<>\"'`\)\]\}]+")
TAG_RE = re.compile(r"<[a-z/!][^<>]*>")
WORD_RE = re.compile(r"[a-z0-9_/-]*[a-z][a-z0-9_/-]*")
SNYK_ID_RE = re.compile(r"^snyk-.+-\d{4,}$")


def tokens(body: str) -> list[str]:
    body = body.lower()
    out = []
    for match in URL_RE.finditer(body):
        host = match.group().split("//", 1)[1].split("/", 1)[0].split("@")[-1].split(":")[0]
        labels = [x for x in host.split(".") if x and x != "www"]
        out.append(f"[{labels[-2] if len(labels) >= 2 else labels[0] if labels else 'link'}-url]")
    out.extend("—" for _ in range(body.count("—")))
    rest = TAG_RE.sub(" ", URL_RE.sub(" ", body))
    for match in WORD_RE.finditer(rest):
        word = match.group().strip("_/").rstrip("-")
        if word:
            out.append("[snyk-id]" if word.startswith("snyk-") and SNYK_ID_RE.fullmatch(word) else word)
    return out


class LoadBearing:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == MODEL_SHA256, "Unexpected model.js snapshot"
        model = json.loads(raw.decode().partition("=")[2].rstrip(" ;\n"))
        self.k = model["k"]
        vocab = []
        previous = ""
        encoded = model["vocab"]
        i = 0
        while i < len(encoded):
            j = i + 1
            while j < len(encoded) and not ("A" <= encoded[j] <= "Z"):
                j += 1
            previous = previous[:ord(encoded[i]) - 65] + encoded[i + 1:j]
            vocab.append(previous)
            i = j
        assert len(vocab) == model["words"]
        self.index = {word: i for i, word in enumerate(vocab)}
        alphabet = {char: i for i, char in enumerate(model["alphabet"])}
        na, esc = len(alphabet), model["escape"]
        lo, split, hi = model["grid"]
        low_step = (split - lo) / (esc - 1)
        high_step = (hi - split) / ((na - 1 - esc) * na - 1)
        values = np.empty(model["words"] * self.k, dtype=np.float64)
        encoded = model["weights"]
        i = 0
        for j in range(len(values)):
            a = alphabet[encoded[i]]
            i += 1
            if a == 0:
                values[j] = 0
            elif a <= esc:
                values[j] = lo + (a - 1) * low_step
            else:
                values[j] = split + ((a - 1 - esc) * na + alphabet[encoded[i]]) * high_step
                i += 1
        assert i == len(encoded)
        self.weights = values.reshape(-1, self.k)
        self.floor = np.asarray(model["floor"], dtype=np.float64)

    def score_one(self, text: str) -> tuple[float, int, int]:
        words = tokens(text)
        count = Counter(self.index[w] for w in words if w in self.index)
        known = sum(count.values())
        if not known:
            return -math.log(self.k - 1), 0, len(words)
        scores = known * self.floor.copy()
        for index, n in count.items():
            scores += n * self.weights[index]
        others = scores[1:]
        top = np.max(others)
        logsumexp_others = top + math.log(np.exp(others - top).sum())
        return float(scores[0] - logsumexp_others), known, len(words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--dataset", choices=["mixed", "paper", "editlens", "pmc", "diverse"], required=True)
    args = parser.parse_args()
    model_path = args.root / "models/load-bearing/model.js"
    model = LoadBearing(model_path)
    folder = args.root / "data" / f"{args.dataset}_pyramid_v1"
    cache_dir = args.root / "runs/load_bearing_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scored = {}
    for split in ("val", "test"):
        path = folder / f"{split}_full.parquet"
        rows = pq.read_table(path, columns=["text", "label", "source", "text_id"]).to_pylist()
        result = [model.score_one(row["text"]) for row in rows]
        score, known, total = (np.asarray(x) for x in zip(*result))
        scored[split] = (rows, score)
        np.savez_compressed(cache_dir / f"{args.dataset}_{split}.npz", score=score,
                            label=np.asarray([r["label"] for r in rows], dtype=np.int8),
                            source=np.asarray([r["source"] for r in rows], dtype=str),
                            known=known, total=total)
        print(args.dataset, split, len(rows), "median vocabulary coverage", float(np.median(known / np.maximum(total, 1))), flush=True)
    val_rows, val_score = scored["val"]
    val_labels = np.asarray([r["label"] for r in val_rows])
    threshold = threshold_for_fpr(val_score, val_labels, .02)
    report = {"model": "load-bearing arriving PR vocabulary cluster", "upstream_commit": COMMIT,
              "model_sha256": MODEL_SHA256, "score": "cluster-0 log odds versus other nine clusters",
              "threshold_source": f"{args.dataset} validation, <=2% human FPR", "threshold": threshold}
    for split in ("val", "test"):
        rows, score = scored[split]
        labels = np.asarray([r["label"] for r in rows])
        sources = np.asarray([r["source"] for r in rows])
        report[split] = metrics(score, labels, threshold)
        report[split]["by_source"] = {
            source: metrics(score[sources == source], labels[sources == source], threshold)
            for source in sorted(set(sources)) if len(set(labels[sources == source])) == 2
        }
        if args.dataset == "diverse":
            domains = np.asarray(pq.read_table(folder / f"{split}_full.parquet", columns=["domain"]).to_pydict()["domain"])
            report[split]["by_domain"] = {
                domain: metrics(score[domains == domain], labels[domains == domain], threshold)
                for domain in sorted(set(domains))
            }
    extras = []
    if args.dataset == "editlens":
        extras.append(("test_enron", folder / "test_enron_full.parquet"))
    if args.dataset == "paper":
        extras.append(("cross_model_test", args.root / "data/paper_cross_model_test_v1/test.parquet"))
    if args.dataset == "diverse":
        extras.extend([
            ("enron_external", args.root / "data/editlens_pyramid_v1/test_enron_full.parquet"),
            ("raid_external", args.root / "data/raid_external_v1/frozen.parquet"),
            ("gpt4_ood", args.root / "data/mage_external_v1/frozen/gpt4_ood.parquet"),
            ("paraphrase", args.root / "data/mage_external_v1/frozen/paraphrase.parquet"),
        ])
    for name, path in extras:
        rows = pq.read_table(path, columns=["text", "label", "source"]).to_pylist()
        score = np.asarray([model.score_one(row["text"])[0] for row in rows])
        labels = np.asarray([row["label"] for row in rows])
        sources = np.asarray([row["source"] for row in rows])
        np.savez_compressed(cache_dir / f"{args.dataset}_{name}.npz", score=score,
                            label=labels.astype(np.int8), source=sources.astype(str))
        report[name] = metrics(score, labels, threshold)
        report[name]["by_source"] = {
            source: metrics(score[sources == source], labels[sources == source], threshold)
            for source in sorted(set(sources)) if len(set(labels[sources == source])) == 2
        }
    output = Path(__file__).resolve().parents[1] / "reports/metrics" / f"baseline_{args.dataset}_load_bearing_full.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, "AUROC", report["test"]["roc_auc"], "FPR", report["test"]["fpr"], "recall", report["test"]["tpr"])


if __name__ == "__main__":
    main()
