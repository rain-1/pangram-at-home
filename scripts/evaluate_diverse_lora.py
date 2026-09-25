"""Score the trained Qwen adapter on frozen domain and source-family tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from run_baselines import metrics, threshold_for_fpr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    if len(set(labels)) == 2:
        return metrics(scores, labels, threshold)
    pred = scores >= threshold
    return {"rows": len(labels), "human": int((labels == 0).sum()), "ai": int((labels == 1).sum()),
            "false_positives": int((pred & (labels == 0)).sum()),
            "fpr": float(pred.mean()) if (labels == 0).all() else None,
            "tpr": float(pred.mean()) if (labels == 1).all() else None,
            "roc_auc": None}


def read_rows(root: Path, name: str) -> tuple[Path, dict]:
    paths = {
        "val": root / "data/diverse_pyramid_v1/val_full.parquet",
        "test": root / "data/diverse_pyramid_v1/test_full.parquet",
        "raid_external": root / "data/raid_external_v1/frozen.parquet",
        "enron_external": root / "data/editlens_pyramid_v1/test_enron_full.parquet",
        "gpt4_ood": root / "data/mage_external_v1/frozen/gpt4_ood.parquet",
        "paraphrase": root / "data/mage_external_v1/frozen/paraphrase.parquet",
        "standard_ebooks_human": root / "data/standard_ebooks_v1/human.parquet",
        "persuade_essays_human": root / "data/persuade_essays_v1/human_eval.parquet",
        "federal_reserve_human": root / "data/federal_reserve_beige_book_v1/human.parquet",
        "stackexchange_writers_human": root / "data/stackexchange_writers_v1/human_eval.parquet",
        "pmc_full_body_human": root / "data/pmc_body_audit_v1/human_test.parquet",
    }
    path = paths[name]
    rows = pq.read_table(path).to_pydict()
    count = len(rows["text"])
    if name == "persuade_essays_human":
        # Fixed audit sample; keep the full restricted corpus local and untouched.
        ids = sorted(range(count), key=lambda i: hashlib.sha256(("persuade-audit-v1:" + str(rows.get("text_id", rows.get("source_id"))[i])).encode()).digest())[:1000]
        rows = {key: [values[i] for i in ids] for key, values in rows.items()}
        count = len(ids)
    if name == "stackexchange_writers_human":
        ids = sorted(range(count), key=lambda i: hashlib.sha256(("stackexchange-audit-v1:" + rows["text_id"][i]).encode()).digest())[:1000]
        rows = {key: [values[i] for i in ids] for key, values in rows.items()}
        count = len(ids)
    rows.setdefault("label", [0] * count)
    rows.setdefault("source", [name] * count)
    rows.setdefault("domain", [name] * count)
    rows.setdefault("generator", ["human" if x == 0 else "AI" for x in rows["label"]])
    if name in {"raid_external", "gpt4_ood", "paraphrase"}:
        rows["domain"] = rows["source"]
    if name == "enron_external":
        rows["domain"] = ["professional_email"] * count
    if name == "standard_ebooks_human":
        rows["domain"] = ["classic_fiction"] * count
        rows["source"] = rows["source_id"]
    if name == "persuade_essays_human":
        rows["domain"] = ["student_essays"] * count
    if name == "federal_reserve_human":
        rows["domain"] = ["professional_finance"] * count
    if name == "stackexchange_writers_human":
        rows["domain"] = ["social_qa_other_platform"] * count
    if name == "pmc_full_body_human":
        rows["domain"] = ["paper_full_body"] * count
    return path, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/f/pangram-at-home"))
    parser.add_argument("--run-name", default="qwen3_17b_diverse_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-run-id", default="")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = args.root / "runs" / args.run_name
    adapter = run / "best_adapter"
    config = json.loads((run / "run_config.json").read_text())
    if "train_sha256" in config:
        assert sha256(Path(config["train_file"])) == config["train_sha256"]
        assert sha256(Path(config["val_file"])) == config["val_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True),
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()
    cache_dir = run / "final_eval"
    cache_dir.mkdir(exist_ok=True)
    names = ["val", "test", "raid_external", "enron_external", "gpt4_ood", "paraphrase",
             "standard_ebooks_human", "persuade_essays_human", "federal_reserve_human",
             "stackexchange_writers_human", "pmc_full_body_human"]
    scored = {}
    for name in names:
        path, rows = read_rows(args.root, name)
        labels = np.asarray(rows["label"], dtype=np.int8)
        source = np.asarray(rows["source"], dtype=str)
        domain = np.asarray(rows["domain"], dtype=str)
        generator = np.asarray(rows["generator"], dtype=str)
        input_hash = sha256(path)
        cache = cache_dir / f"{name}.npz"
        if cache.exists():
            old = np.load(cache)
            assert str(old["input_sha256"]) == input_hash and len(old["margin"]) == len(labels)
            scores = old["margin"]
        else:
            margins = []
            for start in range(0, len(labels), args.batch_size):
                batch = tokenizer(rows["text"][start:start + args.batch_size], return_tensors="pt",
                                  padding=True, truncation=True, max_length=config["max_length"]).to("cuda")
                with torch.inference_mode():
                    logits = model(**batch).logits.float()
                margins.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
                if start and start % 400 < args.batch_size:
                    print(name, min(start + args.batch_size, len(labels)), "/", len(labels), flush=True)
            scores = np.asarray(margins, dtype=np.float32)
            np.savez_compressed(cache, margin=scores, label=labels, source=source,
                                domain=domain, generator=generator, input_sha256=input_hash)
        scored[name] = (scores, labels, source, domain, generator, input_hash)
    val_scores, val_labels, *_ = scored["val"]
    threshold = threshold_for_fpr(val_scores, val_labels, .02)
    report = {"run_name": args.run_name, "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
              "threshold": threshold, "threshold_source": "diverse validation, <=2% human FPR",
              "wandb_url": f"https://wandb.ai/eac-adsf/pangram-at-home/runs/{args.wandb_run_id}" if args.wandb_run_id else None,
              "splits": {}, "operating_points": {}}
    for target in (.005, .01, .02):
        point_threshold = threshold_for_fpr(val_scores, val_labels, target)
        report["operating_points"][f"val_fpr_le_{target:.3f}"] = {
            "threshold": point_threshold,
            "test": group_metrics(scored["test"][0], scored["test"][1], point_threshold),
            "raid_external": group_metrics(scored["raid_external"][0], scored["raid_external"][1], point_threshold),
            "stackexchange_writers_human": group_metrics(scored["stackexchange_writers_human"][0],
                                                            scored["stackexchange_writers_human"][1], point_threshold),
            "pmc_full_body_human": group_metrics(scored["pmc_full_body_human"][0],
                                                   scored["pmc_full_body_human"][1], point_threshold),
        }
    for name, (scores, labels, source, domain, generator, input_hash) in scored.items():
        result = group_metrics(scores, labels, threshold)
        result["input_sha256"] = input_hash
        result["by_domain"] = {d: group_metrics(scores[domain == d], labels[domain == d], threshold)
                               for d in sorted(set(domain))}
        if name in {"test", "raid_external", "enron_external", "standard_ebooks_human",
                    "persuade_essays_human", "federal_reserve_human", "stackexchange_writers_human",
                    "pmc_full_body_human"}:
            result["by_source"] = {s: group_metrics(scores[source == s], labels[source == s], threshold)
                                   for s in sorted(set(source))}
        if name == "raid_external":
            result["by_generator"] = {g: group_metrics(scores[generator == g], labels[generator == g], threshold)
                                      for g in sorted(set(generator))}
        report["splits"][name] = result
        print(name, "AUROC", result.get("roc_auc"), "FPR", result.get("fpr"), "AI recall", result.get("tpr"), flush=True)
    out = Path(__file__).resolve().parents[1] / "reports/metrics" / f"{args.run_name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if args.wandb_run_id:
        try:
            import wandb
            wb = wandb.init(project="pangram-at-home", entity="eac-adsf", id=args.wandb_run_id,
                            resume="must", job_type="final-evaluation")
            for name, result in report["splits"].items():
                for field in ("roc_auc", "fpr", "tpr"):
                    if result.get(field) is not None:
                        wb.summary[f"final/{name}/{field}"] = result[field]
            for point_name, point in report["operating_points"].items():
                for split in ("test", "raid_external", "stackexchange_writers_human", "pmc_full_body_human"):
                    for field in ("fpr", "tpr"):
                        if point[split].get(field) is not None:
                            wb.summary[f"operating_points/{point_name}/{split}/{field}"] = point[split][field]
            wb.finish()
        except Exception as exc:
            print(f"W&B final summary upload failed; local report is saved at {out}: {exc}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
