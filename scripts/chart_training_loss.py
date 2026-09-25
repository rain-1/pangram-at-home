"""Plot the first Qwen3 stage-1 run from its saved Hugging Face Trainer state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("PANGRAM_DATA_ROOT", "/mnt/f/pangram-at-home")))
    parser.add_argument("--run-name", default="qwen3_17b_mixed_stage1_v1")
    args = parser.parse_args()
    state_path = args.data_root / "runs" / args.run_name / "checkpoint-828/trainer_state.json"
    raw = state_path.read_bytes()
    state = json.loads(raw)
    train = [r for r in state["log_history"] if "loss" in r and "eval_loss" not in r]
    val = [r for r in state["log_history"] if "eval_loss" in r]
    assert len(train) == 41 and len(val) == 3 and state["global_step"] == 828
    best = state["best_global_step"]
    assert best == 276

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.5, 6.7), dpi=180)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#ffffff")
    epochs = np.asarray([r["epoch"] for r in train])
    train_loss = np.asarray([r["loss"] for r in train])
    val_epochs = np.asarray([r["epoch"] for r in val])
    val_loss = np.asarray([r["eval_loss"] for r in val])
    ax.plot(epochs, train_loss, color="#28799a", linewidth=2.2, marker="o", markersize=3.5,
            label="Training loss · 20-step averages")
    ax.plot(val_epochs, val_loss, color="#b03a48", linewidth=2, linestyle="--", marker="D", markersize=7,
            label="Validation loss · full split")
    ax.axvline(1, color="#735fb1", linewidth=1.4, linestyle="--", alpha=.85)
    ax.scatter([1], [val_loss[0]], s=180, facecolors="none", edgecolors="#735fb1", linewidths=2, zorder=5)
    ax.annotate("Best checkpoint · epoch 1\nvalidation loss 0.00122 · AUROC 1.0000",
                xy=(1, val_loss[0]), xytext=(1.18, 1.4),
                arrowprops={"arrowstyle": "->", "color": "#735fb1", "linewidth": 1.2},
                color="#493781", fontsize=10, va="center")
    ax.set_yscale("log")
    ax.set_xlim(0, 3.12)
    ax.set_ylim(1e-5, 20)
    ax.set_xticks([0, .5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Cross-entropy loss · logarithmic scale")
    ax.grid(which="major", alpha=.17)
    ax.grid(which="minor", alpha=.07)
    ax.legend(loc="upper right", frameon=True)
    fig.suptitle("Qwen3 1.7B · first stage-1 training run", x=.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(.07, .905, "4,404 balanced mixed training rows · 1,244 validation rows · QLoRA · stopped after epoch 3", fontsize=10, color="#53657a")
    fig.text(.07, .045, "Validation AUROC: 1.0000 → 0.99963 → 0.99989. Higher validation loss after epoch 1 is consistent with more confident mistakes despite high ranking quality.", fontsize=9, color="#53657a")
    fig.text(.07, .023, "Source: Hugging Face Trainer checkpoint-828/trainer_state.json · 41 logged training averages and 3 full validation evaluations · no W&B run", fontsize=8.5, color="#53657a")
    fig.subplots_adjust(left=.09, right=.97, top=.85, bottom=.14)
    out = ROOT / "reports/charts"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"qwen3_stage1_loss.{ext}", facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)
    summary = {
        "run_name": args.run_name, "trainer_state_sha256": hashlib.sha256(raw).hexdigest(),
        "training_points": len(train), "validation_points": len(val), "best_global_step": best,
        "stopped_global_step": state["global_step"],
        "validation": [{"epoch": r["epoch"], "step": r["step"], "loss": r["eval_loss"], "roc_auc": r["eval_roc_auc"]} for r in val],
    }
    (ROOT / "reports/qwen3-stage1-loss.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
