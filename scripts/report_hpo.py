"""Publish readable HPO results without raw training text or credentials."""
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/f/pangram-at-home/vast_results_v1/live")


def main():
    trials = json.loads((ROOT / "hpo_diverse_v3_summary.json").read_text())
    rows = []
    for t in trials:
        m, c = t["metrics"], t["config"]
        rows.append(dict(trial=m["run_name"].split("_")[-2], run_name=m["run_name"],
                         status="failed" if t["error"] else "completed" if m["examples_seen"] == 25600 else "pruned",
                         examples=m["examples_seen"], learning_rate=c["learning_rate"],
                         batch=2*c["gradient_accumulation_steps"], rank=c["lora_rank"], dropout=c["lora_dropout"],
                         score=m["score"], auc=m["val_roc_auc"], pauc=m["val_partial_auc_fpr_5pct"],
                         recall=m["val_ai_recall_at_fpr_2pct"], worst_recall=m["val_worst_domain_ai_recall_at_fpr_2pct"]))
    rows.sort(key=lambda r: (r["status"] != "completed", -r["score"]))
    full = [r for r in rows if r["status"] == "completed"]
    best = full[0]
    ref = next(r for r in rows if r["trial"] == "00000")
    out = REPO / "reports"
    with (out / "hpo_results_v3.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "metrics/hpo_results_v3.json").write_text(json.dumps(rows, indent=2)+"\n")
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    with PdfPages(out / "hpo_results_v3.pdf") as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(13, 7))
        fig.subplots_adjust(top=.75, bottom=.23, left=.07, right=.97, wspace=.4)
        fig.suptitle("Hyperparameter pilot: small gains, useful candidates", fontsize=19, y=.95, color="#17354A")
        fig.text(.07,.85,"24 trials · 7 completed full budget · 17 pruned · 0 Ray failures\nSame 800-row validation set used throughout; results are not held-out test estimates.",fontsize=12)
        for ax, key, title in zip(axes, ("auc","pauc","recall"), ("Full AUROC", "Partial AUROC at ≤5% FPR", "AI recall at validation ≤2% FPR")):
            vals = [100*ref[key],100*best[key]]
            ax.bar([0,1],vals,color=["#8A9BA8","#167C80"],width=.6)
            ax.set_xticks([0,1],["Reference", "Selected"]); ax.set_ylim(0,105); ax.set_title(title,fontsize=11)
            for i,v in enumerate(vals): ax.text(i,v+1,f"{v:.2f}%",ha="center")
        fig.text(.07,.08,"Selected: LR 7.61e−5 · effective batch 8 · rank 32 / alpha 64 · dropout 0.0688\nRecall gain: 3 of 400 AI examples. Worst-category recall: 85.0%, versus reference 87.5%.\nSelection score = 70% standardized partial AUROC + 30% full AUROC.", fontsize=11)
        pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(13,7))
        for r in rows:
            path=ROOT/r["run_name"]/"sweep_metrics.jsonl"
            if not path.exists(): continue
            history=[json.loads(line) for line in path.read_text().splitlines()]
            named=r["trial"] in {best["trial"],ref["trial"],"00003"}
            label={best["trial"]:"Selected (#19)",ref["trial"]:"Reference (#00)","00003":"Higher recall candidate (#03)"}.get(r["trial"])
            ax.plot([x["step"]*r["batch"] for x in history],
                    [.7*x["eval_partial_auc_fpr_5pct"]+.3*x["eval_roc_auc"] for x in history],
                    color=None if named else "#ABB5BF",alpha=1 if named else .4,lw=2.5 if named else 1,label=label)
        ax.set(title="Validation score as training progresses",xlabel="Source examples seen (equal budget across batches)",ylabel="Selection score")
        ax.legend(); ax.grid(alpha=.2); fig.text(.08,.015,"Lines ending early were pruned by ASHA; do not interpret them as full-budget results.",fontsize=10)
        fig.tight_layout(rect=(0,.04,1,1)); pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(13,10));ax.axis("off")
        data=[[r["trial"],r["status"],str(r["examples"]),f'{r["learning_rate"]:.2e}',str(r["batch"]),str(r["rank"]),f'{r["dropout"]:.3f}',f'{r["pauc"]:.4f}',f'{100*r["recall"]:.2f}'] for r in rows]
        table=ax.table(cellText=data,colLabels=["Trial","Status","Examples","LR","Batch","Rank","Dropout","pAUC","Recall %"],loc="center",cellLoc="center")
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
        ax.set_title("All trials: full-budget candidates first",fontsize=18,pad=20)
        fig.text(.07,.04,"Pruned results use less training and are not directly comparable to completed trials.\nRecall uses a threshold chosen on these same validation humans; it is not a guaranteed deployment FPR.",fontsize=10)
        pdf.savefig(fig);plt.close(fig)
    table="\n".join(f'| {r["trial"]} | {r["learning_rate"]:.3g} | {r["batch"]} | {r["rank"]} | {r["dropout"]:.3f} | {r["auc"]:.5f} | {r["pauc"]:.5f} | {100*r["recall"]:.2f}% | {100*r["worst_recall"]:.2f}% |' for r in full)
    text=f'''# Hyperparameter tuning results — diverse v3

24 trials finished: 7 completed 25,600 source examples and 17 were stopped by ASHA. Ray reported no trial failures. W&B marked some intentionally terminated trials as crashed because the old shutdown path did not finalize their runs; the shutdown code has since been corrected for future runs.

The selected configuration is learning rate **{best['learning_rate']:.8g}**, effective batch **8**, LoRA rank **32**, alpha **64**, dropout **{best['dropout']:.6f}**. Other settings: Qwen3-1.7B, BF16 LoRA without quantization, 512 source tokens, attention and feed-forward LoRA, weight decay 0.01, cosine schedule, 5% warmup, seed 42. There was no Repeat2. Batch 8 was implemented as microbatch 2 × accumulation 4.

Selection used the final reported score among full-budget trials: 70% standardized partial AUROC at ≤5% FPR plus 30% full AUROC. Each training run saved its checkpoint with best partial AUROC. The selected trial's saved checkpoint is also its final checkpoint, so its reported selected metrics match that checkpoint.

## Full-budget results

| Trial | LR | Batch | Rank | Dropout | AUROC | Partial AUROC | AI recall at val ≤2% FPR | Worst-domain recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table}

## Interpretation

The reference trial (#00) used LR 5e-5, effective batch 16, rank 16, and dropout 0.1. The selected trial gained {100*(best['auc']-ref['auc']):.3f} percentage points of full AUROC and {100*(best['pauc']-ref['pauc']):.3f} points of standardized partial AUROC. Recall rose from 92.75% to 93.50%: three additional AI passages out of 400. Its worst-domain recall fell from 87.5% to 85.0%. Trial #03 instead achieved 95.0% overall recall and 91.25% worst-domain recall, so retain it as a confirmation candidate.

These are selection results on the same 800 validation passages (400 human / 400 AI), not independent evidence of improvement. Each parameter combination was run once. Jointly varied parameters and ASHA pruning do not establish that an individual rank, batch, or learning rate causes better performance. A threshold permitting up to eight human errors on this validation set does not guarantee 2% FPR on new writing.

The data mixture experiment automatically uses the selected configuration for all nine 4,000-row mixes, with 12,800 source examples seen per run. Results are still pending when this report was generated.

## Next controlled experiment

Run a matched single-copy / Repeat2 pair from the same base checkpoint, seed, data order, source-token cap, hyperparameters, and optimizer-step budget. Use BF16 for both to match tuning. If memory requires microbatch 1 and accumulation 8, apply it to both; this is a controlled local pair rather than an exact numerical reproduction of the remote microbatch configuration. Keep all 512 source tokens before duplicating to up to 1,024 model tokens. Repetition also applies during evaluation. Record wall time and peak GPU memory: equal examples imply more compute for Repeat2.

Repeat2 may help passage classification, but the passage head already sees the whole window. Its stronger motivation is token-level prediction with a causal backbone. A token pilot is a new objective and needs known span provenance; this sweep supplies starting settings, not proven token-task optima.

Artifacts: [charts](hpo_results_v3.pdf), [all-trial CSV](hpo_results_v3.csv), [machine-readable results](metrics/hpo_results_v3.json).
'''
    (out/"hpo_results_v3.md").write_text(text)
    print(out/"hpo_results_v3.pdf")


if __name__=="__main__": main()
