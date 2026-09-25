"""Readable report of the nine completed data-mixture pilots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO=Path(__file__).resolve().parents[1]
SUMMARY=Path("/mnt/f/pangram-at-home/vast_results_v1/live/ablation_diverse_v3_summary.json")
LABELS={"control_35paper":"Control: 35% paper","without_paper":"Without papers",
        "without_reference_education":"Without reference / education","without_creative":"Without creative writing",
        "without_social_qa":"Without social / Q&A","without_reviews":"Without reviews",
        "without_news":"Without news","paper_20pct":"20% paper","paper_50pct":"50% paper"}


def main():
    trials=json.loads(SUMMARY.read_text())
    if len(trials)!=9 or any(t["error"] or t["metrics"]["examples_seen"]!=12800 for t in trials):
        raise RuntimeError("Ablation sweep is incomplete")
    data={t["config"]["dataset"]:t["metrics"] for t in trials}
    order=["control_35paper","paper_20pct","paper_50pct","without_paper","without_creative",
           "without_reference_education","without_social_qa","without_reviews","without_news"]
    if set(order)!=set(data):raise RuntimeError("Unexpected ablation datasets")
    control=data["control_35paper"]["val_partial_auc_fpr_5pct"]
    pdf=REPO/"reports/ablation_results_v3.pdf"
    with PdfPages(pdf) as pages:
        fig,axes=plt.subplots(1,2,figsize=(14,8))
        fig.subplots_adjust(left=.23,right=.96,top=.78,bottom=.19,wspace=.28)
        fig.text(.06,.95,"Which data categories help?",fontsize=20,weight="bold",color="#17354A")
        fig.text(.06,.89,"Nine matched 4,000-row training mixes · 12,800 examples seen each · one seed · same 800-row validation set",fontsize=11)
        fig.text(.06,.85,"All mixes used the selected Qwen training settings. Training categories excluded from a mix were replaced by other categories.",fontsize=10)
        y=list(range(len(order)))
        names=[LABELS[x] for x in order]
        vals=[100*data[x]["val_partial_auc_fpr_5pct"] for x in order]
        recall=[100*data[x]["val_ai_recall_at_fpr_2pct"] for x in order]
        colors=["#17354A" if x=="control_35paper" else "#2C898B" if x.startswith("paper_") else "#8CA5B4" for x in order]
        axes[0].barh(y,vals,color=colors,height=.65)
        axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlim(88,99)
        axes[0].axvline(100*control,color="#17354A",linestyle="--",lw=1)
        axes[0].set_title("Partial AUROC at ≤5% FPR ↑",fontsize=12)
        axes[0].set_xlabel("Percent (axis starts at 88%)")
        axes[1].barh(y,recall,color=colors,height=.65)
        axes[1].set_yticks(y,[""]*len(y));axes[1].invert_yaxis();axes[1].set_xlim(65,100)
        axes[1].axvline(100*data["control_35paper"]["val_ai_recall_at_fpr_2pct"],color="#17354A",linestyle="--",lw=1)
        axes[1].set_title("AI recall at validation ≤2% FPR ↑",fontsize=12)
        axes[1].set_xlabel("Percent (axis starts at 65%)")
        for ax,values in ((axes[0],vals),(axes[1],recall)):
            for pos,value in enumerate(values):ax.text(value+.15,pos,f"{value:.1f}",va="center",fontsize=9)
        fig.text(.06,.07,"Dashed line = control. The paper-share differences are small; leave-one-out differences are larger.\nThis is a single-seed development experiment, not a held-out test or proof of category causality.",fontsize=11)
        pages.savefig(fig);plt.close(fig)
    table="\n".join(f'| {LABELS[x]} | {100*data[x]["val_partial_auc_fpr_5pct"]:.2f}% | {100*(data[x]["val_partial_auc_fpr_5pct"]-control):+.2f} pp | {100*data[x]["val_ai_recall_at_fpr_2pct"]:.2f}% | {100*data[x]["val_worst_domain_ai_recall_at_fpr_2pct"]:.2f}% |' for x in order)
    text=f'''# Dataset mixture pilots — diverse v3

Nine completed training runs, no Ray trial errors. All used the selected tuning parameters, 4,000 balanced training rows, 12,800 source examples seen, the same Qwen backbone/seed, and the same 800-row validation set. The control is 35% paper data within each class.

| Mixture | Partial AUROC at ≤5% FPR | Change vs control | AI recall at validation ≤2% FPR | Worst-domain recall |
| --- | ---: | ---: | ---: | ---: |
{table}

Removing papers or creative writing produced the largest drops on this validation set. Removing the other categories also lowered partial AUROC, but by less. Moving paper share from 35% to 20% or 50% changed partial AUROC by under 0.2 percentage points. Those small differences should not select a final mix by themselves.

This study compares mixtures after a fixed example budget, not a source's intrinsic quality. Removing one category changes the proportions of the others. Every row comes from existing training sources, and all nine runs use one seed and the same validation examples. Selection and repeated inspection make these development results optimistic. Confirm promising mixes on independent source families, with a fresh blind test after choosing the design.

See [chart](ablation_results_v3.pdf) and the private run summaries on F: for full metrics and adapters.
'''
    (REPO/"reports/ablation_results_v3.md").write_text(text)
    print(pdf)


if __name__=="__main__":main()
