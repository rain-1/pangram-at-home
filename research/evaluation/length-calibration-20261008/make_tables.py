"""Baseline check + markdown tables from lencal.py outputs. Usage: python make_tables.py <eval_repo> <out_dir>"""
import json
import sys
from pathlib import Path

import pandas as pd


def md(df, index=True, floatfmt=".1f"):
    df = df.reset_index() if index else df
    def f(v):
        if isinstance(v, float):
            return "" if v != v else format(v, floatfmt)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUN = REPO / "artifacts/runs/ai-detector-panels-v3__577af7a44f"
pm = pd.read_csv(OUT / "panel_metrics.csv")
ct = pd.read_csv(OUT / "calibration_thresholds.csv", float_precision="round_trip")
wc = pd.read_csv(OUT / "window_counts.csv")
base = pd.read_csv(OUT / "baseline_cells.csv")
base = base[base["task_id"].isna()]
cal = {o["calibration_id"]: o for o in json.load(open(RUN / "calibration.json"))["operating_points"]}
out = []

# --- baseline reproduction
out.append("## Baseline reproduction (rule `max` vs results.summary.cells.csv)\n")
chk = []
for (m, op), g in pm[pm["rule"] == "max"].groupby(["model_id", "op"]):
    b = base[(base["model_id"] == m) & (base["operating_point_id"] == op)]
    j = g.merge(b, on=["panel_id", "metric"], suffixes=("", "_ref"), how="outer")
    exact = ((j["numerator"] == j["numerator_ref"]) & (j["denominator"] == j["denominator_ref"])).sum()
    t_ref = cal[f"{m}|shared_windows_v1|{op}"]["threshold"]
    t = ct[(ct.model_id == m) & (ct.op == op) & (ct.rule == "max")]["threshold"].iloc[0]
    chk.append({"model": m, "op": op, "cells": len(j), "exact_num_den_match": int(exact),
                "max_abs_diff": float((j["value"] - j["value_ref"]).abs().max()), "threshold_equal": t == t_ref})
out.append(md(pd.DataFrame(chk), index=False) + "\n")

# --- window counts
w = wc[wc["authorship"] == "all"][["panel_id", "n_docs", "median_windows", "p90_windows", "max_windows", "frac_ge4", "frac_ge8"]]
w = w.sort_values("median_windows")
out.append("## Window counts per panel (all evaluation docs in panel; calibration_human = calibration set)\n")
out.append(md(w, index=False, floatfmt=".2f") + "\n")

RULES = ["max", "binned_max", "binned_max_nomerge", "per_window_n", "top2_mean", "mean"]
SHORT = {"human_fpr": "fpr", "fully_ai_recall": "ai_rec", "mixed_recall": "mix_rec"}
for op in ["human_fpr_1pct", "human_fpr_5pct"]:
    out.append(f"## Per-panel results, {op} (values in %; recalibrated to target on the 3000 calibration humans)\n")
    for m in pm["model_id"].unique():
        g = pm[(pm.model_id == m) & (pm.op == op)].copy()
        g["row"] = g["panel_id"] + " / " + g["metric"].map(SHORT) + " (n=" + g["denominator"].astype(str) + ")"
        t = g.pivot_table(index="row", columns="rule", values="value", sort=False)[RULES] * 100
        c = ct[(ct.model_id == m) & (ct.op == op) & (ct.bin == "all")].set_index("rule")["cal_fpr"].reindex(RULES) * 100
        t.loc["(calibration humans, realised FPR)"] = c.values
        out.append(f"### {m}\n")
        out.append(md(t, floatfmt=".1f") + "\n")
    if op == "human_fpr_1pct":
        out.append("### Calibration documents and thresholds per bin (binned rules, 1%)\n")
        b = ct[(ct.op == op) & ct.rule.isin(["binned_max", "binned_max_nomerge"]) & (ct.bin != "all")]
        out.append(md(b[["model_id", "rule", "bin", "n_cal", "threshold", "cal_fpr"]], index=False, floatfmt=".4g") + "\n")
        p = ct[(ct.op == op) & (ct.rule == "per_window_n") & (ct.bin != "all")]
        p = p.pivot_table(index="bin", columns="model_id", values="threshold")
        p.index = p.index.astype(int)
        n = ct[(ct.op == op) & (ct.rule == "per_window_n") & (ct.bin != "all") & (ct.model_id == "moe_base_s1_mean")].set_index("bin")["n_cal"]
        n.index = n.index.astype(int)
        p.insert(0, "n_cal_docs", n)
        out.append("### per_window_n thresholds t(n) by window count (1%)\n")
        out.append(md(p.sort_index(), floatfmt=".4g") + "\n")
(OUT / "tables.md").write_text("\n".join(out))
print("\n".join(out[:4]))
