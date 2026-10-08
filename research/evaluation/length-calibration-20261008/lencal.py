"""Length-aware aggregation rules for shared_windows_v1 document scores.

Read-only over the eval repo. Recomputes document scores from per-window predictions under
several aggregation rules, re-calibrates each on the benchmark's 3000 human calibration documents
(same order-statistic rule as aidet_eval.calibrate: thr = sorted_desc[floor(alpha*n)], flag iff
score > thr), and reports per-panel human_fpr / fully_ai_recall / mixed_recall on evaluation docs.

Usage: python lencal.py <eval_repo> <out_dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RUN = REPO / "artifacts/runs/ai-detector-panels-v3__577af7a44f"
BENCH = REPO / "artifacts/benchmark-v3"
PROTO = "shared_windows_v1"
MODELS = ["moe_base_s1_mean", "moe_qwen_t21a2_mean", "moe_nemo_t21a2_mean", "open_pangram_llama"]
ALPHAS = {"human_fpr_1pct": 0.01, "human_fpr_5pct": 0.05}
# rule 2 bins (by number of scored windows); bins with < MIN_BIN calibration humans are merged downward
BINS = [(1, 1), (2, 3), (4, 7), (8, 10**6)]
MIN_BIN = 200
RULES = ["max", "binned_max", "binned_max_nomerge", "per_window_n", "top2_mean", "mean"]


def thr_desc(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, float))[::-1]
    k = math.floor(alpha * len(s))
    return float(s[k])


def load_window_scores(model: str) -> dict[str, float]:
    out: dict[str, tuple[str, float | None]] = {}
    with open(RUN / "predictions" / model / f"{PROTO}.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["input_id"]] = (r["status"], r["score"])  # latest record wins, as in docpreds.read_shard
    return out


def doc_table(model: str, win: pd.DataFrame) -> pd.DataFrame:
    rec = load_window_scores(model)
    st = win["window_id"].map(lambda w: rec.get(w, ("missing", None))[0])
    sc = win["window_id"].map(lambda w: rec.get(w, ("missing", None))[1])
    w = pd.DataFrame({"example_id": win["example_id"], "status": st, "score": sc})
    bad = w.groupby("example_id")["status"].apply(lambda s: bool(s.isin(["error", "missing"]).any()))
    ok = w[w["status"] == "ok"].copy()
    ok["score"] = ok["score"].astype(float)
    g = ok.sort_values("score", ascending=False).groupby("example_id")["score"]
    d = pd.DataFrame({"max": g.max(), "mean": g.mean(), "n_ok": g.size()})
    second = ok.sort_values("score", ascending=False).groupby("example_id").nth(1).set_index("example_id")["score"]
    d["second"] = second.reindex(d.index)
    d["top2_mean"] = np.where(d["n_ok"] >= 2, (d["max"] + d["second"]) / 2, d["max"])
    d["error"] = bad.reindex(d.index).fillna(False)
    return d, ok


def bin_of(n: int, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


def merge_bins(cal_n: np.ndarray, min_n: int):
    bins = list(BINS)
    while True:
        counts = [int(((cal_n >= lo) & (cal_n <= hi)).sum()) for lo, hi in bins]
        small = [i for i, c in enumerate(counts) if c < min_n]
        if not small or len(bins) == 1:
            return bins, counts
        i = small[-1]
        j = i - 1 if i > 0 else i + 1
        a, b = sorted((i, j))
        bins[a] = (bins[a][0], bins[b][1])
        del bins[b]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(BENCH / "examples.parquet", columns=["example_id", "role", "authorship", "panel_ids"])
    win = pd.read_parquet(BENCH / "windows" / f"{PROTO}.parquet", columns=["window_id", "example_id"])
    nwin = win.groupby("example_id").size()
    ex["n_windows"] = ex["example_id"].map(nwin).fillna(0).astype(int)
    ex["panels"] = ex["panel_ids"].map(json.loads)
    cal = ex[(ex["role"] == "calibration") & (ex["authorship"] == "human")]
    ev = ex[ex["role"] == "evaluation"]
    panels = sorted({p for ps in ev["panels"] for p in ps})

    # window-count distribution per panel
    wc = []
    for p in panels + ["calibration_human"]:
        sub = (cal if p == "calibration_human" else ev[ev["panels"].map(lambda ps: p in ps)])
        for cls in ["all", "human", "ai", "mixed"]:
            s = sub if cls == "all" else sub[sub["authorship"] == cls]
            if len(s) == 0:
                continue
            n = s["n_windows"]
            wc.append({"panel_id": p, "authorship": cls, "n_docs": len(s), "median_windows": float(n.median()),
                       "p90_windows": float(n.quantile(0.9)), "max_windows": int(n.max()),
                       "frac_ge4": float((n >= 4).mean()), "frac_ge8": float((n >= 8).mean())})
    pd.DataFrame(wc).to_csv(OUT / "window_counts.csv", index=False)

    rows, calrows = [], []
    for model in MODELS:
        if not (RUN / "predictions" / model / f"{PROTO}.jsonl").exists():
            print("skip", model)
            continue
        d, okw = doc_table(model, win)
        d = d[~d["error"]]
        cd = d.reindex(cal["example_id"]).dropna(subset=["max"])
        assert len(cd) == len(cal), (model, len(cd))
        cal_n = cd["n_ok"].to_numpy()
        cal_win = okw[okw["example_id"].isin(set(cal["example_id"]))]["score"].to_numpy()
        merged, merged_counts = merge_bins(cal_n, MIN_BIN)
        evd = ev.merge(d, left_on="example_id", right_index=True, how="left")
        for op, alpha in ALPHAS.items():
            flags_cal, flags_ev = {}, {}
            # 1/4/5: single global threshold on a document score
            for rule, col in (("max", "max"), ("top2_mean", "top2_mean"), ("mean", "mean")):
                t = thr_desc(cd[col].to_numpy(), alpha)
                flags_cal[rule] = cd[col].to_numpy() > t
                flags_ev[rule] = evd[col].to_numpy() > t
                calrows.append({"model_id": model, "op": op, "rule": rule, "bin": "all", "n_cal": len(cd), "threshold": t,
                                "cal_fpr": float(flags_cal[rule].mean())})
            # 2: per-bin max thresholds
            for rule, bins in (("binned_max", merged), ("binned_max_nomerge", BINS)):
                bc = np.array([bin_of(n, bins) for n in cal_n])
                be = np.array([bin_of(int(n), bins) if n == n else -1 for n in evd["n_ok"].to_numpy()])
                fc = np.zeros(len(cd), bool)
                fe = np.zeros(len(evd), bool)
                for i, (lo, hi) in enumerate(bins):
                    s = cd["max"].to_numpy()[bc == i]
                    t = thr_desc(s, alpha)
                    fc[bc == i] = s > t
                    fe[be == i] = evd["max"].to_numpy()[be == i] > t
                    calrows.append({"model_id": model, "op": op, "rule": rule, "bin": f"{lo}-{hi if hi < 10**6 else 'inf'}",
                                    "n_cal": int((bc == i).sum()), "threshold": t, "cal_fpr": float((s > t).mean())})
                flags_cal[rule], flags_ev[rule] = fc, fe
            # 3: per-window threshold for count n: per-window FPR p_n = 1-(1-alpha)^(1/n) on pooled cal windows
            sw = np.sort(cal_win)[::-1]

            def tn(n):
                p = 1 - (1 - alpha) ** (1 / n)
                return float(sw[math.floor(p * len(sw))])
            cache = {}
            def t_of(n):
                if n not in cache:
                    cache[n] = tn(int(n))
                return cache[n]
            flags_cal["per_window_n"] = np.array([m > t_of(n) for m, n in zip(cd["max"], cal_n)])
            flags_ev["per_window_n"] = np.array([(m > t_of(n)) if n == n else False for m, n in zip(evd["max"], evd["n_ok"])])
            for n in sorted(set(int(x) for x in cal_n)):
                sel = cal_n == n
                calrows.append({"model_id": model, "op": op, "rule": "per_window_n", "bin": str(n), "n_cal": int(sel.sum()),
                                "threshold": t_of(n), "cal_fpr": float(flags_cal["per_window_n"][sel].mean())})
            calrows.append({"model_id": model, "op": op, "rule": "per_window_n", "bin": "all", "n_cal": len(cd),
                            "threshold": None, "cal_fpr": float(flags_cal["per_window_n"].mean()), "n_cal_windows": len(sw)})
            for rule in ["binned_max", "binned_max_nomerge"]:
                calrows.append({"model_id": model, "op": op, "rule": rule, "bin": "all", "n_cal": len(cd), "threshold": None,
                                "cal_fpr": float(flags_cal[rule].mean())})
            scored = evd["max"].notna().to_numpy()
            if op == "human_fpr_1pct":
                fl = pd.DataFrame({r: flags_ev[r] for r in RULES})
                fl.insert(0, "n_ok", evd["n_ok"].to_numpy())
                fl.insert(0, "example_id", evd["example_id"].to_numpy())
                fl.to_csv(OUT / f"doc_flags_1pct__{model}.csv.gz", index=False)
            for p in panels:
                pm = evd["panels"].map(lambda ps: p in ps).to_numpy()
                for metric, cls in (("human_fpr", "human"), ("fully_ai_recall", "ai"), ("mixed_recall", "mixed")):
                    m = pm & scored & (evd["authorship"] == cls).to_numpy()
                    if m.sum() == 0:
                        continue
                    for rule in RULES:
                        f = flags_ev[rule][m]
                        rows.append({"model_id": model, "op": op, "panel_id": p, "metric": metric, "rule": rule,
                                     "numerator": int(f.sum()), "denominator": int(m.sum()), "value": float(f.mean())})
        print("done", model, "merged bins:", merged, merged_counts, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "panel_metrics.csv", index=False)
    pd.DataFrame(calrows).to_csv(OUT / "calibration_thresholds.csv", index=False)


if __name__ == "__main__":
    main()
