"""CPU metrics shared by span evaluation and frozen-threshold comparisons."""
import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate_threshold(results, target=.02, unit="token"):
    if not 0 <= target < 1:
        raise ValueError("False-positive target must be in [0, 1)")
    if unit not in {"token", "document"}:
        raise ValueError("Calibration unit must be token or document")
    humans = [np.asarray(r["score"], dtype=np.float64)[r["label"] == 0]
              for r in results if r["row"]["kind"] == "human"]
    humans = [x for x in humans if len(x)]
    if not humans:
        raise ValueError("Calibration requires labeled pure-human documents")
    values = np.concatenate(humans) if unit == "token" else np.array([x.max() for x in humans])
    ordered = np.sort(values)[::-1]
    # Scores are always compared as float64, including when read from caches.
    return float(np.nextafter(ordered[int(np.floor(target * len(ordered)))], np.inf))


def summarize_scores(group, threshold):
    usable = [r for r in group if np.any(r["label"] != -100)]
    if not usable:
        return {"documents": len(group), "tokens": 0, "human_tokens": 0, "ai_tokens": 0,
                "false_positive_tokens": 0, "true_positive_tokens": 0,
                "fpr": None, "ai_recall": None, "precision": None, "roc_auc": None,
                "pure_human_documents": 0, "pure_human_documents_with_false_highlight": 0,
                "pure_human_document_any_false_highlight_rate": None}
    y = np.concatenate([r["label"][r["label"] != -100] for r in usable])
    s = np.concatenate([np.asarray(r["score"], dtype=np.float64)[r["label"] != -100] for r in usable])
    pred = s >= threshold
    human, ai = y == 0, y == 1
    pure = [r for r in usable if r["row"]["kind"] == "human"]
    doc_fp = sum(bool(np.any(np.asarray(r["score"], dtype=np.float64)[r["label"] == 0] >= threshold)) for r in pure)
    return {"documents": len(group), "tokens": len(y), "human_tokens": int(human.sum()),
            "ai_tokens": int(ai.sum()), "false_positive_tokens": int((pred & human).sum()),
            "true_positive_tokens": int((pred & ai).sum()),
            "fpr": float(pred[human].mean()) if human.any() else None,
            "ai_recall": float(pred[ai].mean()) if ai.any() else None,
            "precision": float(ai[pred].mean()) if pred.any() else None,
            "roc_auc": float(roc_auc_score(y, s)) if human.any() and ai.any() else None,
            "pure_human_documents": len(pure), "pure_human_documents_with_false_highlight": doc_fp,
            "pure_human_document_any_false_highlight_rate": doc_fp / len(pure) if pure else None}


def length_band(n):
    return "0001-0512" if n <= 512 else "0513-1024" if n <= 1024 else "1025-2048" if n <= 2048 else "2049+"
