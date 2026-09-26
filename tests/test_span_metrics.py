import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from span_metrics import calibrate_threshold, summarize_scores


def row(scores, labels, kind="human"):
    return {"row": {"kind": kind}, "score": np.array(scores, dtype=np.float32),
            "label": np.array(labels)}


class SpanMetricsTests(unittest.TestCase):
    def test_tied_float32_cutoff_respects_false_positive_budget(self):
        rows = [row([.9375, .9375, .2, .1], [0] * 4)]
        threshold = calibrate_threshold(rows, .25)
        self.assertGreater(threshold, .9375)
        self.assertEqual(summarize_scores(rows, threshold)["fpr"], 0)

    def test_document_calibration_counts_documents_not_length(self):
        rows = [row([i] * (i + 1), [0] * (i + 1)) for i in range(10)]
        threshold = calibrate_threshold(rows, .2, "document")
        result = summarize_scores(rows, threshold)
        self.assertEqual(result["pure_human_documents_with_false_highlight"], 2)
        self.assertEqual(result["pure_human_document_any_false_highlight_rate"], .2)

    def test_human_only_and_empty_strata_are_valid(self):
        self.assertIsNone(summarize_scores([], 0)["fpr"])
        result = summarize_scores([row([100, -1], [-100, 0])], 0)
        self.assertEqual(result["fpr"], 0)
        self.assertIsNone(result["ai_recall"])
        self.assertIsNone(result["roc_auc"])

    def test_masked_ai_edits_do_not_affect_recall(self):
        result = summarize_scores([row([100, 2, -1], [-100, 1, 0], "mixed")], 0)
        self.assertEqual(result["tokens"], 2)
        self.assertEqual(result["ai_recall"], 1)
        self.assertEqual(result["fpr"], 0)


if __name__ == "__main__":
    unittest.main()
