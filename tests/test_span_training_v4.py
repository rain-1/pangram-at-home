import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_span_training_v4 import SourcePool, exact_excerpt, make_document


def fixture_rows():
    rows = []
    for label in (0, 1):
        for i in range(10):
            text = f"opening-{label}-{i}  " + "  ".join(f"word{j}_{label}_{i}" for j in range(220)) + f"  closing-{label}-{i}"
            rows.append({"text_id": f"fixture:{label}:{i}", "text": text,
                         "label": label, "source": "fixture:creative", "domain": "creative",
                         "source_family": "fixture", "source_id": str(i), "group_id": f"fixture:{i}",
                         "generator": "human" if label == 0 else "fixture-ai",
                         "license": "fixture", "text_sha256": str(i)})
    return rows


class SpanTrainingV4Tests(unittest.TestCase):
    def test_excerpt_is_exact_contiguous_source_text(self):
        row = fixture_rows()[0]
        piece, start, end = exact_excerpt(row, 90, random.Random(4))
        self.assertEqual(piece, row["text"][start:end])
        self.assertIn("  ", piece)
        self.assertEqual(len(piece.split()), 90)

    def test_all_document_constructions_preserve_provenance_and_labels(self):
        source = {row["text_id"]: row for row in fixture_rows()}
        pool = SourcePool(list(source.values()), 9, 5)
        for index, construction in enumerate(("unaltered_source", "same_label_short_join",
                                               "same_label_long_join", "matched_pair_short_mix",
                                               "same_source_long_mix")):
            doc = make_document(pool, "train", index, "creative", construction)
            self.assertEqual(len(doc["components"]), len(doc["spans"]))
            for component, span in zip(doc["components"], doc["spans"]):
                piece = doc["text"][component["output_start"]:component["output_end"]]
                original = source[component["text_id"]]["text"]
                self.assertEqual(piece, original[component["source_start"]:component["source_end"]])
                self.assertEqual((span["start"], span["end"], span["label"]),
                                 (component["output_start"], component["output_end"], component["label"]))
            if "same_label" in construction:
                self.assertEqual(len({s["label"] for s in doc["spans"]}), 1)
            if "mix" in construction:
                self.assertEqual({s["label"] for s in doc["spans"]}, {0, 1})

    def test_frozen_dataset_sources_do_not_cross_splits(self):
        root = Path("/mnt/f/pangram-at-home/data/span_training_v4")
        if not root.exists():
            self.skipTest("External frozen data absent")
        source_sets = {}
        for split in ("train", "val"):
            with (root / f"{split}.jsonl").open() as file:
                rows = (json.loads(line) for line in file)
                source_sets[split] = {source_id for row in rows for source_id in row["source_ids"]}
        self.assertFalse(source_sets["train"] & source_sets["val"])


if __name__ == "__main__":
    unittest.main()
