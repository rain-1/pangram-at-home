import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_author_attribution_v1 import group_essays, partition


class AuthorSplitTests(unittest.TestCase):
    def test_crossposted_and_prefaced_versions_stay_together(self):
        body = " ".join(f"word{i}" for i in range(600))
        rows = [{"text": body, "canonical_url": "https://one.test/a", "group_id": "a"},
                {"text": "New preface. " + body, "canonical_url": "https://two.test/b", "group_id": "b"},
                {"text": "Unrelated short essay", "canonical_url": "https://three.test/c", "group_id": "c"}]
        self.assertEqual(sorted(map(len, group_essays(rows))), [1, 2])

    def test_newest_essays_are_reserved_and_author_conflicts_excluded(self):
        groups = [[{"author_id": "author", "document_id": str(i), "publication_date": f"2020-{i+1:02d}-01"}]
                  for i in range(10)]
        groups.append([{"author_id": "x"}, {"author_id": "y"}])
        splits, conflicts = partition(groups)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual([r["document_id"] for r in splits["test"]], ["8", "9"])
        self.assertFalse({r["probe_group_id"] for r in splits["train"]} &
                         {r["probe_group_id"] for r in splits["test"]})


if __name__ == "__main__":
    unittest.main()
