import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from overlap import phrase_fingerprints


class OverlapTests(unittest.TestCase):
    def test_fingerprints_ignore_case_and_punctuation(self):
        words = " ".join(f"word{i}" for i in range(400))
        self.assertEqual(phrase_fingerprints(words), phrase_fingerprints(words.upper().replace(" ", ", ")))

    def test_short_text_has_no_fingerprints(self):
        self.assertEqual(phrase_fingerprints("too short to share a 24 word phrase"), set())


if __name__ == "__main__":
    unittest.main()
