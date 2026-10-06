import unittest
import numpy as np
from calibrated_report import analyze_rows, doc_maxima


def row(split, label, slot='human_paper', condition=None):
    return {'split': split, 'label': label, 'slot': slot, 'condition': condition}


class DocumentAnyTests(unittest.TestCase):
    def setUp(self):
        # 100 human rows per half, each with one hot sentence; dev and test share the same score distribution.
        hot = np.linspace(0, .99, 100)
        self.rows = [row(s, 'human') for s in ('dev', 'test') for _ in hot]
        self.sents = [np.array([[.01, 0], [h, 0], [.02, 0]]) for _ in ('dev', 'test') for h in hot]
        self.rows.append(row('test', 'mixed', 'edit_context', 'sentence')); self.sents.append(np.array([[.995, 1], [.0, 0]]))

    def test_doc_maxima_uses_only_pure_human_rows_of_one_split(self):
        self.assertEqual(len(doc_maxima(self.rows, self.sents, 'test')), 100)
        self.assertAlmostEqual(doc_maxima(self.rows, self.sents, 'dev').max(), .99)

    def test_document_cutoff_bounds_test_documents_not_sentences(self):
        out = analyze_rows(np.zeros(1000), self.rows, self.sents)['0.020']
        # Easy calibration text gives a sentence cutoff that flags nearly every human document.
        self.assertGreater(out['test_human_docs_any_highlight'], .9)
        self.assertLessEqual(out['document_any']['test_human_docs_any_highlight'], .02)
        self.assertEqual(out['document_any']['recall_one'], 1.0)
        self.assertEqual(out['document_any']['dev_human_docs'], 100)

    def test_missing_dev_rows_skip_document_cutoff(self):
        rows = [r for r in self.rows if r['split'] == 'test']; sents = self.sents[100:]
        self.assertNotIn('document_any', analyze_rows(np.zeros(10), rows, sents)['0.010'])


if __name__ == '__main__':
    unittest.main()
