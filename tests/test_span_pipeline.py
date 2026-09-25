import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from span_data import window_starts, token_window, encode_document


class SpanPipelineTests(unittest.TestCase):
    def test_window_coverage_and_end_anchor(self):
        for n in (1,511,512,513,768,769,1800):
            starts=window_starts(n);covered=set()
            for s in starts:covered.update(range(s,min(s+512,n)))
            self.assertEqual(covered,set(range(n)))
            self.assertEqual(starts[-1],max(0,n-512))
            self.assertEqual(len(starts),len(set(starts)))

    def test_second_copy_only_without_label_shift(self):
        x=token_window([10,20,30],[0,1,0])
        self.assertEqual(x["input_ids"],[10,20,30,10,20,30])
        self.assertEqual(x["labels"],[-100,-100,-100,0,1,0])
        self.assertEqual(sum(y!=-100 for y in x["labels"]),3)

    def test_ambiguous_boundary_is_not_given_false_certainty(self):
        class Tokenizer:
            def __call__(self,*args,**kwargs):
                return {"input_ids":[1,2,3,4],"offset_mapping":[(0,3),(3,5),(5,8),(8,10)]}
        row={"text":"abcdefghij","spans":[{"start":0,"end":4,"label":0},{"start":4,"end":8,"label":1}]}
        _,_,labels=encode_document(row,Tokenizer())
        self.assertEqual(labels,[0,-100,1,-100])


if __name__=="__main__":unittest.main()
