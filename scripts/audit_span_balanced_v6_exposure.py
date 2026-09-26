"""Count actual training-window and supervised-token exposure by source."""
from __future__ import annotations

from collections import Counter
import argparse
import json

from transformers import AutoTokenizer

from build_span_balanced_v6 import DATA, ROOT
from span_data import encode_document,window_starts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-folder',default='span_balanced_v6')
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    docs=Counter();windows=Counter();tokens=Counter();labels=Counter()
    with (DATA/args.dataset_folder/'train.jsonl').open() as file:
        for line in file:
            row=json.loads(line)
            source=('DAMASHA' if row['source']=='DAMASHA clean published aggregate' else
                    'LLMTrace' if row['source']=='LLMTrace_detection' else row['source'])
            ids,_,y=encode_document(row,tokenizer)
            docs[source]+=1
            for start in window_starts(len(ids)):
                windows[source]+=1
                for label in y[start:start+512]:
                    if label in (0,1):
                        tokens[source]+=1
                        labels[str(label)]+=1
    total=sum(tokens.values())
    report={'documents':sum(docs.values()),'windows':sum(windows.values()),
            'supervised_token_positions':total,
            'source_documents':dict(docs),'source_windows':dict(windows),
            'source_supervised_tokens':dict(tokens),
            'source_supervised_token_fraction':{k:v/total for k,v in tokens.items()},
            'class_supervised_tokens':dict(labels)}
    assert docs['LLMTrace']/sum(docs.values())==.03
    assert windows['LLMTrace']/sum(windows.values())<=.03
    assert tokens['LLMTrace']/total<=.03
    assert max(tokens.values())/total<.35
    dest=DATA/args.dataset_folder/'exposure_audit.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('documents','windows','supervised_token_positions',
          'source_supervised_token_fraction','class_supervised_tokens')},indent=2))


if __name__=='__main__':main()
