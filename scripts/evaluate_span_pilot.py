"""Whole-document sliding-window diagnostics on synthetic span validation.

Thresholds use this development set's pure-human controls. These are calibration
diagnostics, not blind-test results or a production span decoder.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from span_data import encode_document, window_starts


def threshold_for_humans(scores, target=.02):
    ordered=np.sort(scores)[::-1]
    return float(np.nextafter(ordered[min(int(target*len(ordered)),len(ordered)-1)],np.inf))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    p.add_argument("--run-name",required=True);p.add_argument("--task",choices=["token","sequence"],required=True)
    p.add_argument("--report-to",choices=["none","wandb"],default="none")
    args=p.parse_args();run=args.root/"runs"/args.run_name;config=json.loads((run/"run_config.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(run/"best_adapter")
    cls=AutoModelForTokenClassification if args.task=="token" else AutoModelForSequenceClassification
    base=cls.from_pretrained(config.get("base_model",config.get("model")),num_labels=2,dtype=torch.bfloat16,device_map={"":0})
    base.config.pad_token_id=tokenizer.pad_token_id
    model=PeftModel.from_pretrained(base,run/"best_adapter").eval();torch.set_num_threads(4)
    rows=[json.loads(line) for line in (args.root/"data/span_pilot_v1/val.jsonl").read_text().splitlines()]
    results=[]
    with torch.inference_mode():
        for row in rows:
            ids,offsets,labels=encode_document(row,tokenizer)
            total=np.zeros(len(ids),dtype=np.float64);counts=np.zeros(len(ids),dtype=np.int32)
            for start in window_starts(len(ids)):
                window=ids[start:start+512];n=len(window)
                seq=window+window if config["repeat2"] else window
                x=torch.tensor([seq],device="cuda");logits=model(input_ids=x,attention_mask=torch.ones_like(x)).logits.float()[0]
                if args.task=="token":
                    logits=logits[-n:];m=(logits[:,1]-logits[:,0]).cpu().numpy()
                else:m=float((logits[1]-logits[0]).cpu())
                total[start:start+n]+=m;counts[start:start+n]+=1
            assert np.all(counts>0)
            results.append({"row":row,"score":total/counts,"label":np.array(labels),"offsets":offsets})
    humans=np.concatenate([r["score"][r["label"]==0] for r in results if r["row"]["kind"]=="human"])
    threshold=threshold_for_humans(humans)
    def summarize(group):
        y=np.concatenate([r["label"][r["label"]!=-100] for r in group])
        s=np.concatenate([r["score"][r["label"]!=-100] for r in group]);pred=s>=threshold
        return {"documents":len(group),"tokens":len(y),"fpr":float(pred[y==0].mean()) if (y==0).any() else None,
            "ai_recall":float(pred[y==1].mean()) if (y==1).any() else None,
            "precision":float((y[pred]==1).mean()) if pred.any() else None,
            "roc_auc":float(roc_auc_score(y,s)) if len(set(y))==2 else None}
    purehuman=[r for r in results if r["row"]["kind"]=="human"]
    error=[];false_chars=human_chars=0;predictions=[]
    for r in results:
        mask=r["label"]!=-100;pred=r["score"]>=threshold
        weights=np.array([end-start for start,end in r["offsets"]]);weights=weights*mask
        error.append(abs(float(np.sum(weights*pred)/weights.sum())-float(np.sum(weights*(r["label"]==1))/weights.sum())))
        false_chars+=int(np.sum(weights*(r["label"]==0)*pred));human_chars+=int(np.sum(weights*(r["label"]==0)))
        spans=[]
        for (start,end),label,valid in zip(r["offsets"],pred,mask):
            if not valid:continue
            if spans and spans[-1]["label"]==int(label):spans[-1]["end"]=end
            else:spans.append({"start":start,"end":end,"label":int(label)})
        predictions.append({"id":r["row"]["id"],"spans":spans})
    report={"run_name":args.run_name,"role":"synthetic validation calibration diagnostics; not a blind test",
        "task":args.task,"repeat2":config["repeat2"],"threshold":threshold,
        "threshold_source":"same validation pure-human control tokens; target 2% token FPR",
        "overall":summarize(results),"human_character_false_highlight_rate":false_chars/human_chars,
        "pure_human_document_any_false_highlight_rate":float(np.mean([np.any(r["score"][r["label"]==0]>=threshold) for r in purehuman])),
        "document_ai_character_fraction_mae":float(np.mean(error)),
        "by_kind":{k:summarize([r for r in results if r["row"]["kind"]==k]) for k in ("human","ai","mixed")},
        "by_domain":{d:summarize([r for r in results if r["row"]["domain"]==d]) for d in sorted({r["row"]["domain"] for r in results})}}
    (run/"span_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    (run/"span_validation_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in predictions))
    if args.report_to=="wandb":
        import wandb
        wb=wandb.init(project="pangram-at-home",name=args.run_name+"_span_validation",job_type="span-validation",config={"parent_run":args.run_name})
        wb.summary.update(report);wb.finish()
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
