"""Whole-document sliding-window diagnostics on synthetic span validation.

Thresholds use this development set's pure-human controls. These are calibration
diagnostics, not blind-test results or a production span decoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from span_data import encode_document, window_starts
from span_metrics import calibrate_threshold, summarize_scores, length_band


def threshold_for_humans(scores, target=.02):
    ordered=np.sort(scores)[::-1]
    return float(np.nextafter(ordered[min(int(target*len(ordered)),len(ordered)-1)],np.inf))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    p.add_argument("--run-name",required=True);p.add_argument("--task",choices=["token","sequence"],required=True)
    p.add_argument("--dataset-folder",default="span_pilot_v3")
    p.add_argument("--validation-file",default="val.jsonl")
    p.add_argument("--threshold",type=float,help="Frozen threshold from a separate calibration evaluation")
    p.add_argument("--output-name",default="span_validation_v3")
    p.add_argument("--calibration-unit",choices=["token","document"],default="token")
    p.add_argument("--target-fpr",type=float,default=.02)
    p.add_argument("--report-to",choices=["none","wandb"],default="none")
    args=p.parse_args();run=args.root/"runs"/args.run_name;config=json.loads((run/"run_config.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(run/"best_adapter")
    cls=AutoModelForTokenClassification if args.task=="token" else AutoModelForSequenceClassification
    base=cls.from_pretrained(config.get("base_model",config.get("model")),num_labels=2,dtype=torch.bfloat16,device_map={"":0})
    base.config.pad_token_id=tokenizer.pad_token_id
    model=PeftModel.from_pretrained(base,run/"best_adapter").eval();torch.set_num_threads(4)
    data=args.root/"data"/args.dataset_folder/args.validation_file
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    results=[]
    with torch.inference_mode():
        for row in rows:
            ids,offsets,labels=encode_document(row,tokenizer)
            total=np.zeros(len(ids),dtype=np.float64);counts=np.zeros(len(ids),dtype=np.int32)
            size=config.get("max_length",config.get("max_source_tokens",512))
            stride=config.get("stride",max(1,size//2))
            for start in window_starts(len(ids),size,stride):
                window=ids[start:start+size];n=len(window)
                seq=window+window if config["repeat2"] else window
                x=torch.tensor([seq],device="cuda");logits=model(input_ids=x,attention_mask=torch.ones_like(x)).logits.float()[0]
                if args.task=="token":
                    logits=logits[-n:];m=(logits[:,1]-logits[:,0]).cpu().numpy()
                else:m=float((logits[1]-logits[0]).cpu())
                total[start:start+n]+=m;counts[start:start+n]+=1
            assert np.all(counts>0)
            results.append({"row":row,"score":total/counts,"label":np.array(labels),"offsets":offsets})
    if args.threshold is None:
        threshold=calibrate_threshold(results,args.target_fpr,args.calibration_unit)
        threshold_source=f"same-set pure-human calibration; target {args.target_fpr} {args.calibration_unit} FPR"
    else:
        threshold=args.threshold
        threshold_source="provided frozen threshold from separate calibration evaluation"
    def summarize(group):
        return summarize_scores(group,threshold)
    def mean_or_none(values):
        return float(np.mean(values)) if len(values) else None
    purehuman=[r for r in results if r["row"]["kind"]=="human"]
    error=[];mixed_error=[];false_chars=human_chars=0;predictions=[];ai_span_coverages=[]
    for r in results:
        mask=r["label"]!=-100;pred=r["score"]>=threshold
        weights=np.array([end-start for start,end in r["offsets"]]);weights=weights*mask
        if not weights.sum():
            predictions.append({"id":r["row"]["id"],"spans":[]})
            continue
        fraction_error=abs(float(np.sum(weights*pred)/weights.sum())-float(np.sum(weights*(r["label"]==1))/weights.sum()))
        error.append(fraction_error)
        if r["row"]["kind"]=="mixed":
            mixed_error.append(fraction_error)
            for span in r["row"]["spans"]:
                if span["label"]!=1:continue
                in_span=np.array([start<span["end"] and end>span["start"]
                                  for start,end in r["offsets"]]) & mask
                if in_span.any():
                    ai_span_coverages.append(float(np.sum(weights[in_span]*pred[in_span])/np.sum(weights[in_span])))
        false_chars+=int(np.sum(weights*(r["label"]==0)*pred));human_chars+=int(np.sum(weights*(r["label"]==0)))
        spans=[]
        for (start,end),label,valid in zip(r["offsets"],pred,mask):
            if not valid:continue
            if spans and spans[-1]["label"]==int(label) and spans[-1]["end"]==start:spans[-1]["end"]=end
            else:spans.append({"start":start,"end":end,"label":int(label)})
        predictions.append({"id":r["row"]["id"],"spans":spans})
    size=config.get("max_length",config.get("max_source_tokens",512))
    stride=config.get("stride",max(1,size//2))
    report={"run_name":args.run_name,"role":"span evaluation; see dataset manifest for provenance and holdout status",
        "dataset_folder":args.dataset_folder,"validation_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "task":args.task,"repeat2":config["repeat2"],"window_size":size,
        "window_stride":stride,"threshold":threshold,
        "threshold_source":threshold_source,
        "total_source_tokens":sum(len(r["label"]) for r in results),
        "ignored_source_tokens":sum(int(np.sum(r["label"]==-100)) for r in results),
        "labeled_token_fraction":sum(int(np.sum(r["label"]!=-100)) for r in results)/sum(len(r["label"]) for r in results),
        "overall":summarize(results),"human_character_false_highlight_rate":false_chars/human_chars if human_chars else None,
        "pure_human_document_any_false_highlight_rate":mean_or_none([np.any(r["score"][r["label"]==0]>=threshold) for r in purehuman]),
        "document_ai_character_fraction_mae":mean_or_none(error),
        "mixed_document_ai_character_fraction_mae":mean_or_none(mixed_error),
        "mixed_ai_spans":len(ai_span_coverages),
        "mixed_ai_span_recall_any_highlight":mean_or_none([x>0 for x in ai_span_coverages]),
        "mixed_ai_span_recall_half_covered":mean_or_none([x>=.5 for x in ai_span_coverages]),
        "by_length":{band:summarize([r for r in results if length_band(len(r["label"]))==band])
                     for band in sorted({length_band(len(r["label"])) for r in results})},
        "by_kind":{k:summarize([r for r in results if r["row"]["kind"]==k]) for k in ("human","ai","mixed")},
        "by_construction":{c:summarize([r for r in results if r["row"]["construction"]==c])
                           for c in sorted({r["row"]["construction"] for r in results})},
        "by_source_family":{f:summarize([r for r in results if r["row"]["source"].split(":")[0]==f])
                            for f in sorted({r["row"]["source"].split(":")[0] for r in results})},
        "by_construction_family":{
            f"{construction}:{family}":summarize([
                r for r in results if r["row"]["construction"]==construction
                and r["row"]["source"].split(":")[0]==family])
            for construction,family in sorted({(r["row"]["construction"],
                                                r["row"]["source"].split(":")[0])
                                               for r in results})},
        "by_domain":{d:summarize([r for r in results if r["row"]["domain"]==d]) for d in sorted({r["row"]["domain"] for r in results})}}
    manifest_path=data.parent/"manifest.json"
    report["dataset_manifest_sha256"]=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    (run/f"{args.output_name}.json").write_text(json.dumps(report,indent=2)+"\n")
    (run/f"{args.output_name}_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in predictions))
    np.savez_compressed(run/f"{args.output_name}_scores.npz",
        score=np.concatenate([r["score"][r["label"]!=-100] for r in results]).astype(np.float32),
        label=np.concatenate([r["label"][r["label"]!=-100] for r in results]).astype(np.int8),
        document_offsets=np.cumsum([0]+[int(np.sum(r["label"]!=-100)) for r in results]),
        document_ids=np.array([r["row"]["id"] for r in results]))
    if args.report_to=="wandb":
        import wandb
        wb=wandb.init(project="pangram-at-home",name=args.run_name+"_span_validation",job_type="span-validation",config={"parent_run":args.run_name})
        wb.summary.update(report);wb.finish()
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
