"""Cache frozen v10 Repeat2 features and fit separate linear attribution heads."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(os.environ.get('PANGRAM_DATA_ROOT','/workspace/pangram-data'))
DATA = ROOT/'data/attribution_heads_v1'
RUN = ROOT/'runs/attribution_heads_v1'
BASE = ROOT/'models/Qwen3-1.7B'
ADAPTER = ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k/best_adapter'
TASKS = ('arena','authors')
SPLITS = ('train','val','test')
MAX_SOURCE_TOKENS = 512
STRIDE = 256
MAX_WINDOWS = 8
SEED = 42


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_rows(task: str, split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA/task/(split+'.jsonl')).open()]


def starts(length: int) -> list[int]:
    if length <= MAX_SOURCE_TOKENS:
        return [0]
    points=list(range(0,length-MAX_SOURCE_TOKENS+1,STRIDE))
    end=length-MAX_SOURCE_TOKENS
    if points[-1]!=end:
        points.append(end)
    if len(points)>MAX_WINDOWS:
        indices=np.linspace(0,len(points)-1,MAX_WINDOWS).round().astype(int)
        points=[points[index] for index in indices]
    return points


def feature(text: str, tokenizer, model) -> tuple[np.ndarray,int]:
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if not ids:
        raise ValueError('Empty text after tokenization')
    vectors=[]
    for start in starts(len(ids)):
        window=ids[start:start+MAX_SOURCE_TOKENS]
        sequence=window+window
        tensor=torch.tensor([sequence],device='cuda')
        with torch.inference_mode():
            output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                         output_hidden_states=True,return_dict=True)
            hidden=output.hidden_states[-1][0,len(window):]
            vectors.append(hidden.float().mean(dim=0).cpu().numpy())
    return np.mean(vectors,axis=0).astype(np.float32),len(vectors)


def encode_split(task: str, split: str, rows: list[dict], tokenizer, model) -> dict:
    task_dir=RUN/task
    task_dir.mkdir(parents=True,exist_ok=True)
    destination=task_dir/(split+'_features.npy')
    progress=task_dir/(split+'_progress.json')
    hidden=model.config.hidden_size
    source=DATA/task/(split+'.jsonl')
    source_sha=sha(source)
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['source_sha256']!=source_sha or state['rows']!=len(rows):
            raise ValueError('Feature cache source changed')
        matrix=np.lib.format.open_memmap(destination,mode='r+')
        if matrix.shape!=(len(rows),hidden):
            raise ValueError('Feature cache shape changed')
        start=state['completed']
    else:
        matrix=np.lib.format.open_memmap(destination,mode='w+',dtype=np.float32,
                                         shape=(len(rows),hidden))
        state={'task':task,'split':split,'rows':len(rows),
               'source_sha256':source_sha,'completed':0,'windows':0}
        start=0
    tic=time.monotonic()
    for i in range(start,len(rows)):
        vector,nwindows=feature(rows[i]['text'],tokenizer,model)
        matrix[i]=vector
        state['completed']=i+1
        state['windows']+=nwindows
        if (i+1)%25==0 or i+1==len(rows):
            matrix.flush()
            progress.write_text(json.dumps(state,indent=2)+'\n')
            print(task,split,i+1,'/',len(rows),'windows',state['windows'],
                  'elapsed_s',round(time.monotonic()-tic,1),flush=True)
    return state


def score(y_true: np.ndarray, logits: np.ndarray, labels: list[str]) -> dict:
    pred=logits.argmax(axis=1)
    ranking=np.argsort(logits,axis=1)[:,::-1]
    top5=np.mean([y_true[i] in ranking[i,:min(5,len(labels))]
                  for i in range(len(y_true))])
    return {'rows':len(y_true),'accuracy':float(accuracy_score(y_true,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y_true,pred)),
            'macro_f1':float(f1_score(y_true,pred,labels=list(range(len(labels))),
                                       average='macro',zero_division=0)),
            'top5_accuracy':float(top5),
            'per_label_recall':{label:float(np.mean(pred[y_true==i]==i))
                                for i,label in enumerate(labels)},
            'confusion_matrix':confusion_matrix(y_true,pred,
                              labels=list(range(len(labels)))).tolist()}


def fit_head(task: str, wb) -> dict:
    rows={split:read_rows(task,split) for split in SPLITS}
    labels=sorted({row['label'] for split in SPLITS for row in rows[split]})
    if task=='arena' and len(labels)!=50 or task=='authors' and len(labels)!=4:
        raise ValueError('Unexpected label count')
    index={label:i for i,label in enumerate(labels)}
    matrices={split:np.load(RUN/task/(split+'_features.npy')) for split in SPLITS}
    mean=matrices['train'].mean(axis=0)
    std=matrices['train'].std(axis=0)
    xs={split:torch.from_numpy(np.clip((matrices[split]-mean)/(std+1e-4),-5,5)).float().cuda()
        for split in SPLITS}
    ys={split:torch.tensor([index[row['label']] for row in rows[split]],device='cuda')
        for split in SPLITS}
    counts=Counter(int(y) for y in ys['train'].cpu().tolist())
    weights=torch.tensor([1/counts[i] for i in range(len(labels))],device='cuda')
    weights=weights/weights.mean()
    classifier=torch.nn.Linear(xs['train'].shape[1],len(labels)).cuda()
    optimizer=torch.optim.AdamW(classifier.parameters(),lr=.001,weight_decay=.01)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    rng=torch.Generator(device='cpu').manual_seed(SEED)
    best=None
    wait=0
    batch=256 if task=='arena' else 64
    for epoch in range(1,151):
        classifier.train()
        permutation=torch.randperm(len(xs['train']),generator=rng).cuda()
        train_losses=[]
        for start in range(0,len(permutation),batch):
            selected=permutation[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(classifier(xs['train'][selected]),ys['train'][selected])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        classifier.eval()
        with torch.inference_mode():
            val_logits=classifier(xs['val'])
            val_loss=float(loss_fn(val_logits,ys['val']))
            val_score=score(ys['val'].cpu().numpy(),val_logits.cpu().numpy(),labels)
        wb.log({'epoch':epoch,'train_loss':float(np.mean(train_losses)),
                'val_loss':val_loss,'val_macro_f1':val_score['macro_f1'],
                'val_accuracy':val_score['accuracy']})
        rank=(val_score['macro_f1'],-val_loss)
        if best is None or rank>best['rank']:
            best={'rank':rank,'epoch':epoch,'state':{k:v.detach().cpu().clone()
                                                   for k,v in classifier.state_dict().items()}}
            wait=0
        else:
            wait+=1
        if epoch%10==0 or epoch==1:
            print(task,'epoch',epoch,'train_loss',round(np.mean(train_losses),4),
                  'val_macro_f1',round(val_score['macro_f1'],4),flush=True)
        if wait>=20:
            break
    classifier.load_state_dict(best['state'])
    classifier.eval()
    outputs={}
    for split in ('val','test'):
        with torch.inference_mode():
            logits=classifier(xs[split]).cpu().numpy()
        actual=ys[split].cpu().numpy()
        outputs[split]=score(actual,logits,labels)
        predictions=[{'id':row['id'],'actual':row['label'],
                      'predicted':labels[int(logit.argmax())],
                      'correct':bool(logit.argmax()==actual[i])}
                     for i,(row,logit) in enumerate(zip(rows[split],logits))]
        (RUN/task/(split+'_predictions.jsonl')).write_text(''.join(
            json.dumps(row)+'\n' for row in predictions))
    save_file({key:value.contiguous() for key,value in best['state'].items()},
              str(RUN/task/'linear_head.safetensors'))
    np.savez_compressed(RUN/task/'normalization.npz',mean=mean,std=std)
    report={'task':task,'labels':labels,'best_epoch':best['epoch'],
            'train_rows':len(rows['train']),'validation':outputs['val'],
            'test':outputs['test'],'adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),
            'base_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
            'feature_recipe':{'repeat2':True,'source_tokens':MAX_SOURCE_TOKENS,
                              'stride':STRIDE,'max_windows_per_document':MAX_WINDOWS,
                              'pooling':'mean second-copy final hidden state per window, then document mean'},
            'wandb_url':wb.url,
            'data_sha256':{split:sha(DATA/task/(split+'.jsonl')) for split in SPLITS}}
    (RUN/task/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    wb.summary.update({'test_accuracy':outputs['test']['accuracy'],
                       'test_macro_f1':outputs['test']['macro_f1'],
                       'test_balanced_accuracy':outputs['test']['balanced_accuracy'],
                       'best_epoch':best['epoch']})
    print(task,'test accuracy',outputs['test']['accuracy'],
          'macro_f1',outputs['test']['macro_f1'],flush=True)
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=TASKS,help='Run only one task; default runs both')
    parser.add_argument('--features-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Encode one document and exit')
    args=parser.parse_args()
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    tasks=(args.task,) if args.task else TASKS
    tokenizer=AutoTokenizer.from_pretrained(ADAPTER)
    tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForTokenClassification.from_pretrained(
        BASE,num_labels=2,dtype=torch.bfloat16,device_map={'':0})
    base.config.pad_token_id=tokenizer.pad_token_id
    base.config.use_cache=False
    model=PeftModel.from_pretrained(base,ADAPTER).eval()
    if args.smoke:
        row=read_rows(tasks[0],'train')[0]
        vector,nwindows=feature(row['text'],tokenizer,model)
        print({'task':tasks[0],'feature_shape':vector.shape,'windows':nwindows,
               'finite':bool(np.isfinite(vector).all())})
        return
    for task in tasks:
        for split in SPLITS:
            encode_split(task,split,read_rows(task,split),tokenizer,model)
    del model,base
    torch.cuda.empty_cache()
    if args.features_only:
        return
    import wandb
    for task in tasks:
        wb=wandb.init(project='pangram-at-home',entity='eac-adsf',
                      name='attribution_heads_v1_'+task,job_type='frozen-attribution-head',
                      config={'task':task,'backbone':'qwen3_token_repeat2_essay_paired_v10_20k',
                              'frozen_backbone':True,'seed':SEED})
        try:
            fit_head(task,wb)
        finally:
            wb.finish()


if __name__=='__main__':
    main()
