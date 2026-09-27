"""Full-backbone attribution comparison on the exact frozen-head splits.

Each task starts independently from the merged v10 detector. Training samples one
window per document per epoch; evaluation pools the same windows as the frozen
head. Validation chooses the checkpoint, and test is read only at the end.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train_attribution_heads_v1 import (ADAPTER, BASE, DATA, MAX_SOURCE_TOKENS,
                                        RUN, SEED, feature, read_rows, score,
                                        sha, starts)

OUT = RUN.parent/'attribution_unfrozen_v1'


def encode_eval(text, tokenizer, model):
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    vectors=[]
    for start in starts(len(ids)):
        window=ids[start:start+MAX_SOURCE_TOKENS]
        tensor=torch.tensor([window+window],device='cuda')
        with torch.inference_mode():
            hidden=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                         output_hidden_states=True,return_dict=True).hidden_states[-1]
            vectors.append(hidden[0,len(window):].float().mean(dim=0))
    return torch.stack(vectors).mean(dim=0)


def encode_train(text, tokenizer, model, rng):
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    positions=starts(len(ids))
    start=positions[rng.randrange(len(positions))]
    window=ids[start:start+MAX_SOURCE_TOKENS]
    tensor=torch.tensor([window+window],device='cuda')
    hidden=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                 output_hidden_states=True,return_dict=True).hidden_states[-1]
    return hidden[0,len(window):].float().mean(dim=0)


def evaluate(rows, labels, index, tokenizer, model, head, mean, scale, weights):
    model.eval();head.eval()
    outputs=[];actual=[];losses=[]
    for row in rows:
        vector=encode_eval(row['text'],tokenizer,model)
        logits=head(torch.clamp((vector-mean)/scale,-5,5))
        target=index[row['label']]
        outputs.append(logits.detach().cpu().numpy())
        actual.append(target)
        losses.append(float(torch.nn.functional.cross_entropy(
            logits[None],torch.tensor([target],device='cuda'),weight=weights)))
    outputs=np.stack(outputs);actual=np.asarray(actual)
    result=score(actual,outputs,labels)
    result['loss']=float(np.mean(losses))
    return result,outputs


def train_task(task, epochs, learning_rate, wb):
    rows={split:read_rows(task,split) for split in ('train','val','test')}
    labels=sorted({row['label'] for split_rows in rows.values() for row in split_rows})
    index={label:i for i,label in enumerate(labels)}
    output=OUT/task
    output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(ADAPTER)
    tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForTokenClassification.from_pretrained(
        BASE,num_labels=2,dtype=torch.bfloat16,device_map={'':0})
    base.config.pad_token_id=tokenizer.pad_token_id
    base.config.use_cache=False
    model=PeftModel.from_pretrained(base,ADAPTER).merge_and_unload()
    model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(task,'trainable backbone parameters',trainable,flush=True)
    assert trainable>1_000_000_000
    frozen=RUN/task
    norm=np.load(frozen/'normalization.npz')
    mean=torch.tensor(norm['mean'],device='cuda')
    scale=torch.tensor(norm['std']+1e-4,device='cuda')
    initial=load_file(str(frozen/'linear_head.safetensors'))
    head=torch.nn.Linear(model.config.hidden_size,len(labels),device='cuda')
    head.load_state_dict(initial)
    counts=Counter(index[row['label']] for row in rows['train'])
    weights=torch.tensor([1/counts[i] for i in range(len(labels))],device='cuda')
    weights/=weights.mean()
    optimizer=torch.optim.Adafactor([
        {'params':list(model.parameters()),'lr':learning_rate},
        {'params':list(head.parameters()),'lr':learning_rate*20}],
        weight_decay=.01)
    rng=random.Random(SEED)
    best=None
    step=0
    tic=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();head.train()
        order=list(range(len(rows['train'])))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for number,row_id in enumerate(order,1):
            row=rows['train'][row_id]
            vector=encode_train(row['text'],tokenizer,model,rng)
            logits=head(torch.clamp((vector-mean)/scale,-5,5))
            target=torch.tensor([index[row['label']]],device='cuda')
            loss=torch.nn.functional.cross_entropy(logits[None],target,weight=weights)
            (loss/16).backward()
            losses.append(float(loss.detach()))
            if number%16==0 or number==len(order):
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                step+=1
            if number%100==0 or number==len(order):
                print(task,'epoch',epoch,'train',number,'/',len(order),
                      'elapsed_s',round(time.monotonic()-tic,1),flush=True)
        val,_=evaluate(rows['val'],labels,index,tokenizer,model,head,mean,scale,weights)
        wb.log({'epoch':epoch,'train_loss':float(np.mean(losses)),
                'val_loss':val['loss'],'val_accuracy':val['accuracy'],
                'val_macro_f1':val['macro_f1'],'optimizer_steps':step})
        rank=(val['macro_f1'],-val['loss'])
        print(task,'epoch',epoch,'val_accuracy',val['accuracy'],
              'val_macro_f1',val['macro_f1'],flush=True)
        if best is None or rank>best['rank']:
            best={'rank':rank,'epoch':epoch,'val':val}
            model.save_pretrained(output/'best_backbone',safe_serialization=True)
            save_file({name:param.detach().cpu().contiguous()
                       for name,param in head.state_dict().items()},
                      str(output/'linear_head.safetensors'))
    del model,base,optimizer,head
    torch.cuda.empty_cache()
    base=AutoModelForTokenClassification.from_pretrained(
        output/'best_backbone',dtype=torch.bfloat16,device_map={'':0})
    base.config.pad_token_id=tokenizer.pad_token_id
    base.config.use_cache=False
    head=torch.nn.Linear(base.config.hidden_size,len(labels),device='cuda')
    head.load_state_dict(load_file(str(output/'linear_head.safetensors')))
    test,logits=evaluate(rows['test'],labels,index,tokenizer,base,head,mean,scale,weights)
    predictions=[{'id':row['id'],'actual':row['label'],
                  'predicted':labels[int(logit.argmax())]}
                 for row,logit in zip(rows['test'],logits)]
    (output/'test_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
    report={'task':task,'method':'full_backbone_finetune','start':'merged v10 adapter',
            'trainable_backbone_parameters':trainable,'train_rows':len(rows['train']),
            'validation_rows':len(rows['val']),'test_rows':len(rows['test']),
            'labels':labels,'epochs':epochs,'best_epoch':best['epoch'],
            'learning_rate':learning_rate,'gradient_accumulation':16,
            'training_windows_per_document_per_epoch':1,
            'evaluation_windows':'all up to 8, 512 source tokens, stride 256',
            'validation':best['val'],'test':test,'wandb_url':wb.url,
            'adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),
            'data_sha256':{split:sha(DATA/task/(split+'.jsonl')) for split in rows}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    wb.summary.update({'test_accuracy':test['accuracy'],
                       'test_macro_f1':test['macro_f1'],
                       'best_epoch':best['epoch']})
    print(task,'TEST accuracy',test['accuracy'],'macro_f1',test['macro_f1'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=('arena','authors'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.set_num_threads(4)
    tasks=(args.task,) if args.task else ('arena','authors')
    if args.smoke:
        assert all((RUN/task/'report.json').exists() for task in tasks)
        print('Frozen reports and trainable-run inputs available')
        return
    import wandb
    for task in tasks:
        epochs=2 if task=='arena' else 3
        wb=wandb.init(project='pangram-at-home',entity='eac-adsf',
                      name='attribution_unfrozen_v1_'+task,
                      job_type='full-backbone-attribution',
                      config={'task':task,'epochs':epochs,'backbone_lr':5e-6,
                              'full_backbone':True,'seed':SEED})
        try:
            train_task(task,epochs,5e-6,wb)
        finally:
            wb.finish()


if __name__=='__main__':
    main()
