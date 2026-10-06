#!/usr/bin/env python3
"""Generate topic-matched open-ended AI science features for a v9 pilot."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/mnt/f/pangram-at-home/data/science_articles_v9')
MODELS = {
    'qwen2_5_3b': ('Qwen/Qwen2.5-3B-Instruct', '/mnt/f/pangram-at-home/models/Qwen2.5-3B-Instruct'),
    'smollm2_1_7b': ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '/mnt/f/pangram-at-home/models/SmolLM2-1.7B-Instruct'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, default='qwen2_5_3b')
    p.add_argument('--calibration-limit', type=int, default=41)
    p.add_argument('--test-limit', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--prompt-file', type=Path, default=ROOT/'matched_ai_prompts.jsonl')
    p.add_argument('--output-file', type=Path)
    p.add_argument('--max-new-tokens', type=int, default=800)
    p.add_argument('--min-new-tokens', type=int, default=300)
    args = p.parse_args()
    model_name, model_path = MODELS[args.model]
    prompts = [json.loads(line) for line in args.prompt_file.open()]
    if args.prompt_file == ROOT/'matched_ai_prompts.jsonl':
        by_split = {'calibration_ai_candidate': [], 'locked_test_ai_candidate': []}
        for row in prompts:
            by_split[row['split']].append(row)
        chosen = (by_split['calibration_ai_candidate'][:args.calibration_limit]
                  + by_split['locked_test_ai_candidate'][:args.test_limit])
    else:
        chosen = prompts
    out = args.output_file or ROOT/f'generated_{args.model}_pilot.jsonl'
    done = set()
    if out.exists():
        with out.open() as f:
            done = {json.loads(line)['human_id'] for line in f}
    chosen = [r for r in chosen if r['human_id'] not in done]
    if not chosen:
        print('All selected prompts already generated')
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to('cuda').eval()
    for offset in range(0, len(chosen), args.batch_size):
        batch = chosen[offset:offset+args.batch_size]
        rendered = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                                  tokenize=False, add_generation_prompt=True) for r in batch]
        inputs = tokenizer(rendered, return_tensors='pt', padding=True).to('cuda')
        seed = int(hashlib.sha256(f'v9:{args.model}:{offset}'.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 min_new_tokens=args.min_new_tokens,
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        with out.open('a') as f:
            for prompt, text in zip(batch, generated):
                text = re.sub(r'^\s*(?:#{1,3}\s*)?[^\n]{5,140}\n+', '', text.strip()) if text.lstrip().startswith('#') else text.strip()
                row = {'id': hashlib.sha256((args.model+':'+prompt['human_id']).encode()).hexdigest()[:20],
                       'human_id': prompt['human_id'], 'split': prompt['split'], 'source': prompt['source'],
                       'topic_title': prompt['topic_title'], 'model': model_name,
                       'prompt_version': prompt['prompt_version'], 'prompt': prompt['prompt'],
                       'seed': seed, 'text': text, 'label': 1, 'kind': 'ai',
                       'construction': 'title_topic_open_ended_generation',
                       'spans': [{'start': 0, 'end': len(text), 'label': 1}],
                       'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'human_text_sha256': prompt['human_text_sha256'],
                       'words': len(text.split())}
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(f'{offset+len(batch)}/{len(chosen)} generated', flush=True)


if __name__ == '__main__':
    main()
