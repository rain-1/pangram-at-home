"""Score an exported aidet_eval benchmark (export_inputs.py) with a sweep checkpoint; one GPU per shard, resumable.

Per input it records the mean token AI probability over non-whitespace tokens (the workbench document score), the maximum
sentence score (sentence = mean of its tokens, as in sweep_eval) and the maximum token score. Scoring windows follow
sweep_eval.predict (510-token Repeat2 windows, stride 256, overlaps averaged). native_truncate_v1 inputs are cut to the first
510 tokens, as that protocol requires; shared_windows_v1 inputs (<=192 words) always fit. Run from the sweep root:
  python benchmark/score_benchmark.py TAG CHECKPOINT INPUTS.jsonl.gz OUT_DIR --shard I --nshards N
"""
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import ROOT, require_space  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import sweep_eval as E  # noqa: E402

LIMIT = 510


def main(a):
    require_space(); d = ROOT / 'sweeps' / a.tag; run = json.loads((d / 'run.json').read_text()); ck = d / a.checkpoint
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); dest = out / f'{a.tag}--{ck.stem}--{a.shard:02d}of{a.nshards:02d}.jsonl'
    rows = [json.loads(l) for l in gzip.open(a.inputs, 'rt')]
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r['input_id'])) if i % a.nshards == a.shard]
    done = {json.loads(l)['input_id'] for l in open(dest)} if dest.exists() else set()
    rows = [r for r in rows if r['input_id'] not in done]
    model, tok = E.load_model(run); w = load_file(str(ck)); E.align_expert_nesting(w, model)
    assert set(w) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in w:
                p.copy_(w[n].to(p.dtype))
    meta = {'tag': a.tag, 'checkpoint': ck.name, 'checkpoint_sha256': hashlib.sha256(ck.read_bytes()).hexdigest(),
            'inputs_sha256': hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(), 'scorer': 'score_benchmark.py v1'}
    (out / f'{a.tag}--{ck.stem}.meta.json').write_text(json.dumps(meta, indent=1))
    rows.sort(key=lambda r: len(r['text'])); t0 = time.time()
    for c in range(0, len(rows), 256):
        chunk = []
        for r in rows[c:c + 256]:
            enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
            n = len(enc['input_ids']); cut = r['text'] if n <= LIMIT or r['protocol_id'] != 'native_truncate_v1' else r['text'][:enc['offset_mapping'][LIMIT - 1][1]]
            chunk.append({**r, 'scored_text': cut, 'n_tokens': n, 'truncated': cut != r['text']})
        preds = E.predict([{'text': x['scored_text']} for x in chunk], model, tok, batch_size=a.batch)
        with open(dest, 'a') as f:
            for x, (off, p) in zip(chunk, preds):
                text = x['scored_text']; valid = np.array([bool(text[s:e].strip()) for s, e in off], dtype=bool)
                sents = [float(p[valid & (off[:, 1] > s) & (off[:, 0] < e)].mean()) for s, e in E.sentences(text) if (valid & (off[:, 1] > s) & (off[:, 0] < e)).any()]
                rec = {'input_id': x['input_id'], 'protocol_id': x['protocol_id'], 'status': 'ok' if valid.any() else 'error',
                       'mean_token': float(p[valid].mean()) if valid.any() else None, 'max_sentence': max(sents) if sents else None,
                       'max_token': float(p[valid].max()) if valid.any() else None, 'n_tokens': x['n_tokens'],
                       'tokens_scored': int(len(off)), 'truncated': x['truncated'], 'text_sha256': hashlib.sha256(x['text'].encode()).hexdigest()}
                f.write(json.dumps(rec) + '\n')
        print(json.dumps({'shard': a.shard, 'done': c + len(chunk), 'of': len(rows), 'seconds': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('tag'); p.add_argument('checkpoint'); p.add_argument('inputs'); p.add_argument('out_dir')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1); p.add_argument('--batch', type=int, default=16)
    main(p.parse_args())
