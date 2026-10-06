"""Build a prepared-v2 variant = woog's MoE training data + a curated selection of rain1's span data (no GPU).

Inputs (in the sweep root): additions-v1.jsonl.gz (rain1 v14 span rows + heterogeneous-ai-spans v1.3.0 train split, one schema:
{id, source_key, group_id, pair_id, text, regions[{start,end,label}], generator, noncommercial}), mix-spec.json (per source_key:
keep, max_windows_per_epoch, dataset_name, filters) and leakage-drop-ids.json, both from the mix analysis.

Each kept row is cut into consecutive <=500-token windows with the model's own tokenizer (re-checked against the trainer's 510
limit) and regions clipped to the window. Rows whose sentences (>=10 words) also occur in the sweep evaluation rows or the
selection/calibration windows are dropped, as in splice-sweep-20261006/setup_splice.py. A document and, when
pair_controls_with_mixed is set, its matched human control land in the same stage-2 epoch; per-epoch caps are filled by whole
documents in seeded random order. Stage 1 and the base rows are unchanged; additions are new `dataset` names.
"""
import argparse, gzip, hashlib, json, random, re, shutil, time
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent; SEED = 20261006; WIDTH, LIMIT = 500, 510
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)


def say(log, **k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush(); print(json.dumps(k), flush=True)


def sents(t):
    out = set()
    for m in SENT.finditer(t):
        w = re.findall(r'[a-z0-9]+', m.group().lower())
        if len(w) >= 10:
            out.add(' '.join(w))
    return out


def windows(row, tok):
    """Consecutive token windows; regions clipped and rebased; windows without a 0/1 label are skipped."""
    text = row['text']; off = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    out = []
    for i in range(0, len(off), WIDTH):
        a, b = off[i][0], off[min(i + WIDTH, len(off)) - 1][1]
        regs = [{'start': max(g['start'], a) - a, 'end': min(g['end'], b) - a, 'label': g['label']} for g in row['regions'] if min(g['end'], b) > max(g['start'], a)]
        wt = text[a:b]
        if not any(g['label'] in (0, 1) and wt[g['start']:g['end']].strip() for g in regs):
            continue
        ai = [g for g in regs if g['label'] == 1]
        out.append({'text': wt, 'regions': regs, 'char_start': a,
                    'target_start': min(g['start'] for g in ai) if ai else 0, 'target_end': max(g['end'] for g in ai) if ai else len(wt),
                    'window_kind': 'mixed' if ai and any(g['label'] == 0 for g in regs) else 'ai' if ai else 'human',
                    'ai_span_chars': [g['end'] - g['start'] for g in ai]})
    return out


def keep_window(w, f):
    if f.get('window_kinds') and w['window_kind'] not in f['window_kinds']:
        return False
    if w['ai_span_chars'] and f.get('min_ai_span_chars') and min(w['ai_span_chars']) < f['min_ai_span_chars']:
        return False
    if w['ai_span_chars'] and f.get('max_ai_span_chars') and max(w['ai_span_chars']) > f['max_ai_span_chars']:
        return False
    return True


def main(a):
    from transformers import AutoTokenizer
    log = open(R / f'setup-mix-{a.name}.log', 'a'); say(log, event='start', base=a.base, name=a.name)
    spec = json.loads((R / a.spec).read_text()); drops = json.loads((R / a.drops).read_text()) if (R / a.drops).exists() else {}
    keys = {k: v for k, v in spec.items() if ':' in k and isinstance(v, dict)}
    tok = AutoTokenizer.from_pretrained(R / 'assets' / a.base, local_files_only=True)
    base = R / 'runs' / a.base / 'prepared-v2'; man = json.loads((base / 'manifest.json').read_text())
    held = set()
    for r in (json.loads(l) for l in gzip.open(R / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')):
        held |= sents(r['text'])
    for name in ['selection-windows', 'calibration-windows']:
        for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
            held |= sents(json.loads(l)['text'])
    rows = [json.loads(l) for l in gzip.open(R / 'additions-v1.jsonl.gz', 'rt')]
    drops = {k: set(v) for k, v in drops.items()}
    stats = Counter(); docs = defaultdict(list)
    for r in rows:
        s = keys.get(r['source_key'])
        if not s or not s.get('keep'):
            stats['dropped_by_spec:' + r['source_key']] += 1; continue
        if r['id'] in drops.get(r['source_key'], ()):
            stats['dropped_leakage_list'] += 1; continue
        f = s.get('filters', {})
        if any(g in (r['generator'] or '') for g in f.get('generators_exclude', [])):
            stats['dropped_generator'] += 1; continue
        if sents(r['text']) & held:
            stats['dropped_shared_sentence'] += 1; continue
        ws = [w for w in windows(r, tok) if keep_window(w, f)]
        for j, w in enumerate(ws):
            n = len(tok(w['text'], add_special_tokens=False)['input_ids'])
            if not 0 < n <= LIMIT:
                stats['dropped_window_over_limit'] += 1; continue
            docs[r['source_key']].append({**w, 'doc_id': r['id'], 'pair': r.get('pair_id') or r['group_id'], 'group_id': r['group_id'], 'j': j})
    say(log, event='windowed', stats=dict(stats), windows={k: len(v) for k, v in docs.items()})

    # Epoch of a document = hash of its pair key, so a mixed document and its matched control share an epoch.
    epoch_of = lambda pair: int(hashlib.sha256(f'{SEED}:{pair}'.encode()).hexdigest(), 16) % 3
    chosen = {e: [] for e in range(3)}; rng = random.Random(SEED); picked_pairs = {e: set() for e in range(3)}
    order = sorted(keys, key=lambda k: (not k.startswith('hetero:mixed'), k))  # mixed first, so controls can follow their pairs
    for k in order:
        s = keys[k]; cap = s.get('max_windows_per_epoch', 0)
        if not s.get('keep') or not cap:
            continue
        by_doc = defaultdict(list)
        for w in docs.get(k, []):
            by_doc[w['doc_id']].append(w)
        for e in range(3):
            pool = [d for d, ws in by_doc.items() if epoch_of(ws[0]['pair']) == e]; rng.shuffle(pool)
            if spec.get('pair_controls_with_mixed') and k.startswith('hetero:human_control'):
                pool.sort(key=lambda d: by_doc[d][0]['pair'] not in picked_pairs[e])  # paired controls first
            n = 0
            for d in pool:
                if n >= cap:
                    break
                ws = by_doc[d][:cap - n]; n += len(ws); picked_pairs[e].add(ws[0]['pair'])
                chosen[e] += [(k, s['dataset_name'], w) for w in ws]
    out = R / 'runs' / a.name / 'prepared-v2'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    shutil.copytree(base, out); m2 = json.loads(json.dumps(man)); m2['additions'] = {'spec_sha256': hashlib.sha256((R / a.spec).read_bytes()).hexdigest(),
                                                                                      'additions_sha256': hashlib.sha256((R / 'additions-v1.jsonl.gz').read_bytes()).hexdigest(), 'per_epoch': {}}
    for e in range(3):
        key = f'stage2-epoch{e}'; rows_e = [json.loads(l) for l in gzip.open(base / f'{key}.jsonl.gz', 'rt')]
        for i, (k, ds, w) in enumerate(chosen[e]):
            rows_e.append({'id': f"{w['doc_id']}/w{w['j']}", 'paper_id': w['group_id'], 'kind': 'rain1_' + w['window_kind'], 'text': w['text'], 'regions': w['regions'],
                           'source_start': w['char_start'], 'source_end': w['char_start'] + len(w['text']), 'target_start': w['target_start'], 'target_end': w['target_end'],
                           'dataset': ds, 'group': w['group_id'], 'draw_id': f'{key}-rain1-{i}', 'source_key': k})
        b = ''.join(json.dumps(r) + '\n' for r in rows_e).encode(); (out / f'{key}.jsonl.gz').write_bytes(gzip.compress(b))
        m2['files'][key] = {**m2['files'][key], 'rows': len(rows_e), 'sha256': hashlib.sha256(b).hexdigest()}
        m2['counts'][key] = dict(Counter(r['dataset'] for r in rows_e))
        m2['additions']['per_epoch'][key] = dict(Counter(k for k, _, _ in chosen[e]))
    (out / 'manifest.json').write_text(json.dumps(m2, indent=1))
    models = json.loads((R / 'models.json').read_text()); spec_m = next(m for m in models if m['name'] == a.base)
    (R / 'assets' / a.name).symlink_to(R / 'assets' / a.base)
    (R / 'models.json').write_text(json.dumps([m for m in models if m['name'] != a.name] + [{**spec_m, 'name': a.name}], indent=1))
    say(log, event='done', counts=m2['counts']['stage2-epoch0'], manifest_sha256=hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', default='qwen36-35b-a3b'); p.add_argument('--name', default='qwen36-35b-a3b-mix')
    p.add_argument('--spec', default='mix-spec.json'); p.add_argument('--drops', default='leakage-drop-ids.json'); main(p.parse_args())
