"""Leakage and redundancy: normalized >=10-word sentence overlap (same rule as setup_mix.py) plus 12-word shingle near-dup check."""
import gzip, json, re, sys, zlib
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent; W = H / 'work'; RM = H / 'remote'
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
K = 12


def sents(t):
    out = set()
    for m in SENT.finditer(t):
        w = re.findall(r'[a-z0-9]+', m.group().lower())
        if len(w) >= 10:
            out.add(' '.join(w))
    return out


def shingles(t):
    w = re.findall(r'[a-z0-9]+', t.lower())
    return {hash(' '.join(w[i:i + K])) for i in range(0, max(0, len(w) - K + 1))}


def rows(path):
    for l in gzip.open(path, 'rt'):
        yield json.loads(l)


def texts_of(r):
    out = [r.get('text') or '']
    for k in ('original_target', 'generated_target', 'source_text'):
        if r.get(k):
            out.append(r[k])
    return out


protected = {
    'builder': ['sweep-eval-rows', 'selection-windows', 'calibration-windows'],
    'suite': ['eval-workflow', 'eval-comparison'],
}
P_sent = {}; P_sh = {}
for name, files in protected.items():
    s = set(); sh = set()
    for f in files:
        for r in rows(RM / f'{f}.jsonl.gz'):
            for t in texts_of(r):
                s |= sents(t); sh |= shingles(t)
    P_sent[name] = s; P_sh[name] = sh; print(name, len(s), len(sh), file=sys.stderr)
# hetero val/test (new held-out eval)
s = set(); sh = set(); hid = set(); htitles = set()
for sp in ['validation', 'test']:
    for r in rows(W / f'hetero-{sp}.jsonl.gz'):
        for t in texts_of(r):
            s |= sents(t); sh |= shingles(t)
        htitles.add((r['source_title'] or '').strip().lower())
P_sent['hetero_eval'] = s; P_sh['hetero_eval'] = sh

# base training sentences by dataset
base_sent = defaultdict(set)
for f in ['stage1-epoch0', 'stage2-epoch0', 'stage2-epoch1', 'stage2-epoch2']:
    for r in rows(RM / f'{f}.jsonl.gz'):
        base_sent[r['dataset']] |= sents(r['text'])
print({k: len(v) for k, v in base_sent.items()}, file=sys.stderr)

res = {}; drops = defaultdict(list); detail = defaultdict(list)
for name in ['v14-train', 'hetero-train']:
    for r in rows(W / f'{name}.jsonl.gz'):
        k = r['source_key']; st = res.setdefault(k, Counter()); st['docs'] += 1
        ss = sents(r['text']); sh = shingles(r['text'])
        if r.get('source_text'):
            ss |= sents(r['source_text']); sh |= shingles(r['source_text'])
        st['docs_with_10w_sentences'] += bool(ss)
        hit = {}
        for p in P_sent:
            ns = len(ss & P_sent[p]); nsh = len(sh & P_sh[p]); cov = nsh / max(1, len(sh))
            hit[p] = (ns, nsh, cov)
            if ns:
                st[f'sent_{p}'] += 1
            if nsh >= 3 or cov >= 0.05:
                st[f'near_{p}'] += 1
        for b, bs in base_sent.items():
            n = len(ss & bs)
            if n:
                st[f'base_{b}'] += 1
                if ss and n / len(ss) >= 0.5:
                    st[f'base_{b}_majority'] += 1
        builder_drops = hit['builder'][0] > 0
        st['builder_will_drop'] += builder_drops
        extra = (not builder_drops) and (hit['suite'][0] > 0 or hit['suite'][1] >= 3 or hit['suite'][2] >= 0.05
                                         or hit['builder'][1] >= 3 or hit['hetero_eval'][0] > 0 or hit['hetero_eval'][1] >= 3)
        if builder_drops or extra:
            drops[k].append(r['id'])
            detail[k].append({'id': r['id'], 'builder_sent': hit['builder'][0], 'builder_sh': hit['builder'][1], 'suite_sent': hit['suite'][0],
                              'suite_sh': hit['suite'][1], 'suite_cov': round(hit['suite'][2], 3), 'hetero_eval_sent': hit['hetero_eval'][0],
                              'hetero_eval_sh': hit['hetero_eval'][1]})
        st['drop_total'] += builder_drops or extra; st['drop_extra_beyond_builder'] += extra
        if r['source_key'].startswith('hetero') and (r.get('source_title') or '').strip().lower() in htitles:
            st['hetero_title_shared_with_eval_split'] += 1

# hetero val/test vs base training (eval fairness, affects both arms equally)
ev = Counter()
for sp in ['validation', 'test']:
    for r in rows(W / f'hetero-{sp}.jsonl.gz'):
        ss = sents(r['source_text'] or r['text'])
        ev['docs'] += 1
        for b, bs in base_sent.items():
            if ss & bs:
                ev[f'{r["source_key"].split(":")[2]}:base_{b}'] += 1
        # v14 train sentences
json.dump({'per_source': res, 'hetero_eval_vs_base': ev}, open(W / 'overlap.json', 'w'), indent=1)
json.dump(drops, open(W / 'drops-raw.json', 'w'), indent=1)
json.dump(detail, open(W / 'drops-detail.json', 'w'), indent=1)
for k, v in res.items():
    print(k, dict(v))
print(dict(ev))
