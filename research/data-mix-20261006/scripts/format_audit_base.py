"""Formatting-cue audit of a prepared-v2 epoch file by dataset and label (run from the sweep root on the Space)."""
import gzip, json, re, collections
feats = {'CR': r'\r', 'single \\n': r'(?<!\n)\n(?!\n)', '\\n\\n': r'\n\n', 'curly quotes': r'[“”‘’]', 'straight quotes': r'["\']',
         'em/en dash': r'[—–]', '--': r'--', 'ellipsis …': r'…', 'non-ASCII': r'[^\x00-\x7f]', 'markdown **/#': r'(?m)\*\*|^#{1,6} ', 'bullets': r'(?m)^\s*[-*•] '}
by = collections.defaultdict(collections.Counter); chars = collections.Counter()
for l in gzip.open('runs/qwen36-35b-a3b/prepared-v2/stage2-epoch0.jsonl.gz', 'rt'):
    r = json.loads(l); t = r['text']
    if r.get('supervision') == 'document_only':
        segs = [(t, 'doc' + str(r.get('document_label')))]
    else:
        segs = [(t[g['start']:g['end']], {0: 'human', 1: 'ai'}.get(g['label'])) for g in r['regions'] if g['label'] in (0, 1)]
    for seg, lab in segs:
        k = (r['dataset'], lab); chars[k] += len(seg)
        for n, p in feats.items():
            by[k][n] += len(re.findall(p, seg))
keys = sorted(chars)
print(f"{'per 10k chars':16s}" + ''.join(f"{(d[:8]+':'+l)[:16]:>17s}" for d, l in keys))
for n in feats:
    print(f"{n:16s}" + ''.join(f"{1e4*by[k][n]/max(1,chars[k]):17.1f}" for k in keys))
