"""Typography/formatting-cue audit of a prepared-v2 epoch file: AI vs human text overall, per dataset, and inside mixed rows.

A cue that separates the classes for rendering reasons (quote style, line breaks, whitespace) rather than writing is a shortcut the
detector can learn, as heterogeneous-ai-spans v1.3.0 showed. Rates are per 10k characters of labelled region text. The mixed-row
contrast counts rows holding both labels where a feature occurs in one label's text and is absent from the other's (only rows
where both sides have >= 200 chars). Usage: python format_audit_t21.py PREPARED_DIR [epoch-key]
"""
import gzip, json, re, sys, collections
FEATS = {'curly "': r'[“”]', "curly '": r'[‘’]', 'straight "': r'"', "straight '": r"'", 'em dash —': r'—', 'en dash –': r'–',
         '--': r'--', 'ellipsis …': r'…', '...': r'\.\.\.', 'nbsp/thin': r'[   ]', 'double space': r'(?<=\S)  (?=\S)',
         'single \\n': r'(?<!\n)\n(?!\n)', '\\n\\n': r'\n\n', 'CR': r'\r', 'trailing ws': r'[ \t]+\n', 'non-ASCII': r'[^\x00-\x7f]',
         'markdown': r'(?m)\*\*|^#{1,6} |^\s*[-*•] '}
P = {k: re.compile(v) for k, v in FEATS.items()}
d = sys.argv[1]; key = sys.argv[2] if len(sys.argv) > 2 else 'stage2-epoch0'
by = collections.defaultdict(collections.Counter); chars = collections.Counter(); contrast = collections.Counter(); nmixed = 0
for l in gzip.open(f'{d}/{key}.jsonl.gz', 'rt'):
    r = json.loads(l); t = r['text']
    if r.get('supervision') == 'document_only':
        continue
    side = collections.defaultdict(str)
    for g in r['regions']:
        if g['label'] in (0, 1):
            side[g['label']] += t[g['start']:g['end']] + '\x00'
    for lab, seg in side.items():
        for k in ((r['dataset'], lab), ('ALL', lab)):
            chars[k] += len(seg)
            for n, p in P.items():
                by[k][n] += len(p.findall(seg))
    if len(side.get(0, '')) >= 200 and len(side.get(1, '')) >= 200:
        nmixed += 1
        for n, p in P.items():
            h, a = bool(p.search(side[0])), bool(p.search(side[1]))
            if a and not h: contrast[(n, 'ai_only')] += 1
            if h and not a: contrast[(n, 'human_only')] += 1
rate = lambda k, n: 1e4 * by[k][n] / max(1, chars[k])
print(f'file {d}/{key}; labelled chars AI {chars[("ALL", 1)]:,}, human {chars[("ALL", 0)]:,}; mixed rows with both sides >=200 chars: {nmixed}')
print(f"\n{'feature':14s}{'AI':>8s}{'human':>8s}{'ratio':>8s}{'mixed: AI-only':>16s}{'human-only':>12s}")
for n in FEATS:
    a, h = rate(('ALL', 1), n), rate(('ALL', 0), n)
    print(f"{n:14s}{a:8.1f}{h:8.1f}{(a + .05) / (h + .05):8.2f}{contrast[(n, 'ai_only')]:16d}{contrast[(n, 'human_only')]:12d}")
dsets = sorted({k[0] for k in chars if k[0] != 'ALL'}, key=lambda s: -sum(chars[(s, l)] for l in (0, 1)))
show = ['curly "', "curly '", 'straight "', "straight '", 'em dash —', 'single \\n', '\\n\\n', 'double space', 'non-ASCII']
print(f"\nper dataset and label (per 10k chars)\n{'dataset:label':24s}{'kchars':>8s}" + ''.join(f'{n[:10]:>11s}' for n in show))
for s in dsets:
    for lab in (1, 0):
        k = (s, lab)
        if chars[k]:
            print(f"{s[:20] + ':' + ('AI' if lab else 'hu'):24s}{chars[k] / 1e3:8.0f}" + ''.join(f'{rate(k, n):11.1f}' for n in show))
