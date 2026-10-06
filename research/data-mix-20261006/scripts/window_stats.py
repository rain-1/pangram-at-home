"""Window additions exactly like setup_mix.py (500-token consecutive windows, native Qwen3.5/3.6 tokenizer) and compute span stats.
Also computes the same stats on woog's base stage-2 epochs. Writes work/windows-<name>.jsonl.gz and work/stats.json."""
import gzip, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
from tokenizers import Tokenizer

H = Path(__file__).resolve().parent; W = H / 'work'; RM = H / 'remote'
WIDTH, LIMIT = 500, 510
tok = Tokenizer.from_file(str(RM / 'tokenizer.json'))


def windows(row, off):
    text = row['text']; out = []
    for i in range(0, len(off), WIDTH):
        a, b = off[i][0], off[min(i + WIDTH, len(off)) - 1][1]
        regs = [{'start': max(g['start'], a) - a, 'end': min(g['end'], b) - a, 'label': g['label'],
                 'clipped': g['start'] < a or g['end'] > b}
                for g in row['regions'] if min(g['end'], b) > max(g['start'], a)]
        wt = text[a:b]
        if not any(g['label'] in (0, 1) and wt[g['start']:g['end']].strip() for g in regs):
            continue
        ai = [g for g in regs if g['label'] == 1]
        kind = 'mixed' if ai and any(g['label'] == 0 for g in regs) else 'ai' if ai else 'human'
        out.append({'text': wt, 'regions': regs, 'char_start': a, 'window_kind': kind, 'ntok': min(i + WIDTH, len(off)) - i})
    return out


def boundary_type(text, regs):
    """Classify each 0<->1 transition (skipping -100 separators) as paragraph / sentence / intra-sentence."""
    out = []
    lab = [(g['start'], g['end'], g['label']) for g in sorted(regs, key=lambda g: g['start'])]
    prev = None; sep = ''
    for s, e, l in lab:
        if l not in (0, 1):
            sep += text[s:e]; continue
        if prev is not None and prev[2] != l:
            p = s
            left = text[max(0, prev[1] - 4):prev[1]] + sep; right = text[s:s + 4]
            if '\n' in left or '\n' in right or '\n' in sep:
                out.append('paragraph')
            else:
                lt = text[:prev[1]].rstrip()
                out.append('sentence' if lt and lt[-1] in '.!?"\')”’:;' else 'intra_sentence')
        prev = (s, e, l); sep = ''
    return out


def win_stats(ws):
    st = {'windows': len(ws), 'kinds': Counter(), 'ai_span_chars': [], 'ai_span_chars_unclipped': [], 'spans_per_mixed_window': [],
          'ai_frac': [], 'boundaries': Counter(), 'short_unclipped_lt200_windows': 0, 'ntok': [], 'kindlist': []}
    for w in ws:
        st['kinds'][w['window_kind']] += 1; st['ntok'].append(w['ntok']); st['kindlist'].append(w['window_kind'])
        ai = [g for g in w['regions'] if g['label'] == 1]
        lab = sum(g['end'] - g['start'] for g in w['regions'] if g['label'] in (0, 1))
        st['ai_frac'].append(sum(g['end'] - g['start'] for g in ai) / lab if lab else 0)
        for g in ai:
            st['ai_span_chars'].append(g['end'] - g['start'])
            if not g.get('clipped'):
                st['ai_span_chars_unclipped'].append(g['end'] - g['start'])
        if w['window_kind'] == 'mixed':
            st['spans_per_mixed_window'].append(len(ai))
            st['boundaries'].update(boundary_type(w['text'], w['regions']))
        if any(not g.get('clipped') and g['end'] - g['start'] < 200 for g in ai) and w['window_kind'] == 'mixed':
            st['short_unclipped_lt200_windows'] += 1
    return st


def summarize(st):
    def bins(xs):
        n = len(xs) or 1
        return {'n': len(xs), 'median': statistics.median(xs) if xs else None,
                'lt200': round(100 * sum(x < 200 for x in xs) / n, 1), '200_600': round(100 * sum(200 <= x <= 600 for x in xs) / n, 1),
                'gt600': round(100 * sum(x > 600 for x in xs) / n, 1)}
    n = st['windows'] or 1; mixed = st['kinds']['mixed']
    return {'windows': st['windows'], 'pct_human': round(100 * st['kinds']['human'] / n, 1), 'pct_ai': round(100 * st['kinds']['ai'] / n, 1),
            'pct_mixed': round(100 * mixed / n, 1), 'ai_spans_all': bins(st['ai_span_chars']), 'ai_spans_unclipped': bins(st['ai_span_chars_unclipped']),
            'mean_spans_per_mixed_window': round(statistics.mean(st['spans_per_mixed_window']), 2) if st['spans_per_mixed_window'] else None,
            'mean_ai_frac_all': round(statistics.mean(st['ai_frac']), 3) if st['ai_frac'] else None,
            'mean_ai_frac_mixed': round(statistics.mean([f for f, k in zip(st['ai_frac'], st['kindlist']) if k == 'mixed']), 3) if mixed else None,
            'boundaries_mixed_windows': dict(st['boundaries']), 'mixed_windows_with_unclipped_span_lt200': st['short_unclipped_lt200_windows'],
            'median_tokens': statistics.median(st['ntok']) if st['ntok'] else None}


def run_additions(name):
    rows = [json.loads(l) for l in gzip.open(W / f'{name}.jsonl.gz', 'rt')]
    encs = tok.encode_batch([r['text'] for r in rows], add_special_tokens=False)
    by_key = defaultdict(list); doc = defaultdict(lambda: {'docs': 0, 'chars': [], 'tokens': [], 'doc_ai_spans': [], 'doc_boundaries': Counter(),
                                                           'doc_ai_frac': [], 'gens': Counter(), 'construction': Counter(), 'nc': 0})
    with gzip.open(W / f'windows-{name}.jsonl.gz', 'wt') as f:
        for r, e in zip(rows, encs):
            k = r['source_key']; d = doc[k]; d['docs'] += 1; d['chars'].append(len(r['text'])); d['tokens'].append(len(e.offsets))
            ai = [g for g in r['regions'] if g['label'] == 1]
            d['doc_ai_spans'] += [g['end'] - g['start'] for g in ai]
            lab = sum(g['end'] - g['start'] for g in r['regions'] if g['label'] in (0, 1))
            d['doc_ai_frac'].append(sum(g['end'] - g['start'] for g in ai) / lab if lab else 0)
            d['doc_boundaries'].update(boundary_type(r['text'], r['regions']))
            d['gens'][r['generator']] += 1; d['construction'][r.get('construction') or r.get('cohort')] += 1; d['nc'] += bool(r['noncommercial'])
            ws = windows(r, e.offsets)
            for j, w in enumerate(ws):
                w.update(doc_id=r['id'], source_key=k, j=j, pair=r.get('pair_id') or r['group_id'])
                f.write(json.dumps(w) + '\n')
            by_key[k] += ws
    out = {}
    for k in sorted(doc):
        d = doc[k]
        out[k] = {'docs': d['docs'], 'median_doc_chars': statistics.median(d['chars']), 'median_doc_tokens': statistics.median(d['tokens']),
                  'chars_per_token': round(sum(d['chars']) / max(1, sum(d['tokens'])), 2),
                  'doc_mean_ai_frac': round(statistics.mean(d['doc_ai_frac']), 3), 'doc_ai_spans': summarize({'windows': 0, 'kinds': Counter(), 'ai_span_chars': d['doc_ai_spans'],
                  'ai_span_chars_unclipped': [], 'spans_per_mixed_window': [], 'ai_frac': [], 'boundaries': Counter(), 'short_unclipped_lt200_windows': 0, 'ntok': [], 'kindlist': []})['ai_spans_all'],
                  'doc_boundaries': dict(d['doc_boundaries']), 'generators_distinct': len([g for g in d['gens'] if g != 'human']),
                  'top_generators': d['gens'].most_common(6), 'construction': dict(d['construction']), 'noncommercial_docs': d['nc'],
                  'windowed': summarize(win_stats(by_key[k]))}
    return out


def run_base():
    out = {}
    per = defaultdict(list)
    for e in range(3):
        for l in gzip.open(RM / f'stage2-epoch{e}.jsonl.gz', 'rt'):
            r = json.loads(l)
            if r.get('supervision') == 'document_only':
                per[r['dataset']].append({'text': r['text'], 'regions': [], 'window_kind': 'doc_only:' + str(r.get('document_label')), 'ntok': 0}); continue
            regs = r['regions']
            ai = [g for g in regs if g['label'] == 1]
            kind = 'mixed' if ai and any(g['label'] == 0 for g in regs) else 'ai' if ai else 'human'
            # base windows are crops of longer rows: mark AI span clipped if it touches the window edge and the row is a crop
            for g in regs:
                g['clipped'] = (g['start'] == 0 and r.get('source_start', 0) > 0) or (g['end'] == len(r['text']))
            per[r['dataset']].append({'text': r['text'], 'regions': regs, 'window_kind': kind, 'ntok': 0})
    allw = []
    for k, ws in per.items():
        out[k] = summarize(win_stats(ws)); allw += ws
        if k == 'gradtex':
            out[k]['doc_labels'] = dict(Counter(w['window_kind'] for w in ws))
    out['ALL'] = summarize(win_stats(allw))
    return out


if __name__ == '__main__':
    res = {'base_stage2_3epochs': run_base()}
    for n in ['v14-train', 'hetero-train', 'hetero-validation', 'hetero-test']:
        res[n] = run_additions(n); print(n, 'done', file=sys.stderr)
    (W / 'stats.json').write_text(json.dumps(res, indent=1, default=str))
