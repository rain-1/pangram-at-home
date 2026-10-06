"""Write mix-spec.json + leakage-drop-ids.json, then simulate setup_mix.py's selection on the locally windowed rows."""
import gzip, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent; W = H / 'work'
SEED = 20261006

HM = {'window_kinds': ['mixed', 'ai']}
spec = {
    '_notes': ('Per-epoch caps are stage-2 windows (500-token windows from setup_mix.py). Each document lands in exactly one stage-2 epoch '
               '(hash of pair key), so a cap above ~1/3 of a key\'s available windows is never reached. For hetero rows set pair_id = source_id '
               '(shared by a mixed document and its control). Base rows are unchanged; additions total ~2,850 windows/epoch (~12% of 24,000).'),
    'stage1': {'include': False, 'notes': 'Stage 1 unchanged (12,000 rows). Keeps the A/B difference to stage 2 and avoids doubling exposure of a small pool.'},
    'pair_controls_with_mixed': True,
    # heterogeneous-ai-spans v1.3.0, mixed documents: keep only windows that contain AI text (pure-human windows are covered by the controls)
    'hetero:mixed:jmlr_pre2015':        {'keep': True, 'max_windows_per_epoch': 600, 'dataset_name': 'hetero_mixed', 'filters': HM},
    'hetero:mixed:gutenberg_selected':  {'keep': True, 'max_windows_per_epoch': 650, 'dataset_name': 'hetero_mixed', 'filters': HM},
    'hetero:mixed:standardebooks':      {'keep': True, 'max_windows_per_epoch': 200, 'dataset_name': 'hetero_mixed', 'filters': HM},
    'hetero:mixed:wikitext2_raw':       {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': 'hetero_mixed', 'filters': HM},
    'hetero:mixed:hansard':             {'keep': True, 'max_windows_per_epoch': 2, 'dataset_name': 'hetero_mixed', 'filters': HM},
    'hetero:mixed:beigebook':           {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': 'hetero_mixed', 'filters': {}},
    'hetero:human_control:jmlr_pre2015':       {'keep': True, 'max_windows_per_epoch': 450, 'dataset_name': 'hetero_controls', 'filters': {}},
    'hetero:human_control:gutenberg_selected': {'keep': True, 'max_windows_per_epoch': 450, 'dataset_name': 'hetero_controls', 'filters': {}},
    'hetero:human_control:standardebooks':     {'keep': True, 'max_windows_per_epoch': 120, 'dataset_name': 'hetero_controls', 'filters': {}},
    'hetero:human_control:wikitext2_raw':      {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': 'hetero_controls', 'filters': {}},
    'hetero:human_control:hansard':            {'keep': True, 'max_windows_per_epoch': 2, 'dataset_name': 'hetero_controls', 'filters': {}},
    'hetero:human_control:beigebook':          {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': 'hetero_controls', 'filters': {}},
    # rain1 v14
    'v14:LLMTrace_detection':        {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': 'v14_mixed', 'filters': {}},
    'v14:elisabeth-pl-pl/GRADTEX':   {'keep': True, 'max_windows_per_epoch': 200, 'dataset_name': 'v14_mixed', 'filters': {}},
    'v14:science_v9:noaa_fisheries':         {'keep': True, 'max_windows_per_epoch': 140, 'dataset_name': 'v14_human', 'filters': {'window_kinds': ['human']}},
    'v14:science_v9:nasa_earth_observatory': {'keep': True, 'max_windows_per_epoch': 55, 'dataset_name': 'v14_human', 'filters': {'window_kinds': ['human']}},
}
DROP = {
    'v14:paper:acl_anthology': 'v14_mixed', 'v14:paper:pmc_oa': 'v14_mixed', 'v14:asap2_student_essay': 'v14_mixed', 'v14:persuade_2.0': 'v14_human',
    'v14:writers.stackexchange.com': 'v14_human', 'v14:databricks/databricks-dolly-15k': 'v14_human', 'v14:craphound.com': 'v14_human',
    **{f'v14:commonpile:news-{p}': 'v14_human' for p in ['360info', 'altnews', 'factly', 'milwaukeenns', 'newcanadianmedia']},
    **{f'v14:mage:{p}': 'v14_mixed' for p in ['cmv', 'eli5', 'roct', 'sci', 'squad', 'tldr', 'wp', 'xsum', 'yelp']},
}
for k, ds in DROP.items():
    spec[k] = {'keep': False, 'max_windows_per_epoch': 0, 'dataset_name': ds, 'filters': {}}


# ---- leakage drop list: strict rule over the loose candidates found by overlap.py
det = json.load(open(W / 'drops-detail.json'))
def strict(x):
    return (x['builder_sent'] > 0 or x['suite_sent'] > 0 or x['hetero_eval_sent'] > 0 or x['builder_sh'] >= 8 or x['suite_sh'] >= 8
            or x['hetero_eval_sh'] >= 8 or (x['suite_cov'] >= 0.05 and x['suite_sh'] >= 3))
rows = {}
for n in ['v14-train', 'hetero-train']:
    for l in gzip.open(W / f'{n}.jsonl.gz', 'rt'):
        r = json.loads(l); rows[r['id']] = r
drops = defaultdict(set); weak = Counter()
for k, xs in det.items():
    for x in xs:
        if strict(x):
            drops[k].add(x['id'])
        else:
            weak[k] += 1
# pair closure for hetero (drop the partner of any dropped mixed/control document)
pair_ids = defaultdict(list)
for r in rows.values():
    if r['source_key'].startswith('hetero'):
        pair_ids[r['pair_id']].append(r)
for k in list(drops):
    if k.startswith('hetero'):
        for i in list(drops[k]):
            for p in pair_ids[rows[i]['pair_id']]:
                drops[p['source_key']].add(p['id'])
for k, v in json.load(open(W / 'v3-drops.json')).items():
    drops[k] |= set(v)
for k in list(drops):  # pair closure again after adding benchmark-v3 hits
    if k.startswith('hetero'):
        for i in list(drops[k]):
            for p in pair_ids[rows[i]['pair_id']]:
                drops[p['source_key']].add(p['id'])
drops = {k: sorted(v) for k, v in sorted(drops.items())}
(H / 'leakage-drop-ids.json').write_text(json.dumps(drops, indent=1))
(H / 'mix-spec.json').write_text(json.dumps(spec, indent=1))
print('drops', {k: len(v) for k, v in drops.items()}, 'weak(kept)', dict(weak))

# ---- simulate setup_mix.py selection
keys = {k: v for k, v in spec.items() if ':' in k and isinstance(v, dict)}
docs = defaultdict(list)
for n in ['v14-train', 'hetero-train']:
    for l in gzip.open(W / f'windows-{n}.jsonl.gz', 'rt'):
        w = json.loads(l); k = w['source_key']; s = keys.get(k)
        if not s or not s['keep'] or w['doc_id'] in set(drops.get(k, ())):
            continue
        f = s.get('filters', {})
        if f.get('window_kinds') and w['window_kind'] not in f['window_kinds']:
            continue
        docs[k].append(w)
avail = {k: len(v) for k, v in docs.items()}
epoch_of = lambda pair: int(hashlib.sha256(f'{SEED}:{pair}'.encode()).hexdigest(), 16) % 3
chosen = {e: [] for e in range(3)}; rng = random.Random(SEED); picked = {e: set() for e in range(3)}
for k in sorted(keys, key=lambda k: (not k.startswith('hetero:mixed'), k)):
    s = keys[k]; cap = s['max_windows_per_epoch']
    if not s['keep'] or not cap:
        continue
    by_doc = defaultdict(list)
    for w in docs.get(k, []):
        by_doc[w['doc_id']].append(w)
    for e in range(3):
        pool = [d for d, ws in by_doc.items() if epoch_of(ws[0]['pair']) == e]; rng.shuffle(pool)
        if k.startswith('hetero:human_control'):
            pool.sort(key=lambda d: by_doc[d][0]['pair'] not in picked[e])
        n = 0
        for d in pool:
            if n >= cap:
                break
            ws = by_doc[d][:cap - n]; n += len(ws); picked[e].add(ws[0]['pair'])
            chosen[e] += [(k, s['dataset_name'], w) for w in ws]
gen = {r['id']: r['generator'] for r in rows.values()}
sim = {'available_windows_after_filters_and_drops': avail, 'per_epoch': {}}
mixed_pairs_all = defaultdict(set)
for e in range(3):
    c = chosen[e]
    by_key = Counter(k for k, _, _ in c); by_ds = Counter(ds for _, ds, _ in c); kinds = Counter((ds, w['window_kind']) for _, ds, w in c)
    gens = Counter(gen[w['doc_id']] for k, _, w in c if k.startswith('hetero:mixed'))
    mp = {w['pair'] for k, _, w in c if k.startswith('hetero:mixed')}; cp = {w['pair'] for k, _, w in c if k.startswith('hetero:human_control')}
    ai_chars = sum(g['end'] - g['start'] for _, _, w in c for g in w['regions'] if g['label'] == 1)
    lab_chars = sum(g['end'] - g['start'] for _, _, w in c for g in w['regions'] if g['label'] in (0, 1))
    short = sum(1 for _, _, w in c if w['window_kind'] == 'mixed' and any(g['label'] == 1 and not g['clipped'] and g['end'] - g['start'] < 200 for g in w['regions']))
    sim['per_epoch'][e] = {'total': len(c), 'by_key': dict(by_key), 'by_dataset': dict(by_ds), 'kinds': {f'{a}:{b}': v for (a, b), v in kinds.items()},
                           'hetero_mixed_generators': dict(gens), 'hetero_mixed_pairs': len(mp), 'controls_pairs': len(cp),
                           'controls_paired_with_selected_mixed': len(cp & mp), 'controls_unpaired': len(cp - mp),
                           'mixed_pairs_without_control': len(mp - cp), 'ai_char_fraction': round(ai_chars / lab_chars, 3),
                           'mixed_windows_with_unclipped_ai_span_lt200': short}
(W / 'simulation.json').write_text(json.dumps(sim, indent=1))
print(json.dumps(sim, indent=1))
