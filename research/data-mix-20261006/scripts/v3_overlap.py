import gzip, json, re, hashlib
from collections import Counter, defaultdict
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
def sents(t):
    out=set()
    for m in SENT.finditer(t):
        w=re.findall(r'[a-z0-9]+',m.group().lower())
        if len(w)>=10: out.add(' '.join(w))
    return out
def sh(t):
    w=re.findall(r'[a-z0-9]+',t.lower()); return {hash(' '.join(w[i:i+12])) for i in range(max(0,len(w)-11))}
v3=[json.loads(l) for l in gzip.open('work/v3.jsonl.gz','rt')]
S=defaultdict(set); SH={}; keys={}; shas={}
for r in v3:
    for s in sents(r['text']): S[s].add(r['dataset_key'])
    for x in sh(r['text']): SH.setdefault(x, r['dataset_key'])
    for k in json.loads(r['leakage_keys'] or '[]'): keys[k.split(':',1)[1] if ':' in k else k]=r['dataset_key']
    keys[r['source_group_id'].split(':',1)[-1]]=r['dataset_key']
    shas[r['text_sha256']]=r['dataset_key']
v3titles=Counter()
res=defaultdict(Counter); drops=defaultdict(set); why=defaultdict(Counter)
for n in ['v14-train','hetero-train']:
    for l in gzip.open(f'work/{n}.jsonl.gz','rt'):
        r=json.loads(l); k=r['source_key']; res[k]['docs']+=1
        t=r['text']; ss=sents(t)|(sents(r['source_text']) if r.get('source_text') else set())
        hs=sh(t)|(sh(r['source_text']) if r.get('source_text') else set())
        sm=Counter(d for s in ss for d in S.get(s,()))
        hm=Counter(SH[x] for x in hs if x in SH)
        ids=set()
        for cand in [r['group_id'], r.get('pair_id') or '', r['id']]:
            c=cand.split(':',1)[-1] if cand.startswith('llmtrace:') else cand
            for kk in (cand, c, 'jmlr:'+cand.split(':',1)[-1] if cand.startswith('jmlr') else cand):
                if kk in keys: ids.add(keys[kk])
        ex=hashlib.sha256(t.encode()).hexdigest() in shas
        hit = bool(sm) or sum(hm.values())>=8 or bool(ids) or ex
        if hit:
            drops[k].add(r['id'])
            for d in set(sm)|{d for d,v in hm.items() if v>=8}|ids: why[k][d]+=1
        res[k]['hit']+=hit; res[k]['sent']+=bool(sm); res[k]['id']+=bool(ids); res[k]['exact']+=ex
json.dump({k:sorted(v) for k,v in drops.items()}, open('work/v3-drops.json','w'), indent=1)
json.dump({'per_source':res,'by_v3_dataset':why}, open('work/v3-overlap.json','w'), indent=1)
for k in sorted(res):
    if res[k]['hit']: print(k, dict(res[k]), dict(why[k]))
print('clean:', [k for k in sorted(res) if not res[k]['hit']])
