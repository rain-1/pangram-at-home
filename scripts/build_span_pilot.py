"""Known-provenance synthetic mixtures for a localization pilot, not a realism benchmark."""
import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
import pyarrow.parquet as pq


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(text, rng, target):
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if x.strip()]
    start = rng.randrange(len(sentences))
    selected = []
    for sentence in sentences[start:]:
        selected.append(sentence)
        if len(" ".join(selected).split()) >= target: break
    return " ".join(selected)


def build(rows, count, seed, split):
    rng = random.Random(seed)
    pools=defaultdict(list)
    for r in rows: pools[(r["source"],r["label"])].append(r)
    sources = sorted(s for s,l in pools if l == 0 and (s,1) in pools)
    weights=[len(pools[(s,0)]) for s in sources]
    out=[]
    for index in range(count):
        source=rng.choices(sources,weights=weights)[0]
        kind=["human","ai","mixed","mixed"][index%4]
        if kind == "mixed":
            first=rng.randrange(2)
            labels=[(first+i)%2 for i in range(rng.choice([2,3,4,6]))]
        else: labels=[int(kind == "ai")]
        text="";spans=[]; ids=[]; groups=[]
        for label in labels:
            row=rng.choice(pools[(source,label)])
            piece=excerpt(row["text"],rng,rng.choice([25,60,120,240]) if kind=="mixed" else 500)
            if text: text+="\n\n"
            start=len(text);text+=piece
            spans.append({"start":start,"end":len(text),"label":label})
            ids.append(row["text_id"]);groups.append(row["group_id"])
        out.append({"id":f"{split}_{index:05d}","text":text,"spans":spans,"kind":kind,
                    "source":source,"domain":row["domain"],"source_ids":ids,"source_groups":groups,
                    "construction":"same-source excerpt concatenation; topic consistency not guaranteed"})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/mnt/f/pangram-at-home"))
    p.add_argument("--train-docs",type=int,default=1200);p.add_argument("--val-docs",type=int,default=240)
    args=p.parse_args();base=args.root/"data/diverse_pyramid_v1";out=args.root/"data/span_pilot_v1"
    if out.exists(): raise SystemExit(f"Refusing to overwrite {out}")
    data={s:pq.read_table(base/f"{s}_full.parquet").to_pylist() for s in ("train","val")}
    for key in ("text_sha256","group_id"):
        assert not ({r[key] for r in data["train"]}&{r[key] for r in data["val"]}),key
    out.mkdir(parents=True)
    manifest={"labels":{"human":0,"ai_generated":1},"ai_assisted_supported":False,
              "role":"synthetic development pilot; same-source joins may expose topic/format artifacts",
              "test_data_used":False,"splits":{}}
    for s,n,seed in (("train",args.train_docs,42),("val",args.val_docs,142)):
        rows=build(data[s],n,seed,s);path=out/f"{s}.jsonl"
        path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
        manifest["splits"][s]={"documents":n,"kinds":dict(Counter(r["kind"] for r in rows)),
            "domains":dict(Counter(r["domain"] for r in rows)),"sha256":sha(path),
            "parent_sha256":sha(base/f"{s}_full.parquet"),"seed":seed}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(out)


if __name__=="__main__":main()
