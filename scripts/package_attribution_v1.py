"""Write checksums for files sent to the private Vast attribution run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files={}
    for name in ('requirements-span.txt','scripts/bootstrap_attribution_vast_v1.sh',
                 'scripts/launch_attribution_vast_v1.sh',
                 'scripts/train_attribution_heads_v1.py'):
        files['pangram-at-home/'+name]=REPO/name
    for task in ('arena','authors'):
        for split in ('train','val','test'):
            name=f'data/attribution_heads_v1/{task}/{split}.jsonl'
            files['pangram-data/'+name]=ROOT/name
        name=f'data/attribution_heads_v1/{task}/manifest.json'
        files['pangram-data/'+name]=ROOT/name
    for path in (ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k/best_adapter').iterdir():
        if path.is_file():
            files['pangram-data/'+str(path.relative_to(ROOT))]=path
    manifest={'purpose':'private frozen-feature attribution probes',
              'files':{name:{'bytes':path.stat().st_size,'sha256':sha(path)}
                       for name,path in sorted(files.items())}}
    output=ROOT/'packages/attribution_heads_v1_upload_manifest.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(output,len(files),'files',sum(value['bytes'] for value in manifest['files'].values()),'bytes')


if __name__=='__main__':
    main()
