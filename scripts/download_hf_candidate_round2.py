"""Download the pinned public Hugging Face files for candidate round two."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT=Path('/mnt/f/pangram-at-home/data/candidate_span_sources/round2')
SOURCES={
    'travis_shortstory':('Travis-ML/ShortStory-SFT-jsonl',
        '8051abeeb58c7b94177d42b00676f07173a718f3',['data/train.jsonl']),
    'gradtex':('elisabeth-pl-pl/GRADTEX',
        '553d859da0255d75a39c385c208f7522a2007f53',['train.parquet']),
    'story_contrastive':('schonsense/human_ai_story_contrastive_v4',
        'f7700e7baa1a3796f3a1de30099c3e6709ef48fc',['dataset.jsonl']),
    'shp':('stanfordnlp/SHP',
        'e94b5f32602712d78ed494fe79105b1959396686',
        [f'{name}/train.json' for name in (
            'askculinary','askhistorians','askphilosophy','askscience','asksciencefiction')]),
    'dolly':('databricks/databricks-dolly-15k',
        'bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
        ['databricks-dolly-15k.jsonl']),
}


def main() -> None:
    manifest={}
    for name,(repository,revision,files) in SOURCES.items():
        saved=[]
        for filename in files:
            path=Path(hf_hub_download(repository,filename,repo_type='dataset',
                revision=revision,local_dir=ROOT/name))
            saved.append({'path':filename,'bytes':path.stat().st_size,
                          'sha256':sha256(path.read_bytes()).hexdigest()})
        manifest[name]={'repository':repository,'revision':revision,'files':saved}
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({name:{'repository':row['repository'],
                            'files':len(row['files'])} for name,row in manifest.items()},indent=2))


if __name__=='__main__':
    main()
