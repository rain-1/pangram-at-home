"""Create a checksumed, secret-free Vast payload for the v14 run."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile


REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')
DATA=ROOT/'data'


def add(files: list[tuple[Path,str]], path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    files.append((path,name))


def main() -> None:
    files=[]
    for name in ('requirements-span.txt','scripts/train_token_lora.py',
                 'scripts/train_segment_lora.py','scripts/span_data.py',
                 'scripts/span_metrics.py','scripts/evaluate_span_pilot.py',
                 'scripts/run_span_hardneg_v14.py',
                 'scripts/bootstrap_span_hardneg_v14.sh',
                 'scripts/launch_span_hardneg_v14.sh'):
        add(files,REPO/name,'pangram-at-home/'+name)
    datasets={
        'span_hardneg_v14':('train.jsonl','val.jsonl','new_source_holdout.jsonl',
                            'manifest.json','exposure_audit.json'),
        'span_human_eval_v2':('calibration.jsonl','test.jsonl'),
        'asap2_student_essays_v10':('locked_test_human.jsonl',),
        'span_ai_eval_candidate_v1':('test.jsonl',),
        'span_size_curve_v5/size_20000':('test_llmtrace.jsonl',),
        'span_sources_v5/normalized_aitdna_real':('locked_test.jsonl',),
        'pmc_publication_v6':('test.jsonl',),
        'cnn_dailymail_v1':('locked_test.jsonl',),
    }
    for folder,names in datasets.items():
        for name in names:
            path=(REPO/'data'/folder/name) if folder=='span_hardneg_v14' else (DATA/folder/name)
            add(files,path,'pangram-data/data/'+folder+'/'+name)
    adapter=ROOT/'runs/vast_hpo_selected_v3/best_adapter'
    for path in adapter.iterdir():
        if path.is_file():
            add(files,path,'pangram-data/runs/vast_hpo_selected_v3/best_adapter/'+path.name)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'purpose':'private noncommercial v14 training and frozen evaluations',
              'datasets':datasets,'files':{}}
    output=REPO/'artifacts/span_hardneg_v14_package.tar.gz'
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output,'w:gz') as tar:
        for path,name in files:
            payload=path.read_bytes()
            manifest['files'][name]=sha256(payload).hexdigest()
            info=tarfile.TarInfo(name)
            info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
        payload=json.dumps(manifest,indent=2).encode()
        info=tarfile.TarInfo('pangram-at-home/span_v14_package_manifest.json')
        info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,
                      'sha256':sha256(output.read_bytes()).hexdigest(),
                      'files':len(files)},indent=2))


if __name__=='__main__':
    main()
