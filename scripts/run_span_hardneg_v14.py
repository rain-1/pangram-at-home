"""Durable Vast v14 training, calibration, locked evaluation, and export."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback


ROOT = Path('/workspace/pangram-data')
REPO = Path('/workspace/pangram-at-home')
NAME = 'qwen3_token_repeat2_hardneg_v14_21k'
RUN = ROOT/'runs'/NAME
STATUS = ROOT/'span_hardneg_v14_status.json'
EVALS = [
    ('span_hardneg_v14','new_source_holdout.jsonl','v14_new_source_holdout'),
    ('asap2_student_essays_v10','locked_test_human.jsonl','v14_asap2_locked_test'),
    ('span_ai_eval_candidate_v1','test.jsonl','v14_external_articles'),
    ('span_human_eval_v2','test.jsonl','v14_human_locked_test'),
    ('span_size_curve_v5/size_20000','test_llmtrace.jsonl','v14_llmtrace_heldout'),
    ('span_sources_v5/normalized_aitdna_real','locked_test.jsonl','v14_aitdna'),
    ('pmc_publication_v6','test.jsonl','v14_pmc_article_test'),
    ('cnn_dailymail_v1','locked_test.jsonl','v14_cnn_article_test'),
]


def save(state: dict) -> None:
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temp,STATUS)


def command(args: list[str], logfile: Path) -> None:
    env = dict(os.environ,TOKENIZERS_PARALLELISM='false',WANDB_PROJECT='pangram-at-home',
               WANDB_ENTITY='eac-adsf',WANDB_DIR=str(ROOT/'wandb'),WANDB_LOG_MODEL='false')
    with logfile.open('w') as stream:
        subprocess.run(args,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def export(state: dict) -> None:
    files=[]
    for path in RUN.rglob('*'):
        if not path.is_file():
            continue
        if 'checkpoint-' in str(path) or '/wandb/' in str(path):
            continue
        if path.suffix in {'.json','.jsonl','.npz','.log','.safetensors','.model'} or path.name in {
                'tokenizer.json','tokenizer_config.json','special_tokens_map.json'}:
            files.append(path)
    files += [path for path in (STATUS,ROOT/(NAME+'.train.log'),ROOT/'span_v14_bootstrap.log') if path.exists()]
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'files':{str(path.relative_to(ROOT)):{'sha256':digest(path),'bytes':path.stat().st_size}
                       for path in files}}
    manifest_path=ROOT/'span_hardneg_v14_export_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'span_hardneg_v14_export.tar.gz'
    temporary=archive.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz') as tar:
        tar.add(manifest_path,arcname=manifest_path.name)
        for path in files:
            tar.add(path,arcname=str(path.relative_to(ROOT)))
    os.replace(temporary,archive)
    state['export']={'archive':str(archive),'sha256':digest(archive),
                     'bytes':archive.stat().st_size,'files':len(files)}
    save(state)


def main() -> None:
    manifest=json.loads((ROOT/'data/span_hardneg_v14/manifest.json').read_text())
    audit=json.loads((ROOT/'data/span_hardneg_v14/exposure_audit.json').read_text())
    assert manifest['documents']==audit['documents']==21200
    assert .40<=audit['ai_supervised_token_fraction']<=.50
    assert max(audit['source_supervised_token_fraction'].values())<.35
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing')
    steps=(audit['windows']+7)//8
    state={'phase':'training','run':NAME,'max_steps':steps,
           'dataset_sha256':manifest['train_sha256'],'completed_evaluations':[]}
    save(state)
    try:
        command([sys.executable,'-u',str(REPO/'scripts/train_token_lora.py'),
                 '--root',str(ROOT),'--model',str(ROOT/'models/Qwen3-1.7B'),
                 '--init-adapter',str(ROOT/'runs/vast_hpo_selected_v3/best_adapter'),
                 '--run-name',NAME,'--dataset-folder','span_hardneg_v14',
                 '--max-steps',str(steps),'--eval-steps','500',
                 '--learning-rate','7.607757094022466e-5','--lora-rank','32',
                 '--lora-alpha','64','--lora-dropout','.068837366330751',
                 '--batch-size','2','--accumulation','4','--hours','5',
                 '--report-to','wandb','--seed','42'],ROOT/(NAME+'.train.log'))
        summary=json.loads((RUN/'train_summary.json').read_text())
        if summary['global_step']!=steps:
            raise RuntimeError(f'Training stopped at {summary["global_step"]}/{steps}')
        state['phase']='evaluating'
        state['wandb_url']=summary['wandb_run_url']
        save(state)
        threshold=None
        for dataset,filename,output in [('span_human_eval_v2','calibration.jsonl',
                                         'v14_human_calibration'),*EVALS]:
            args=[sys.executable,'-u',str(REPO/'scripts/evaluate_span_pilot.py'),
                  '--root',str(ROOT),'--run-name',NAME,'--task','token',
                  '--dataset-folder',dataset,'--validation-file',filename,
                  '--output-name',output,'--report-to','wandb']
            if threshold is None:
                args+=['--calibration-unit','document','--target-fpr','.02']
            else:
                args+=['--threshold',repr(threshold)]
            state['active_evaluation']=output
            save(state)
            command(args,RUN/(output+'.log'))
            report=json.loads((RUN/(output+'.json')).read_text())
            if threshold is None:
                threshold=report['threshold']
                state['threshold']=threshold
            elif report['threshold']!=threshold:
                raise RuntimeError('Frozen threshold changed')
            state['completed_evaluations'].append(output)
            save(state)
        state['phase']='complete'
    except Exception:
        state['phase']='failed'
        state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr,flush=True)
    finally:
        save(state)
        try:
            export(state)
        except Exception:
            state['export_error']=traceback.format_exc()
            save(state)
            print(state['export_error'],file=sys.stderr,flush=True)
    if state['phase']!='complete' or 'export' not in state:
        raise SystemExit(1)


if __name__=='__main__':
    main()
