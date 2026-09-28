"""Train half-dose GRADTEX v13 and trigger frozen-threshold evaluations."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from dotenv import load_dotenv

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')
FOLDER='span_new_sources_v13'
NAME='qwen3_token_repeat2_new_sources_v13_20k'
RUN=ROOT/'runs'/NAME
STATUS=ROOT/'span_new_sources_v13_status.json'
EVALS=[
    (FOLDER,'new_source_holdout.jsonl','v13_new_source_holdout'),
    ('asap2_student_essays_v10','locked_test_human.jsonl','v13_asap2_locked_test'),
    ('span_ai_eval_candidate_v1','test.jsonl','v13_external_articles'),
    ('span_human_eval_v2','test.jsonl','v13_human_locked_test'),
    ('span_size_curve_v5/size_20000','test_llmtrace.jsonl','v13_llmtrace_heldout'),
    ('span_sources_v5/normalized_aitdna_real','locked_test.jsonl','v13_aitdna'),
    ('pmc_publication_v6','test.jsonl','v13_pmc_article_test'),
    ('cnn_dailymail_v1','locked_test.jsonl','v13_cnn_article_test'),
]


def save(state: dict) -> None:
    state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temporary,STATUS)


def command(args: list[str], logfile: Path) -> None:
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',WANDB_PROJECT='pangram-at-home',
             WANDB_ENTITY='eac-adsf',WANDB_DIR=str(ROOT/'wandb'),WANDB_LOG_MODEL='false')
    with logfile.open('w') as stream:
        subprocess.run(args,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main() -> None:
    load_dotenv(REPO/'.env',override=False)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing')
    manifest=json.loads((ROOT/'data'/FOLDER/'manifest.json').read_text())
    audit=json.loads((ROOT/'data'/FOLDER/'exposure_audit.json').read_text())
    assert manifest['documents']==audit['documents']==20000
    assert manifest['gradtex_mixed_documents']==500
    assert .42<=audit['ai_supervised_token_fraction']<=.58
    assert max(audit['source_supervised_token_fraction'].values())<.35
    assert audit['source_supervised_token_fraction']['LLMTrace_detection']<=.03
    steps=(audit['windows']+7)//8
    state={'phase':'training','run':NAME,'max_steps':steps,
           'dataset_sha256':manifest['train_sha256'],'completed_evaluations':[]}
    save(state)
    try:
        command([sys.executable,'-u',str(REPO/'scripts/train_token_lora.py'),
                 '--root',str(ROOT),'--model',str(ROOT/'models/Qwen3-1.7B'),
                 '--init-adapter',str(ROOT/'runs/vast_hpo_selected_v3/best_adapter'),
                 '--run-name',NAME,'--dataset-folder',FOLDER,
                 '--max-steps',str(steps),'--eval-steps','500',
                 '--learning-rate','7.607757094022466e-5','--lora-rank','32',
                 '--lora-alpha','64','--lora-dropout','.068837366330751',
                 '--batch-size','2','--accumulation','4','--hours','5',
                 '--report-to','wandb','--seed','42'],ROOT/(NAME+'.train.log'))
        summary=json.loads((RUN/'train_summary.json').read_text())
        if summary['global_step']!=steps:
            raise RuntimeError(f'Training stopped at {summary["global_step"]}/{steps}')
        state['phase']='evaluating';state['wandb_url']=summary['wandb_run_url'];save(state)
        threshold=None
        for dataset,filename,output in [('span_human_eval_v2','calibration.jsonl',
                                         'v13_human_calibration'),*EVALS]:
            args=[sys.executable,'-u',str(REPO/'scripts/evaluate_span_pilot.py'),
                  '--root',str(ROOT),'--run-name',NAME,'--task','token',
                  '--dataset-folder',dataset,'--validation-file',filename,
                  '--output-name',output,'--report-to','wandb']
            if threshold is None:
                args+=['--calibration-unit','document','--target-fpr','.02']
            else:
                args+=['--threshold',repr(threshold)]
            command(args,RUN/(output+'.log'))
            report=json.loads((RUN/(output+'.json')).read_text())
            if threshold is None:
                threshold=report['threshold'];state['threshold']=threshold
            elif report['threshold']!=threshold:
                raise RuntimeError('Frozen threshold changed')
            state['completed_evaluations'].append(output);save(state)
        state['phase']='reporting';save(state)
        command([sys.executable,'-u',str(REPO/'scripts/report_span_new_sources_v13.py')],
                ROOT/'span_new_sources_v13_report.log')
        state['report']=str(REPO/'reports/span_new_sources_v13.md')
        state['phase']='complete'
    except Exception:
        state['phase']='failed';state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr,flush=True)
    finally:
        save(state)
    if state['phase']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
