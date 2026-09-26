"""Train the source-capped mix and evaluate at one independent fixed threshold."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
NAME = 'qwen3_token_repeat2_balanced_v6_20k'
RUN = ROOT/'runs'/NAME
STATUS = ROOT/'span_balanced_v6_status.json'
EVALS = [
    ('span_size_curve_v5/size_20000','test_llmtrace.jsonl','v6_llmtrace_heldout'),
    ('span_training_v4','val.jsonl','v6_synthetic_v4_val'),
    ('span_ai_eval_candidate_v1','test.jsonl','v6_external_articles'),
    ('span_human_eval_v2','test.jsonl','v6_human_locked_test'),
    ('span_sources_v5/normalized_aitdna_real','locked_test.jsonl','v6_aitdna'),
    ('span_realistic_eval_v1','test.jsonl','v6_coauthor'),
]


def save(state):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temporary,STATUS)


def command(args, logfile):
    env = dict(os.environ,TOKENIZERS_PARALLELISM='false',WANDB_PROJECT='pangram-at-home',
               WANDB_ENTITY='eac-adsf',WANDB_DIR=str(ROOT/'wandb'),WANDB_LOG_MODEL='false')
    (ROOT/'wandb').mkdir(exist_ok=True)
    with logfile.open('w') as file:
        subprocess.run(args,cwd=REPO,env=env,stdout=file,stderr=subprocess.STDOUT,check=True)


def main():
    load_dotenv(REPO/'.env',override=False)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing')
    manifest=json.loads((ROOT/'data/span_balanced_v6/manifest.json').read_text())
    assert manifest['llmtrace_document_fraction'] == .03
    assert manifest['llmtrace_window_fraction'] <= .03
    assert max(manifest['source_documents'].values())/manifest['documents'] <= .30
    steps=(sum(manifest['source_windows_512_256'].values())+7)//8
    state={'phase':'training','run':NAME,'max_steps':steps,'completed_evaluations':[]}
    save(state)
    try:
        command([sys.executable,'-u',str(REPO/'scripts/train_token_lora.py'),
                 '--root',str(ROOT),'--model',str(ROOT/'models/Qwen3-1.7B'),
                 '--init-adapter',str(ROOT/'runs/vast_hpo_selected_v3/best_adapter'),
                 '--run-name',NAME,'--dataset-folder','span_balanced_v6',
                 '--max-steps',str(steps),'--eval-steps','500',
                 '--learning-rate','7.607757094022466e-5','--lora-rank','32',
                 '--lora-alpha','64','--lora-dropout','.068837366330751',
                 '--batch-size','2','--accumulation','4','--hours','5',
                 '--report-to','wandb','--seed','42'],ROOT/(NAME+'.train.log'))
        summary=json.loads((RUN/'train_summary.json').read_text())
        if summary['global_step'] != steps:
            raise RuntimeError(f'Training ended at {summary["global_step"]}/{steps}')
        state['phase']='evaluating';state['wandb_url']=summary['wandb_run_url'];save(state)
        threshold=None
        for dataset,filename,output in [('span_human_eval_v2','calibration.jsonl','v6_human_calibration'),*EVALS]:
            cmd=[sys.executable,'-u',str(REPO/'scripts/evaluate_span_pilot.py'),
                 '--root',str(ROOT),'--run-name',NAME,'--task','token',
                 '--dataset-folder',dataset,'--validation-file',filename,
                 '--output-name',output,'--report-to','wandb']
            if threshold is None:
                cmd+=['--calibration-unit','document','--target-fpr','.05']
            else:
                cmd+=['--threshold',repr(threshold)]
            command(cmd,RUN/(output+'.log'))
            report=json.loads((RUN/(output+'.json')).read_text())
            if threshold is None:
                threshold=report['threshold'];state['threshold']=threshold
            elif report['threshold'] != threshold:
                raise RuntimeError('Frozen threshold changed')
            state['completed_evaluations'].append(output);save(state)
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
