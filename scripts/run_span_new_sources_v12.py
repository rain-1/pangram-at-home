"""Train the fixed-size v12 mixture and evaluate each held-out source automatically."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
FOLDER = 'span_new_sources_v12'
NAME = 'qwen3_token_repeat2_new_sources_v12_20k'
RUN = ROOT/'runs'/NAME
STATUS = ROOT/'span_new_sources_v12_status.json'
EVALS = [
    (FOLDER, 'new_source_holdout.jsonl', 'v12_new_source_holdout'),
    ('asap2_student_essays_v10', 'locked_test_human.jsonl', 'v12_asap2_locked_test'),
    ('span_ai_eval_candidate_v1', 'test.jsonl', 'v12_external_articles'),
    ('span_human_eval_v2', 'test.jsonl', 'v12_human_locked_test'),
    ('span_size_curve_v5/size_20000', 'test_llmtrace.jsonl', 'v12_llmtrace_heldout'),
    ('span_sources_v5/normalized_aitdna_real', 'locked_test.jsonl', 'v12_aitdna'),
    ('pmc_publication_v6', 'test.jsonl', 'v12_pmc_article_test'),
    ('cnn_dailymail_v1', 'locked_test.jsonl', 'v12_cnn_article_test'),
]


def save(state: dict) -> None:
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    os.replace(temporary, STATUS)


def command(args: list[str], logfile: Path) -> None:
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', WANDB_PROJECT='pangram-at-home',
               WANDB_ENTITY='eac-adsf', WANDB_DIR=str(ROOT/'wandb'), WANDB_LOG_MODEL='false')
    (ROOT/'wandb').mkdir(exist_ok=True)
    with logfile.open('w') as file:
        subprocess.run(args, cwd=REPO, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)


def compute_pids() -> list[str]:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]


def wait_for_gpu(state: dict, timeout_hours: float = 12) -> None:
    deadline = time.monotonic()+timeout_hours*3600
    while True:
        pids = compute_pids()
        if not pids:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'GPU remained busy for {timeout_hours} hours: {pids}')
        state['phase'] = 'waiting_for_gpu'
        state['other_compute_pids'] = pids
        save(state)
        time.sleep(60)


def main() -> None:
    load_dotenv(REPO/'.env', override=False)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing')
    manifest = json.loads((ROOT/'data'/FOLDER/'manifest.json').read_text())
    audit = json.loads((ROOT/'data'/FOLDER/'exposure_audit.json').read_text())
    assert manifest['documents'] == audit['documents'] == 20000
    assert manifest['added_documents'] == 1860
    assert audit['source_supervised_token_fraction']['LLMTrace_detection'] <= .03
    assert max(audit['source_supervised_token_fraction'].values()) < .35
    assert .42 <= audit['ai_supervised_token_fraction'] <= .58
    steps = (audit['windows']+7)//8
    state = {'phase': 'preparing', 'run': NAME, 'max_steps': steps,
             'completed_evaluations': [], 'dataset_sha256': manifest['train_sha256']}
    save(state)
    try:
        wait_for_gpu(state)
        state['phase'] = 'training'
        state.pop('other_compute_pids', None)
        save(state)
        command([sys.executable, '-u', str(REPO/'scripts/train_token_lora.py'),
                 '--root', str(ROOT), '--model', str(ROOT/'models/Qwen3-1.7B'),
                 '--init-adapter', str(ROOT/'runs/vast_hpo_selected_v3/best_adapter'),
                 '--run-name', NAME, '--dataset-folder', FOLDER,
                 '--max-steps', str(steps), '--eval-steps', '500',
                 '--learning-rate', '7.607757094022466e-5', '--lora-rank', '32',
                 '--lora-alpha', '64', '--lora-dropout', '.068837366330751',
                 '--batch-size', '2', '--accumulation', '4', '--hours', '5',
                 '--report-to', 'wandb', '--seed', '42'], ROOT/(NAME+'.train.log'))
        summary = json.loads((RUN/'train_summary.json').read_text())
        if summary['global_step'] != steps:
            raise RuntimeError(f'Training stopped at {summary["global_step"]}/{steps}')
        state['phase'] = 'evaluating'
        state['wandb_url'] = summary['wandb_run_url']
        save(state)
        threshold = None
        for dataset, filename, output in [('span_human_eval_v2', 'calibration.jsonl',
                                           'v12_human_calibration'), *EVALS]:
            args = [sys.executable, '-u', str(REPO/'scripts/evaluate_span_pilot.py'),
                    '--root', str(ROOT), '--run-name', NAME, '--task', 'token',
                    '--dataset-folder', dataset, '--validation-file', filename,
                    '--output-name', output, '--report-to', 'wandb']
            if threshold is None:
                args += ['--calibration-unit', 'document', '--target-fpr', '.02']
            else:
                args += ['--threshold', repr(threshold)]
            command(args, RUN/(output+'.log'))
            report = json.loads((RUN/(output+'.json')).read_text())
            if threshold is None:
                threshold = report['threshold']
                state['threshold'] = threshold
            elif report['threshold'] != threshold:
                raise RuntimeError('Frozen threshold changed')
            state['completed_evaluations'].append(output)
            save(state)
        # Score the same newly held-out rows with the already trained v10 model.
        state['phase'] = 'comparing_models'
        save(state)
        v10_run = ROOT/'runs/qwen3_token_repeat2_essay_paired_v10_20k'
        v10_threshold = json.loads((v10_run/'v10_human_calibration.json').read_text())['threshold']
        command([sys.executable, '-u', str(REPO/'scripts/evaluate_span_pilot.py'),
                 '--root', str(ROOT), '--run-name', v10_run.name, '--task', 'token',
                 '--dataset-folder', FOLDER, '--validation-file', 'new_source_holdout.jsonl',
                 '--output-name', 'v10_new_source_holdout', '--threshold', repr(v10_threshold),
                 '--report-to', 'wandb'], v10_run/'v10_new_source_holdout.log')
        for model in ('roberta', 'llama'):
            output_name = f'open_pangram_editlens_{model}_new_sources_v12'
            command([sys.executable, '-u', str(REPO/'scripts/evaluate_open_pangram_span_v5.py'),
                     '--model', model, '--sets', 'calibration', 'new_source_holdout',
                     '--target-fpr', '.02', '--output-name', output_name],
                    ROOT/(output_name+'.log'))
            state['completed_evaluations'].append(output_name)
            save(state)
        state['phase'] = 'complete'
    except Exception:
        state['phase'] = 'failed'
        state['traceback'] = traceback.format_exc()
        print(state['traceback'], file=sys.stderr, flush=True)
    finally:
        save(state)
    if state['phase'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
