"""Finish AI generation, audit data, train v10, evaluate, and publish charts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
DATA = ROOT/'data/asap2_student_essays_v10'
STATUS = ROOT/'span_essay_paired_v10_pipeline_status.json'
RUNS = ROOT/'runs'


def save(state):
    state['updated_at_unix'] = time.time()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    os.replace(temporary, STATUS)


def count(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def command(script, arguments, logfile):
    with logfile.open('w') as handle:
        subprocess.run([sys.executable, '-u', str(REPO/'scripts'/script), *arguments],
                       cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def baseline_threshold(run, prefix):
    path = RUNS/run/(prefix+'_human_calibration_scores.npz')
    with np.load(path) as data:
        scores, offsets = data['score'], data['document_offsets']
        maxima = np.array([scores[offsets[i]:offsets[i+1]].max()
                           for i in range(len(offsets)-1)])
    ordered = np.sort(maxima)[::-1]
    return float(np.nextafter(ordered[int(np.floor(.02*len(ordered)))], np.inf))


def main():
    state = {'phase':'waiting_for_qwen', 'completed':[]}
    save(state)
    try:
        qwen = DATA/'generated_qwen2_5_3b_train.jsonl'
        deadline = time.monotonic()+2*3600
        while count(qwen) < 119:
            if time.monotonic() > deadline:
                raise TimeoutError('Qwen essay generation did not reach 119 rows within two hours')
            time.sleep(30)
        assert count(qwen) == 119
        state['completed'].append('qwen_generation')
        state['phase'] = 'generating_smollm'
        save(state)
        command('generate_science_articles_v9_ai.py',
                ['--model','smollm2_1_7b',
                 '--prompt-file',str(DATA/'train_prompts_smollm2_1_7b.jsonl'),
                 '--output-file',str(DATA/'generated_smollm2_1_7b_train.jsonl'),
                 '--batch-size','2','--max-new-tokens','1000','--min-new-tokens','280'],
                ROOT/'asap2_v10_smollm_generation.log')
        if count(DATA/'generated_smollm2_1_7b_train.jsonl') != 137:
            raise ValueError('SmolLM generation row count mismatch')
        state['completed'].append('smollm_generation')
        state['phase'] = 'auditing'
        save(state)
        command('audit_asap2_ai_v10.py', [], ROOT/'asap2_v10_ai_audit.log')
        audit = json.loads((DATA/'ai_audit_manifest.json').read_text())
        if audit['accepted_pairs'] < 150:
            raise ValueError(f'Too few accepted AI pairs: {audit["accepted_pairs"]}')
        state['accepted_pairs'] = audit['accepted_pairs']
        state['completed'].append('ai_audit')
        command('build_span_essays_v10.py', [], ROOT/'asap2_v10_build.log')
        command('audit_span_balanced_v6_exposure.py',
                ['--dataset-folder','span_essay_paired_v10'],
                ROOT/'asap2_v10_exposure.log')
        state['completed'].append('dataset_build')
        state['phase'] = 'baseline_scoring'
        save(state)
        for tag,run in [('v8','qwen3_token_repeat2_publication_v8_20k'),
                        ('v9','qwen3_token_repeat2_science_paired_v9_20k')]:
            stem = f'v10_asap2_locked_test_{tag}'
            cutoff = baseline_threshold(run,tag)
            output = RUNS/run/(stem+'.json')
            if not output.exists():
                command('evaluate_span_pilot.py',
                        ['--run-name',run,'--task','token',
                         '--dataset-folder','asap2_student_essays_v10',
                         '--validation-file','locked_test_human.jsonl',
                         '--threshold',repr(cutoff),'--output-name',stem],
                        RUNS/run/(stem+'.log'))
            if json.loads(output.read_text())['threshold'] != cutoff:
                raise ValueError(f'Threshold mismatch in {stem}')
            state['completed'].append(stem)
            save(state)
        state['phase'] = 'training_and_evaluating'
        save(state)
        command('run_span_essays_v10.py', [], ROOT/'span_essay_paired_v10_driver.log')
        upstream = json.loads((ROOT/'span_essay_paired_v10_status.json').read_text())
        if upstream['phase'] != 'complete':
            raise RuntimeError('v10 training/evaluation did not complete')
        state['completed'].append('v10_training_and_evaluation')
        state['phase'] = 'reporting'
        save(state)
        command('report_span_essays_v10.py', [], ROOT/'reports/essay_paired_v10_comparison.log')
        state['completed'].append('comparison_report')
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
