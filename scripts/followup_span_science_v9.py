"""Wait for v9 completion, run independent magazine checks, and build report."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
STATUS = ROOT/'span_science_paired_v9_status.json'
OUTPUT = ROOT/'span_science_paired_v9_followup_status.json'
V8 = 'qwen3_token_repeat2_publication_v8_20k'
V9 = 'qwen3_token_repeat2_science_paired_v9_20k'


def save(state):
    state['updated_at_unix'] = time.time()
    temp = OUTPUT.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    os.replace(temp, OUTPUT)


def command(args, log):
    with log.open('w') as f:
        subprocess.run([sys.executable, '-u', str(REPO/'scripts'/args[0]), *args[1:]],
                       cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    (ROOT/'reports').mkdir(exist_ok=True)
    state = {'phase': 'waiting_for_v9', 'completed': []}
    save(state)
    deadline = time.monotonic()+8*3600
    try:
        while True:
            upstream = json.loads(STATUS.read_text())
            if upstream['phase'] == 'complete':
                break
            if upstream['phase'] == 'failed':
                raise RuntimeError('Upstream v9 runner failed; inspect '+str(STATUS))
            if time.monotonic() > deadline:
                raise TimeoutError('v9 did not complete within eight hours')
            time.sleep(30)
        threshold_v9 = upstream['threshold']
        assert (ROOT/'runs'/V9/'best_adapter/adapter_model.safetensors').exists()
        state['phase'] = 'scoring'
        save(state)
        jobs = [
            (V8, 'smithsonian_archive_v10', 'train_candidates.jsonl', 'v10_magazine_train_v8', 5.03125),
            (V8, 'smithsonian_archive_v10', 'locked_test_human.jsonl', 'v10_magazine_test_v8', 5.03125),
            (V8, 'science_articles_v9', 'paired_archived_locked_test_windows_pilot.jsonl',
             'v9_archived_epa_paired_v8', 5.03125),
            (V9, 'smithsonian_archive_v10', 'train_candidates.jsonl', 'v10_magazine_train_v9', threshold_v9),
            (V9, 'smithsonian_archive_v10', 'locked_test_human.jsonl', 'v10_magazine_test_v9', threshold_v9),
        ]
        for run, folder, filename, name, threshold in jobs:
            existing = ROOT/'runs'/run/(name+'.json')
            if not existing.exists():
                command(['evaluate_span_pilot.py', '--run-name', run, '--task', 'token',
                         '--dataset-folder', folder, '--validation-file', filename,
                         '--threshold', repr(threshold), '--output-name', name],
                        ROOT/'runs'/run/(name+'.log'))
            result = json.loads(existing.read_text())
            if result['threshold'] != threshold:
                raise ValueError(f'Frozen threshold mismatch in {name}')
            state['completed'].append(name)
            save(state)
        state['phase'] = 'analyzing'
        save(state)
        for run, suffix, generic, magazine, external, magtest, epa in (
            (V8, 'v8', 'v8_human_calibration', 'v10_magazine_train_v8',
             'v8_external_articles', 'v10_magazine_test_v8', 'v9_archived_epa_human_v8'),
            (V9, 'v9', 'v9_human_calibration', 'v10_magazine_train_v9',
             'v9_external_articles', 'v10_magazine_test_v9', 'v9_archived_epa_human')):
            output = ROOT/'reports'/f'science_v9_magazine_calibration_{suffix}.json'
            command(['analyze_publication_calibration_v10.py', '--run-name', run,
                     '--generic-calibration', generic, '--magazine-calibration', magazine,
                     '--external-test', external, '--magazine-test', magtest,
                     '--archived-epa-test', epa, '--output', str(output)],
                    ROOT/'reports'/f'science_v9_magazine_calibration_{suffix}.log')
            state['completed'].append(str(output))
            save(state)
        command(['report_span_science_v9.py'], ROOT/'reports/science_paired_v9_comparison.log')
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
