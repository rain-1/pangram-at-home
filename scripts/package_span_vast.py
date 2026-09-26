"""Package reviewed span data and adapters for the private Vast experiment."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/mnt/f/pangram-at-home')
REPO = Path(__file__).resolve().parents[1]


def main():
    files = []
    code = ['requirements-span.txt', 'scripts/bootstrap_span_vast.sh', 'scripts/launch_span_vast.sh',
            'scripts/train_token_lora.py', 'scripts/train_segment_lora.py',
            'scripts/span_data.py', 'scripts/span_metrics.py', 'scripts/evaluate_span_pilot.py',
            'scripts/run_span_v4_experiment.py', 'scripts/report_span_v4.py']
    for name in code:
        files.append((REPO / name, 'pangram-at-home/' + name))
    datasets = {'span_training_v4': ['train.jsonl', 'val.jsonl'],
                'span_pilot_v3': ['val.jsonl'],
                'span_human_eval_v2': ['calibration.jsonl', 'test.jsonl']}
    realistic = ROOT / 'data/span_realistic_eval_v1'
    if (realistic / 'manifest.json').exists():
        datasets[realistic.name] = ['test.jsonl']
    for folder, names in datasets.items():
        for name in names + ['manifest.json']:
            path = ROOT / 'data' / folder / name
            files.append((path, 'pangram-data/' + str(path.relative_to(ROOT))))
    for name in ('vast_hpo_selected_v3', 'qwen3_token_repeat2_v3_pilot1'):
        run = ROOT / 'runs' / name
        for path in (run / 'best_adapter').iterdir():
            if path.is_file():
                files.append((path, 'pangram-data/' + str(path.relative_to(ROOT))))
        for filename in ('run_config.json', 'train_summary.json'):
            path = run / filename
            if path.exists():
                files.append((path, 'pangram-data/' + str(path.relative_to(ROOT))))
    manifest = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'files': {}, 'purpose': 'private research training and post-training frozen evaluation',
                'datasets': datasets}
    output = ROOT / 'packages/span_v4.tar.gz'
    with tarfile.open(output, 'w:gz') as tar:
        for path, name in files:
            payload = path.read_bytes()
            # The old checkpoint config names the local drive. Relocate only its paths.
            if path.name == 'run_config.json':
                payload = payload.replace(str(ROOT).encode(), b'/workspace/pangram-data')
            manifest['files'][name] = hashlib.sha256(payload).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('pangram-at-home/span_package_manifest.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    print(output, output.stat().st_size)


if __name__ == '__main__':
    main()
