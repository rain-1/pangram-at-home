"""Finalize the autonomous local size-curve run after training and evaluation end."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/f/pangram-at-home')
STATUS = ROOT / 'span_size_curve_v5_local_status.json'
OUTPUTS = (
    REPO / 'reports/span_size_curve_v5.md',
    REPO / 'reports/span_size_curve_v5.pdf',
    REPO / 'reports/span_size_curve_v5_roc.pdf',
    REPO / 'reports/span_size_curve_v5_domains.pdf',
)


def log(message: str) -> None:
    print(f'{datetime.now(timezone.utc).isoformat()} {message}', flush=True)


def main() -> None:
    deadline = time.monotonic() + 7 * 3600
    while time.monotonic() < deadline:
        status = json.loads(STATUS.read_text())
        if status['phase'] == 'failed':
            raise RuntimeError(status.get('traceback', 'Local runner failed'))
        if status['phase'] == 'complete':
            break
        time.sleep(45)
    else:
        raise TimeoutError('Local size curve did not finish within seven hours')
    log('All four runs complete; generating report')
    subprocess.run([sys.executable, str(REPO / 'scripts/report_span_size_curve_v5.py')], cwd=REPO, check=True)
    for path in OUTPUTS:
        if not path.is_file() or path.stat().st_size < 500:
            raise RuntimeError(f'Missing or empty report: {path}')
    subprocess.run(['git', 'add', *[str(path.relative_to(REPO)) for path in OUTPUTS]], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode == 0:
        log('Report already committed')
        return
    subprocess.run(['git', 'commit', '-m', 'Report completed span data-size curve'], cwd=REPO, check=True)
    subprocess.run(['git', 'push'], cwd=REPO, check=True)
    log('Report committed and pushed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'ERROR: {exc}')
        raise
