"""Generate the v8 scorecard immediately when its evaluation queue completes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')
UPSTREAM=ROOT/'span_publication_hardneg_v8_status.json'
STATUS=ROOT/'span_publication_hardneg_v8_report_status.json'


def write(state):
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temporary,STATUS)


def main():
    state={'phase':'waiting_for_evaluation'};write(state)
    try:
        while True:
            upstream=json.loads(UPSTREAM.read_text())
            if upstream['phase']=='failed':raise RuntimeError('v8 training/evaluation failed; see upstream status')
            if upstream['phase']=='complete':break
            time.sleep(30)
        state['phase']='reporting';write(state)
        with (ROOT/'span_publication_hardneg_v8.report.log').open('w') as log:
            subprocess.run([sys.executable,str(REPO/'scripts/report_publication_hardneg_v8.py')],
                           cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['phase']='complete'
        state['pdf']=str(REPO/'reports/publication_hardneg_v8.pdf')
        state['markdown']=str(REPO/'reports/publication_hardneg_v8.md')
    except Exception:
        state['phase']='failed';state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr,flush=True)
    finally:write(state)
    if state['phase']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
