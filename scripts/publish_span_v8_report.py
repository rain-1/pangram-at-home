"""Commit and push the publication scorecard after the background report finishes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')
UPSTREAM=ROOT/'span_publication_hardneg_v8_report_status.json'
STATUS=ROOT/'span_publication_hardneg_v8_publish_status.json'
FILES=['reports/publication_hardneg_v8.md','reports/publication_hardneg_v8.pdf']


def save(state):
    STATUS.write_text(json.dumps(state,indent=2)+'\n')


def main():
    state={'phase':'waiting_for_report'};save(state)
    try:
        while True:
            upstream=json.loads(UPSTREAM.read_text())
            if upstream['phase']=='failed':
                raise RuntimeError('The v8 report failed; see its status file')
            if upstream['phase']=='complete':break
            time.sleep(30)
        state['phase']='publishing';save(state)
        subprocess.run(['git','add','--',*FILES],cwd=REPO,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet','--',*FILES],cwd=REPO)
        if changed.returncode==1:
            subprocess.run(['git','commit','-m','Report publication pilot v8 against balanced and Pangram baselines',
                            '--',*FILES],cwd=REPO,check=True)
        elif changed.returncode!=0:
            raise RuntimeError(f'git diff returned {changed.returncode}')
        subprocess.run(['git','push'],cwd=REPO,check=True)
        state['phase']='complete'
        state['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    except Exception:
        state['phase']='failed';state['traceback']=traceback.format_exc()
        raise
    finally:save(state)


if __name__=='__main__':main()
