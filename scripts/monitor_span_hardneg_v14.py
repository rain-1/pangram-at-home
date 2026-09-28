"""Collect and verify the v14 Vast export, then close its rental.

Run this locally as a detached process. All progress is written under
artifacts/vast_span_v14 so an interrupted interactive session cannot orphan
the rental.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from dotenv import load_dotenv


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / 'artifacts/vast_span_v14'
ROOT = Path('/mnt/f/pangram-at-home')
INSTANCE = 53167732
HOST = '154.64.230.50'
PORT = 50482
NAME = 'qwen3_token_repeat2_hardneg_v14_21k'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
       '-i', str(Path.home()/'.ssh/id_ed25519'), '-p', str(PORT), f'root@{HOST}']
RSYNC = ['rsync', '--partial', '--append-verify', '-e',
         'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 '
         f'-o ServerAliveCountMax=3 -i {Path.home()/".ssh/id_ed25519"} -p {PORT}']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def record(**kwargs) -> None:
    path = LOCAL/'monitor_status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(kwargs, updated_at_unix=time.time())
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(tmp, path)


def copy(remote: str, local: Path) -> None:
    for attempt in range(6):
        result = run(RSYNC+[f'root@{HOST}:{remote}', str(local)], timeout=900)
        if result.returncode == 0:
            return
        record(transfer_error=result.stderr[-500:], transfer_attempt=attempt+1)
        time.sleep(15)
    raise RuntimeError(f'Could not copy {remote}')


def collect(state: dict) -> None:
    exported = state.get('export')
    if not exported:
        raise RuntimeError('Remote terminal status has no export')
    archive = LOCAL/'span_hardneg_v14_export.tar.gz'
    record(phase='copying', remote_phase=state['phase'])
    copy(exported['archive'], archive)
    if sha(archive) != exported['sha256'] or archive.stat().st_size != exported['bytes']:
        raise RuntimeError('Export archive checksum/size mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        manifest_name = 'span_hardneg_v14_export_manifest.json'
        if len(names) != len(set(names)) or manifest_name not in names:
            raise RuntimeError('Duplicate/missing export member')
        manifest = json.load(tar.extractfile(manifest_name))
        if set(manifest['files']) != set(names)-{manifest_name}:
            raise RuntimeError('Manifest does not list exactly the archive files')
        for member in members:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or member.size > 5*1024**3):
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            if member.name != manifest_name and not (
                    member.name.startswith(f'runs/{NAME}/') or member.name in {
                        'span_hardneg_v14_status.json', NAME+'.train.log',
                        'span_v14_bootstrap.log'}):
                raise RuntimeError(f'Unexpected archive member: {member.name}')
        for name, meta in manifest['files'].items():
            source = tar.extractfile(name)
            if source is None:
                raise RuntimeError(f'Missing export member: {name}')
            target = ROOT/name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            temp = target.with_name(target.name+'.v14tmp')
            size = 0
            with temp.open('wb') as out:
                for block in iter(lambda: source.read(1024*1024), b''):
                    out.write(block); h.update(block); size += len(block)
            if h.hexdigest() != meta['sha256'] or size != meta['bytes']:
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'Export member checksum/size mismatch: {name}')
            os.replace(temp, target)
    record(phase='verified', archive_sha256=sha(archive),
           archive_bytes=archive.stat().st_size, files=len(manifest['files']))


def close() -> None:
    cli = '/home/ubuntu/.venvs/pangram-vast-cli/bin/vastai'
    result = run([cli, '--api-key', os.environ['VAST_API_KEY'], '--raw',
                  'destroy', 'instance', str(INSTANCE), '-y'])
    if result.returncode:
        raise RuntimeError('Vast destroy failed: '+result.stderr[-500:])
    record(phase='closed', closed_at_unix=time.time())


def main() -> None:
    load_dotenv(REPO/'.env')
    LOCAL.mkdir(parents=True, exist_ok=True)
    state = json.loads((LOCAL/'instance.json').read_text())
    assert state['instance_id'] == INSTANCE
    deadline = state['created_at_unix'] + 3600*state['max_rental_hours']
    record(phase='monitoring', instance_id=INSTANCE)
    while True:
        result = run(SSH+['cat /workspace/pangram-data/span_hardneg_v14_status.json'], timeout=30)
        if result.returncode == 0:
            remote = json.loads(result.stdout)
            record(remote_phase=remote['phase'], remote_step=remote.get('global_step'),
                   completed_evaluations=remote.get('completed_evaluations', []))
            if remote['phase'] in {'complete', 'failed'} and 'export' in remote:
                collect(remote)
                close()
                return
            if remote['phase'] == 'failed' and 'export' not in remote:
                record(phase='remote_failed_without_export')
                for filename in ('span_hardneg_v14_status.json', 'span_v14_bootstrap.log',
                                 NAME+'.train.log', 'span_v14_controller.log'):
                    try:
                        copy('/workspace/pangram-data/'+filename, LOCAL/filename)
                    except Exception:
                        pass
                close()
                return
        if time.time() >= deadline:
            record(phase='deadline_exceeded', error='Rental reached eight-hour cap')
            for filename in ('span_hardneg_v14_status.json', 'span_v14_bootstrap.log',
                             NAME+'.train.log', 'span_v14_controller.log'):
                try:
                    copy('/workspace/pangram-data/'+filename, LOCAL/filename)
                except Exception:
                    pass
            close()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record(phase='monitor_error', error=str(exc))
        raise
