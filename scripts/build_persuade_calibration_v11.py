"""Make an essay-only, evaluation-only calibration set disjoint from locked tests."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from build_span_human_eval_v2 import from_row
from prepare_asap2_student_essays_v10 import shingles

ROOT = Path('/mnt/f/pangram-at-home/data')
SOURCE = ROOT/'persuade_essays_v1/human_eval.parquet'
LOCKED = ROOT/'span_human_eval_v2/test.jsonl'
OUTPUT = ROOT/'persuade_essay_calibration_v11'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUTPUT}')
    locked = [json.loads(line) for line in LOCKED.open()]
    locked = [row for row in locked if row['source'] == 'persuade_2.0']
    assert len(locked) == 3000
    locked_ids = {record['text_id'] for row in locked for record in row['source_records']}
    protected = set()
    for row in locked:
        protected.update(shingles(row['text']))
    rows = pq.read_table(SOURCE).to_pylist()
    ordered = sorted(rows, key=lambda row:sha(('persuade-cal-v11:'+row['text_id']).encode()))
    selected = []
    rejected = {'locked_id':0,'length':0,'24_word_overlap':0}
    for row in ordered:
        if row['text_id'] in locked_ids:
            rejected['locked_id'] += 1
        elif not 300 <= row['word_count'] <= 1200:
            rejected['length'] += 1
        elif shingles(row['text']) & protected:
            rejected['24_word_overlap'] += 1
        else:
            selected.append(from_row(row,'essay_calibration_v11'))
            if len(selected) == 1000:
                break
    assert len(selected) == 1000
    assert not ({rec['text_id'] for row in selected for rec in row['source_records']} & locked_ids)
    OUTPUT.mkdir(parents=True)
    path = OUTPUT/'calibration.jsonl'
    with path.open('w') as handle:
        for row in selected:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest = {'role':'student-essay threshold calibration only; never training',
                'source':'PERSUADE 2.0 local research evaluation corpus',
                'documents':len(selected),'protected_locked_test_documents':len(locked),
                'protected_24_word_shingles':len(protected),'rejections_before_selection':rejected,
                'locked_test_sha256':sha(LOCKED.read_bytes()),
                'source_sha256':sha(SOURCE.read_bytes()),
                'calibration_sha256':sha(path.read_bytes()),
                'license_caveat':'local evaluation only under ingested CC BY-NC-SA 4.0 source terms',
                'privacy':'no student demographics exported; raw essay text stays on external disk'}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in
                      ('documents','protected_24_word_shingles','rejections_before_selection')},indent=2))


if __name__=='__main__':
    main()
