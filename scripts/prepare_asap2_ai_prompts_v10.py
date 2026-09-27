"""Select v9 essay false alarms and prepare source-matched AI writing prompts."""
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/f/pangram-at-home')
DATA = ROOT/'data/asap2_student_essays_v10'
RUN = ROOT/'runs/qwen3_token_repeat2_science_paired_v9_20k'
SCORE = RUN/'v10_asap2_train_diagnostic_v9_scores.npz'
REPORT = RUN/'v10_asap2_train_diagnostic_v9.json'
MODELS = ('qwen2_5_3b', 'smollm2_1_7b')
PROMPT_PDFS = {
    'Car-free cities': 'FL2_car-free_cities.pdf',
    'Does the electoral college work?': 'FL1_does_the_electoral_college_work.pdf',
}
QUOTAS = {'Car-free cities': 192, 'Does the electoral college work?': 64}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_texts():
    texts = {}
    with zipfile.ZipFile(DATA/'ASAP_2_source_texts.zip') as archive:
        for name, pdf_name in PROMPT_PDFS.items():
            member = next(path for path in archive.namelist()
                          if path.endswith('/'+pdf_name) and not path.startswith('__MACOSX'))
            with tempfile.NamedTemporaryFile(suffix='.pdf') as handle:
                handle.write(archive.read(member))
                handle.flush()
                raw = subprocess.run(['pdftotext', '-layout', handle.name, '-'],
                                     capture_output=True, text=True, check=True).stdout
            text = re.sub(r'[ \t]+', ' ', raw).replace('\f', '\n').strip()
            assert len(text.split()) > 1000
            texts[name] = text
    return texts


def main():
    rows = [json.loads(line) for line in (DATA/'train_candidates.jsonl').open()]
    report = json.loads(REPORT.read_text())
    assert report['validation_sha256'] == sha((DATA/'train_candidates.jsonl').read_bytes())
    with np.load(SCORE) as data:
        offsets, scores, ids = (data[key] for key in
                                ('document_offsets', 'score', 'document_ids'))
        risks = {str(row_id): scores[offsets[i]:offsets[i+1]]
                 for i, row_id in enumerate(ids)}
    assert set(risks) == {row['id'] for row in rows}
    threshold = report['threshold']
    for row in rows:
        scores = risks[row['id']]
        row['v9_false_positive_tokens'] = int((scores >= threshold).sum())
        row['v9_max_score'] = float(scores.max())
    selected = []
    for prompt_name, limit in QUOTAS.items():
        candidates = [row for row in rows if row['prompt_name'] == prompt_name]
        ordered = sorted(candidates,
                         key=lambda row: (row['v9_false_positive_tokens'],
                                          row['v9_max_score'], row['id']), reverse=True)
        assert len(ordered) >= limit
        selected.extend(ordered[:limit])
    # Keep only fields needed for text provenance and hardness; no demographics.
    for row in selected:
        row['selection'] = 'highest_v9_false_positive_token_count_within_prompt'
    with (DATA/'selected_train_human.jsonl').open('w') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    with zipfile.ZipFile(DATA/'ASAP_2_Final_github_train.zip') as archive:
        frame = pd.read_csv(archive.open('ASAP_2_Final_github_train.csv'),
                            usecols=['essay_id', 'assignment', 'prompt_name'])
    assignment = {}
    for prompt_name in QUOTAS:
        values = frame.loc[frame.prompt_name == prompt_name, 'assignment'].dropna()
        assignment[prompt_name] = values.value_counts().index[0]
    sources = source_texts()
    prompts = {model: [] for model in MODELS}
    for row in selected:
        model = MODELS[int(sha(row['id'].encode()), 16) % 2]
        target = min(600, max(330, row['words']))
        prompt = (f'Write an original student essay of about {target} words in response to '
                  'the assignment below. Use the supplied reading as context and write a '
                  'complete response in natural prose with paragraphs. Write only the essay '
                  'body, without a title, list, or introductory note. Do not copy the reading.\n\n'
                  f'Assignment: {assignment[row["prompt_name"]]}\n\n'
                  f'Source reading:\n{sources[row["prompt_name"]]}')
        prompts[model].append({'human_id': row['id'], 'split': 'asap2_train_ai_candidate',
                               'source': row['source'], 'topic_title': row['prompt_name'],
                               'human_text_sha256': row['text_sha256'],
                               'prompt_version': 'asap2_source_based_essay_v1',
                               'target_words': target, 'prompt': prompt})
    for model, items in prompts.items():
        (DATA/f'train_prompts_{model}.jsonl').write_text(''.join(
            json.dumps(item, ensure_ascii=False)+'\n' for item in items))
    manifest = {'selected_human': len(selected),
                'selection_by_prompt': dict(Counter(row['prompt_name'] for row in selected)),
                'false_positive_documents': sum(row['v9_false_positive_tokens'] > 0 for row in selected),
                'false_positive_tokens': sum(row['v9_false_positive_tokens'] for row in selected),
                'model_prompt_counts': {model: len(items) for model, items in prompts.items()},
                'v9_diagnostic_sha256': sha(REPORT.read_bytes()),
                'source_pdf_sha256': sha((DATA/'ASAP_2_source_texts.zip').read_bytes()),
                'selected_human_sha256': sha((DATA/'selected_train_human.jsonl').read_bytes())}
    (DATA/'selection_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
