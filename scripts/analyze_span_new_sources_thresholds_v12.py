"""Compare v10/v12 operating points using disjoint human calibration sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/f/pangram-at-home')
REPO = Path(__file__).resolve().parents[1]
MODELS = {
    'v10': ('qwen3_token_repeat2_essay_paired_v10_20k', 'v10'),
    'v12': ('qwen3_token_repeat2_new_sources_v12_20k', 'v12'),
}
SETS = {
    'generic_calibration': ('span_human_eval_v2/calibration.jsonl',
                            {'v10':'v10_human_calibration','v12':'v12_human_calibration'}),
    'essay_calibration': ('persuade_essay_calibration_v11/calibration.jsonl',
                          {'v10':'v11_persuade_calibration','v12':'v12_persuade_calibration'}),
    'human_locked': ('span_human_eval_v2/test.jsonl',
                     {'v10':'v10_human_locked_test','v12':'v12_human_locked_test'}),
    'external': ('span_ai_eval_candidate_v1/test.jsonl',
                 {'v10':'v10_external_articles','v12':'v12_external_articles'}),
    'llmtrace': ('span_size_curve_v5/size_20000/test_llmtrace.jsonl',
                 {'v10':'v10_llmtrace_heldout','v12':'v12_llmtrace_heldout'}),
    'aitdna': ('span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
               {'v10':'v10_aitdna','v12':'v12_aitdna'}),
    'new_holdout': ('span_new_sources_v12/new_source_holdout.jsonl',
                    {'v10':'v10_new_source_holdout','v12':'v12_new_source_holdout'}),
}


def load(model: str, set_name: str) -> dict:
    folder, _ = MODELS[model]
    data_file, stems = SETS[set_name]
    path = ROOT/'runs'/folder/(stems[model]+'_scores.npz')
    with np.load(path) as data:
        scores, labels, offsets, ids = [data[key] for key in
                                        ('score','label','document_offsets','document_ids')]
    rows = [json.loads(line) for line in (ROOT/'data'/data_file).open()]
    assert len(rows) == len(ids) == len(offsets)-1
    assert all(row['id'] == str(identity) for row, identity in zip(rows, ids))
    return {'score':scores.astype(np.float64), 'label':labels,
            'offsets':offsets, 'rows':rows}


def cutoff(cache: dict, target: float) -> float:
    values = np.asarray([cache['score'][cache['offsets'][i]:cache['offsets'][i+1]].max()
                         for i in range(len(cache['rows']))])
    assert all(row['kind']=='human' for row in cache['rows'])
    ordered = np.sort(values)[::-1]
    return float(np.nextafter(ordered[int(np.floor(target*len(ordered)))], np.inf))


def count(cache: dict, threshold: float, source: str | None = None,
          kind: str | None = None) -> dict:
    rows = cache['rows']; offsets = cache['offsets']
    indices = [i for i,row in enumerate(rows)
               if (source is None or row['source']==source)
               and (kind is None or row['kind']==kind)]
    flagged = sum(bool(np.any(cache['score'][offsets[i]:offsets[i+1]] >= threshold))
                  for i in indices)
    selected = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    y = cache['label'][selected]; pred = cache['score'][selected]>=threshold
    human = y==0; ai = y==1
    return {'documents':len(indices),'flagged_documents':flagged,
            'human_token_fpr':float(pred[human].mean()) if human.any() else None,
            'ai_token_recall':float(pred[ai].mean()) if ai.any() else None}


def report_point(caches: dict, threshold: float) -> dict:
    human = caches['human_locked']
    external = caches['external']
    llm = caches['llmtrace']
    ait = caches['aitdna']
    new = caches['new_holdout']
    return {'threshold':threshold,
            'persuade':count(human,threshold,'persuade_2.0','human'),
            'writers':count(human,threshold,'writers.stackexchange.com','human'),
            'external_human':count(external,threshold,kind='human'),
            'external_ai':count(external,threshold,kind='ai'),
            'llmtrace_human':count(llm,threshold,kind='human'),
            'llmtrace_ai':count(llm,threshold,kind='ai'),
            'llmtrace_mixed':count(llm,threshold,kind='mixed'),
            'aitdna_mixed':count(ait,threshold,kind='mixed'),
            'new_human':count(new,threshold,kind='human'),
            'new_mixed':count(new,threshold,kind='mixed')}


def pct(value) -> str:
    return '—' if value is None else f'{100*value:.1f}%'


def main() -> None:
    caches = {model:{name:load(model,name) for name in SETS} for model in MODELS}
    output = {}
    lines = ['# v12 threshold transfer', '',
             'Thresholds below are chosen only from the existing, disjoint pure-human',
             'generic and student-essay calibration sets. The source-aware rule takes',
             'their maximum at the same target rate. Locked tests are measured only',
             'after the threshold is fixed; their labels do not set the cutoff.', '',
             '| Model | Calibration target | Rule | PERSUADE alarms | Writers alarms | '
             'External human alarms | External AI-token recall | '
             'LLMTrace mixed AI-token recall / human-token FPR |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for model, by_set in caches.items():
        output[model] = {}
        for target in (.005, .01, .02, .05):
            generic = cutoff(by_set['generic_calibration'], target)
            essay = cutoff(by_set['essay_calibration'], target)
            variants = {'generic':generic, 'source_aware_max':max(generic,essay)}
            outcome = {name:report_point(by_set,cut) for name,cut in variants.items()}
            output[model][f'{target:.1%}'] = {'generic_threshold':generic,
                'essay_threshold':essay,'variants':outcome}
            for name, point in outcome.items():
                l=point['llmtrace_mixed']
                lines.append(f'| {model} | {target:.1%} | {name.replace("_"," ")} | '
                    f'{point["persuade"]["flagged_documents"]}/3,000 | '
                    f'{point["writers"]["flagged_documents"]}/579 | '
                    f'{point["external_human"]["flagged_documents"]}/150 | '
                    f'{pct(point["external_ai"]["ai_token_recall"])} | '
                    f'{pct(l["ai_token_recall"])} / {pct(l["human_token_fpr"])} |')
    out=REPO/'reports/span_new_sources_thresholds_v12'
    out.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    lines += ['', 'The new-source holdout contains no pure AI documents; its mixed',
              'recall/FPR and human alarms are included in the JSON report. A tighter',
              'threshold changes the recall–FPR balance; it does not repair a lack of',
              'discrimination on a source. Treat these as development operating points.', '']
    out.with_suffix('.md').write_text('\n'.join(lines))
    print(out.with_suffix('.md'))


if __name__ == '__main__':
    main()
