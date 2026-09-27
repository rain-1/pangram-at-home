"""Compare frozen calibration thresholds without selecting on test articles."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/f/pangram-at-home')


def load_scores(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def calibration_threshold(data, target):
    offsets, scores = data['document_offsets'], data['score']
    values = np.array([scores[offsets[i]:offsets[i+1]].max()
                       for i in range(len(offsets)-1)])
    ordered = np.sort(values)[::-1]
    return float(np.nextafter(ordered[int(np.floor(target*len(ordered)))], np.inf))


def evaluate(data, threshold):
    offsets, scores, labels = data['document_offsets'], data['score'], data['label']
    predicted = scores >= threshold
    human, ai = labels == 0, labels == 1
    human_any = ai_any = human_docs = ai_docs = 0
    for i in range(len(offsets)-1):
        a, b = offsets[i:i+2]
        y = labels[a:b]
        if len(y) == 0:
            continue
        if np.all(y == 0):
            human_docs += 1
            human_any += bool(predicted[a:b].any())
        elif np.all(y == 1):
            ai_docs += 1
            ai_any += bool(predicted[a:b].any())
    return {'threshold': threshold, 'human_token_fpr': float(predicted[human].mean()) if human.any() else None,
            'ai_token_recall': float(predicted[ai].mean()) if ai.any() else None,
            'human_documents_with_any_highlight': human_any,
            'human_documents': human_docs,
            'ai_documents_with_any_highlight': ai_any, 'ai_documents': ai_docs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--generic-calibration', required=True, help='Artifact stem without _scores.npz')
    p.add_argument('--magazine-calibration', required=True)
    p.add_argument('--external-test', required=True)
    p.add_argument('--magazine-test', required=True)
    p.add_argument('--archived-epa-test', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run = ROOT/'runs'/args.run_name
    stems = {'generic': args.generic_calibration, 'magazine': args.magazine_calibration,
             'external': args.external_test, 'magazine_test': args.magazine_test,
             'archived_epa': args.archived_epa_test}
    scores = {key: load_scores(run/(stem+'_scores.npz')) for key, stem in stems.items()}
    thresholds = {}
    for target in (.01, .02, .05):
        generic = calibration_threshold(scores['generic'], target)
        magazine = calibration_threshold(scores['magazine'], target)
        thresholds[f'{int(target*100)}pct'] = {'generic': generic, 'magazine': magazine,
                                               'source_aware_max': max(generic, magazine)}
    report = {'run_name': args.run_name,
              'calibration_document_counts': {key: len(scores[key]['document_offsets'])-1
                                              for key in ('generic', 'magazine')},
              'thresholds': thresholds, 'evaluations': {}}
    for target, variants in thresholds.items():
        report['evaluations'][target] = {
            variant: {name: evaluate(scores[name], threshold)
                      for name in ('external', 'magazine_test', 'archived_epa')}
            for variant, threshold in variants.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
