"""Wait for training evaluation, then score independent publication controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from dotenv import load_dotenv
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/f/pangram-at-home')
RUN=ROOT/'runs/qwen3_token_repeat2_balanced_v6_20k'
TRAIN_STATUS=ROOT/'span_balanced_v6_status.json'
STATUS=ROOT/'span_balanced_v6_publication_status.json'
EVALS=[('pmc_publication_v6','calibration.jsonl','v6_pmc_article_calibration'),
       ('pmc_publication_v6','test.jsonl','v6_pmc_article_test'),
       ('cnn_dailymail_v1','calibration.jsonl','v6_cnn_article_calibration'),
       ('cnn_dailymail_v1','locked_test.jsonl','v6_cnn_article_test')]


def save(state):
    temp=STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temp,STATUS)


def doc_scores(name):
    with np.load(RUN/f'{name}_scores.npz') as data:
        scores=data['score'];offsets=data['document_offsets']
        return np.array([scores[offsets[i]:offsets[i+1]].max() for i in range(len(offsets)-1)])


def five_percent_threshold(scores):
    ordered=np.sort(scores)[::-1]
    return float(np.nextafter(ordered[int(.05*len(ordered))],np.inf))


def counts(name,threshold):
    score=doc_scores(name)
    return {'flagged':int((score>=threshold).sum()),'total':len(score),
            'rate':float((score>=threshold).mean())}


def external_counts(threshold):
    name='v6_external_articles'
    score=doc_scores(name)
    rows=[json.loads(line) for line in (ROOT/'data/span_ai_eval_candidate_v1/test.jsonl').open()]
    assert len(rows)==len(score)
    labels=np.array([row['kind']=='ai' for row in rows])
    return {'human_false_alarms':int(((score>=threshold)&~labels).sum()),
            'human_total':int((~labels).sum()),
            'ai_caught':int(((score>=threshold)&labels).sum()),
            'ai_total':int(labels.sum())}


def main():
    load_dotenv(REPO/'.env',override=False)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing')
    state={'phase':'waiting_for_main_evaluation','completed':[]};save(state)
    try:
        while True:
            upstream=json.loads(TRAIN_STATUS.read_text())
            if upstream['phase']=='failed':
                raise RuntimeError('Main balanced run failed; see its status')
            if upstream['phase']=='complete':break
            time.sleep(30)
        state['phase']='evaluating_publications';save(state)
        threshold=json.loads((RUN/'v6_human_calibration.json').read_text())['threshold']
        env=dict(os.environ,TOKENIZERS_PARALLELISM='false',WANDB_PROJECT='pangram-at-home',
                 WANDB_ENTITY='eac-adsf',WANDB_DIR=str(ROOT/'wandb'),WANDB_LOG_MODEL='false')
        for dataset,filename,name in EVALS:
            cmd=[sys.executable,'-u',str(REPO/'scripts/evaluate_span_pilot.py'),
                 '--root',str(ROOT),'--run-name',RUN.name,'--task','token',
                 '--dataset-folder',dataset,'--validation-file',filename,
                 '--output-name',name,'--threshold',repr(threshold),'--report-to','wandb']
            with (RUN/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['completed'].append(name);save(state)
        state['phase']='reporting';save(state)
        thresholds={'original_1120_human':threshold,
                    'pmc_200':five_percent_threshold(doc_scores('v6_pmc_article_calibration')),
                    'cnn_300':five_percent_threshold(doc_scores('v6_cnn_article_calibration'))}
        thresholds['source_aware']=max(thresholds.values())
        results={}
        for method,t in [('original',threshold),('source_aware',thresholds['source_aware'])]:
            results[method]={'threshold':t,'external_articles':external_counts(t),
                             'pmc_locked_human':counts('v6_pmc_article_test',t),
                             'cnn_locked_human':counts('v6_cnn_article_test',t)}
        report={'thresholds':thresholds,'results':results,
                'method':'Each source calibration uses its own 5% document false-alarm threshold; source-aware uses their maximum. No locked test or external stress article is used for threshold selection.',
                'note':'PMC test: 346 pre-2023 articles; CNN/Daily Mail test: 500 historical journalist articles.'}
        (RUN/'v6_publication_threshold_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
        output=REPO/'reports/balanced_publication_thresholds_v6.md'
        lines=['# Independent publication calibration of balanced Qwen v6','',
               report['method'],'','## Thresholds','',
               '| Human calibration source | Threshold |','| --- | ---: |']
        for key,value in thresholds.items():lines.append(f'| {key} | {value:.4f} |')
        lines+=['','## Locked evaluation','',
                '| Decision rule | External attributed-human false alarms | External AI articles caught | PMC human false alarms | CNN/Daily Mail human false alarms |',
                '| --- | ---: | ---: | ---: | ---: |']
        for method,row in results.items():
            e=row['external_articles'];p=row['pmc_locked_human'];c=row['cnn_locked_human']
            lines.append(f"| {method} | {e['human_false_alarms']}/{e['human_total']} | "
                         f"{e['ai_caught']}/{e['ai_total']} | {p['flagged']}/{p['total']} | "
                         f"{c['flagged']}/{c['total']} |")
        lines+=['','These thresholds operate on each document’s maximum token score. A threshold can lower false alarms at the cost of AI recall; consult the Qwen/baseline ROC and mixed-token charts as well. The 150 external human articles have attributed authors but their workflows were not independently verified as AI-free.']
        output.write_text('\n'.join(lines)+'\n')
        subprocess.run([sys.executable,str(REPO/'scripts/report_balanced_retrain_v6.py')],
                       cwd=REPO,env=env,check=True)
        state['phase']='complete';state['report']=str(output)
    except Exception:
        state['phase']='failed';state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr,flush=True)
    finally:save(state)
    if state['phase']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
