"""Evaluate disjoint student-essay calibration and source-aware thresholds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_span_essays_v10 import MODELS, ROOT, RUNS, REPORTS, load, result, threshold

REPO = Path(__file__).resolve().parents[1]
STATUS = ROOT/'persuade_calibration_v11_followup_status.json'
UPSTREAM = ROOT/'span_essay_paired_v10_status.json'
DATASET = 'persuade_essay_calibration_v11'
STEM = 'v11_persuade_calibration'


def save(state):
    state['updated_at_unix'] = time.time()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temp,STATUS)


def essay_scores(tag):
    import numpy as np
    path = RUNS/MODELS[tag][0]/(STEM+'_scores.npz')
    with np.load(path) as data:
        return {key:data[key] for key in ('score','label','document_offsets')}


def student_indices():
    path = ROOT/'data/span_human_eval_v2/test.jsonl'
    return [i for i,line in enumerate(path.open())
            if json.loads(line)['source']=='persuade_2.0']


def analyze():
    REPORTS.mkdir(exist_ok=True)
    students=student_indices()
    report={}
    for tag in MODELS:
        generic=load(tag,'calibration')
        student=essay_scores(tag)
        report[tag]={}
        for target in (.005,.01,.02,.05):
            tg=threshold(generic,target)
            ts=threshold(student,target)
            variants={'generic_only':tg,'source_aware_max':max(tg,ts)}
            outcomes={}
            for name,cutoff in variants.items():
                outcomes[name]={
                    'threshold':cutoff,
                    'persuade':result(load(tag,'generic'),cutoff,students),
                    'cnn':result(load(tag,'cnn'),cutoff),
                    'external':result(load(tag,'external'),cutoff),
                    'llmtrace':result(load(tag,'llmtrace'),cutoff),
                }
            report[tag][f'{target:.1%}']={'generic_threshold':tg,
                                         'essay_threshold':ts,'variants':outcomes}
    (REPORTS/'student_calibration_v11.json').write_text(json.dumps(report,indent=2)+'\n')
    pdf=REPORTS/'student_calibration_v11.pdf'
    with PdfPages(pdf) as pages:
        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        for tag,(_,color) in MODELS.items():
            for variant,linestyle in [('generic_only','--'),('source_aware_max','-')]:
                points=[]
                for row in report[tag].values():
                    values=row['variants'][variant]
                    points.append((values['persuade']['human_flagged'],
                                   values['external']['human_flagged'],
                                   100*values['external']['ai_token_recall']))
                axes[0].plot([p[0] for p in points],[p[2] for p in points],
                             color=color,ls=linestyle,marker='o',label=f'{tag} {variant}')
                axes[1].plot([p[1] for p in points],[p[2] for p in points],
                             color=color,ls=linestyle,marker='o',label=f'{tag} {variant}')
        axes[0].set(xlabel='PERSUADE human essays flagged (of 3,000)',
                    title='Student essay false alarms')
        axes[1].set(xlabel='External human articles flagged (of 150)',
                    title='Publication false alarms')
        for ax in axes:
            ax.set_ylabel('External AI-token recall (%)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Source-aware thresholds from separate human calibration sets',
                     x=.06,ha='left',fontsize=15)
        fig.text(.06,.01,'Dashed = generic calibration only; solid = maximum of generic and '
                 'independent essay calibration thresholds.',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);pages.savefig(fig);plt.close(fig)
    markdown=['# Student essay calibration transfer','',
              'A separate 1,000-essay PERSUADE calibration set is disjoint by essay ID and '
              'exhaustive 24-word passage matching from the locked 3,000-essay test. '
              'Each source-aware threshold is the maximum of the generic-human and '
              'student-essay calibration cutoffs at the stated target. '
              'No locked-test labels set these cutoffs. This is an in-domain calibration '
              'experiment, not a guarantee for other student populations.','',
              '| Model | Calibration target | Rule | PERSUADE human alarms | CNN alarms | '
              'External human alarms | External AI-token recall | LLMTrace AI-token recall / FPR |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for tag in MODELS:
        for target,row in report[tag].items():
            for variant,values in row['variants'].items():
                p=values['persuade'];c=values['cnn'];e=values['external'];m=values['llmtrace']
                markdown.append(f'| {tag} | {target} | {variant.replace("_"," ")} | '
                                f'{p["human_flagged"]}/3,000 | {c["human_flagged"]}/500 | '
                                f'{e["human_flagged"]}/150 | {e["ai_token_recall"]:.1%} | '
                                f'{m["ai_token_recall"]:.1%} / {m["human_token_fpr"]:.1%} |')
    markdown+=['','PERSUADE text and calibration files stay on local research storage; '
               'they are not added to model training or the Git repository. '
               'Older v8/v9 saved score exports were float32, so retrospective threshold '
               'tie counts can differ by one document from direct evaluation.','',
               f'[Download calibration chart]({pdf.name})','']
    (REPORTS/'student_calibration_v11.md').write_text('\n'.join(markdown))


def main():
    state={'phase':'waiting_for_v10','completed':[]}
    save(state)
    deadline=time.monotonic()+8*3600
    try:
        while True:
            if UPSTREAM.exists():
                upstream=json.loads(UPSTREAM.read_text())
                if upstream['phase']=='complete':break
                if upstream['phase']=='failed':raise RuntimeError('v10 upstream failed')
            if time.monotonic()>deadline:raise TimeoutError('v10 did not finish within eight hours')
            time.sleep(30)
        state['phase']='scoring';save(state)
        for tag,(run,_) in MODELS.items():
            output=RUNS/run/(STEM+'.json')
            if not output.exists():
                log=RUNS/run/(STEM+'.log')
                with log.open('w') as handle:
                    subprocess.run([sys.executable,'-u',str(REPO/'scripts/evaluate_span_pilot.py'),
                                    '--run-name',run,'--task','token',
                                    '--dataset-folder',DATASET,
                                    '--validation-file','calibration.jsonl',
                                    '--calibration-unit','document','--target-fpr','.02',
                                    '--output-name',STEM],cwd=REPO,
                                   stdout=handle,stderr=subprocess.STDOUT,check=True)
            state['completed'].append(tag)
            save(state)
        state['phase']='analyzing';save(state)
        analyze()
        state['phase']='complete'
    except Exception:
        state['phase']='failed'
        state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr,flush=True)
    finally:
        save(state)
    if state['phase']!='complete':raise SystemExit(1)


if __name__=='__main__':
    main()
