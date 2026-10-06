"""Summarize held-out attribution results and plot top-1 accuracy."""
from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/mnt/f/pangram-at-home/runs')
REPO=Path(__file__).resolve().parents[1]
REPORTS=REPO/'reports'
CONDITIONS=[
    ('v10 frozen','attribution_heads_v1'),
    ('v10 trainable','attribution_unfrozen_v1'),
    ('Qwen frozen','attribution_base_frozen_v1'),
    ('Qwen trainable','attribution_base_unfrozen_v1'),
]


def wilson(k,n):
    z=1.959963984540054
    p=k/n
    den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    spread=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return centre-spread,centre+spread


def main():
    data={}
    for name,folder in CONDITIONS:
        data[name]={task:json.loads((ROOT/folder/task/'report.json').read_text())
                    for task in ('arena','authors')}
    REPORTS.mkdir(exist_ok=True)
    for task,filename in (('arena','attribution_arena_by_model_v1.csv'),
                          ('authors','attribution_authors_by_writer_v1.csv')):
        labels=data[CONDITIONS[0][0]][task]['labels']
        source=ROOT.parent/'data/attribution_heads_v1'/task/'train.jsonl'
        train_counts=Counter(json.loads(line)['label'] for line in source.open())
        with (REPORTS/filename).open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['label','train_examples','test_examples']+
                            [name+' correct' for name,_ in CONDITIONS])
            n=3
            for label in labels:
                writer.writerow([label,train_counts[label],n]+
                                [round(data[name][task]['test']['per_label_recall'][label]*n)
                                             for name,_ in CONDITIONS])
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    colours=['#5178b9','#25477f','#b98a51','#845520']
    for ax,task,title,chance in zip(axes,('arena','authors'),
                                    ('AI model attribution · 50 labels · 3 prompts',
                                     'Human writer attribution · 4 labels'),
                                    (.02,.25)):
        n=data[CONDITIONS[0][0]][task]['test']['rows']
        values=[data[name][task]['test']['accuracy'] for name,_ in CONDITIONS]
        ci=[wilson(round(v*n),n) for v in values]
        lower=np.array([v-a for v,(a,b) in zip(values,ci)])
        upper=np.array([b-v for v,(a,b) in zip(values,ci)])
        bars=ax.bar(range(4),values,color=colours,width=.7)
        ax.errorbar(range(4),values,yerr=[lower,upper],fmt='none',
                    ecolor='#252525',capsize=4,linewidth=1.3)
        ax.axhline(chance,color='#888',linestyle='--',linewidth=1)
        ax.set_xticks(range(4),[name.replace(' ','\n') for name,_ in CONDITIONS])
        ax.set_ylim(0,1.08)
        ax.set_ylabel('Held-out top-1 accuracy')
        ax.set_title(title+f' · n={n}')
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.1%}',
                    ha='center',va='bottom',fontsize=9)
    fig.suptitle('Attribution heads: frozen vs full-backbone training',fontweight='bold')
    fig.savefig(REPORTS/'attribution_comparison_v1.png',dpi=180)
    plt.close(fig)
    lines=['# Attribution head comparison','',
           'All four conditions use identical train, validation, and test documents. '
           'Validation selects the epoch; test is evaluated once per condition.','',
           '![Held-out top-1 accuracy](attribution_comparison_v1.png)','',
           '| Initialization | Backbone | Arena (50 AI models, n=150) | Top-5 | '
           'Four writers (n=12) |',
           '|---|---|---:|---:|---:|']
    for name,_ in CONDITIONS:
        a=data[name]['arena']['test'];h=data[name]['authors']['test']
        initialization='v10 detector' if name.startswith('v10') else 'Original Qwen3-1.7B'
        mode='Frozen' if 'frozen' in name else 'Fully trainable'
        lines.append(f'| {initialization} | {mode} | '
                     f"{a['accuracy']:.1%} ({round(a['accuracy']*a['rows'])}/{a['rows']}) | "
                     f"{a['top5_accuracy']:.1%} | "
                     f"{h['accuracy']:.1%} ({round(h['accuracy']*h['rows'])}/{h['rows']}) |")
    lines.extend(['','A random uniform guess is 2% for Arena and 25% for the writers. '
                  'The bars show row-level 95% Wilson intervals. Arena responses '
                  'share just three prompts, so its intervals understate uncertainty '
                  'across new prompts. '
                  'Per-label counts are in [Arena CSV](attribution_arena_by_model_v1.csv) '
                  'and [writers CSV](attribution_authors_by_writer_v1.csv).','',
                  '## Data and protocol','',
                  '- [Arena Prose](https://huggingface.co/datasets/woog/arena-prose-100-49-models): '
                  '4,998 nonempty responses from 50 labels; 4,698 train, '
                  '150 validation, 150 test. Three responses per label are in each '
                  'held-out split, with prompts disjoint across splits. Two source '
                  'responses were empty and excluded.',
                  '- Named writers: 299 whole works across Gwern, Paul Graham, '
                  'Scott Alexander, and Eliezer Yudkowsky; 275 train, 12 validation, '
                  '12 test. The newest three works per author are test, the preceding '
                  'three validation.',
                  '- Each document is represented by up to eight 512-source-token '
                  'Repeat2 windows, stride 256, with second-copy hidden states '
                  'averaged by window and document. Full-backbone training samples '
                  'one window per document per epoch; evaluation averages all windows.',
                  '- Frozen heads are linear softmax classifiers. Trainable conditions '
                  'initialize from their corresponding frozen head and update all '
                  '1.72B backbone weights with Adafactor; the v10 adapter is merged '
                  'before training. Each task has an independent backbone checkpoint.',
                  '', '## Metrics','',
                  '| Condition | Task | Validation accuracy | Test accuracy | '
                  'Test macro-F1 | Best epoch | W&B |',
                  '|---|---|---:|---:|---:|---:|---|'])
    for name,_ in CONDITIONS:
        for task in ('arena','authors'):
            r=data[name][task]
            lines.append(f"| {name} | {task} | {r['validation']['accuracy']:.1%} | "
                         f"{r['test']['accuracy']:.1%} | {r['test']['macro_f1']:.3f} | "
                         f"{r['best_epoch']} | [run]({r['wandb_url']}) |")
    arena_rows={row['id']:row for row in map(json.loads,
        (ROOT.parent/'data/attribution_heads_v1/arena/test.jsonl').open())}
    by_category={}
    for name,folder in CONDITIONS:
        counts={category:[0,0] for category in ('explanatory','creative','practical')}
        predictions=ROOT/folder/'arena/test_predictions.jsonl'
        for prediction in map(json.loads,predictions.open()):
            category=arena_rows[prediction['id']]['prompt_category']
            counts[category][0]+=int(prediction['actual']==prediction['predicted'])
            counts[category][1]+=1
        by_category[name]=counts
    lines.extend(['','## Arena by held-out prompt','',
                  'Each category below is one held-out prompt answered by all 50 '
                  'models. Performance varies substantially by prompt.','',
                  '| Prompt category | v10 frozen | v10 trainable | Qwen frozen | Qwen trainable |',
                  '|---|---:|---:|---:|---:|'])
    for category in ('explanatory','creative','practical'):
        scores=[by_category[name][category] for name,_ in CONDITIONS]
        lines.append('| '+category+' | '+' | '.join(f'{k}/{n}' for k,n in scores)+' |')
    lines.extend(['','## Interpretation','',
                  'The Arena test contains only three prompts and three examples per '
                  'model. Per-model recall and the aggregate are therefore noisy; '
                  'a larger prompt-disjoint test is needed before claiming robust '
                  'model-name attribution.',
                  'The writer test has just 12 essays and each author comes from a '
                  'distinct publication source. High accuracy may reflect source, '
                  'format, or era cues as well as authorial style. This is an '
                  'exploratory probe, not a reliable identity detector.',''])
    (REPORTS/'attribution_comparison_v1.md').write_text('\n'.join(lines))
    print(REPORTS/'attribution_comparison_v1.md')


if __name__=='__main__':
    main()
