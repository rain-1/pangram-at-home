"""Summarize held-out attribution results and plot top-1 accuracy."""
from __future__ import annotations

import json
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
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    colours=['#5178b9','#25477f','#b98a51','#845520']
    for ax,task,title,chance in zip(axes,('arena','authors'),
                                    ('AI model attribution · 50 labels',
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
                  'The bars show 95% Wilson intervals for test accuracy.','',
                  '## Data and protocol','',
                  '- Arena Prose: 4,998 nonempty responses from 50 labels; 4,698 train, '
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
    lines.extend(['','## Interpretation','',
                  'The Arena test contains only three examples per model. Per-model '
                  'recall is therefore noisy; aggregate results and larger external '
                  'tests are needed before claiming robust model-name attribution.',
                  'The writer test has just 12 essays and each author comes from a '
                  'distinct publication source. High accuracy may reflect source, '
                  'format, or era cues as well as authorial style. This is an '
                  'exploratory probe, not a reliable identity detector.',''])
    (REPORTS/'attribution_comparison_v1.md').write_text('\n'.join(lines))
    print(REPORTS/'attribution_comparison_v1.md')


if __name__=='__main__':
    main()
