"""Render the Qwen v5 training mixture audit as tables and charts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / 'reports'
ROOT = Path('/mnt/f/pangram-at-home')
AUDIT = OUT / 'dataset_mixture_audit_v5.json'
STAGE1 = ROOT / 'data/diverse_pyramid_v1/train_full.parquet'
RUN1 = ROOT / 'runs/vast_hpo_selected_v3/run_config.json'
RUN2 = ROOT / 'runs/qwen3_token_repeat2_v5_20k_e1_local/run_config.json'


def ratio(part, total):
    return 100*part/total


def write_markdown(audit, stage1, config1, config2):
    total = audit['total'];groups = audit['groups']
    ll = groups['family']['LLMTrace']
    v4 = groups['family']['earlier v4 composites']
    overlap = audit['initialization_overlap']
    labeled = total['supervised_human_tokens']+total['supervised_ai_tokens']
    ll_labeled = ll['supervised_human_tokens']+ll['supervised_ai_tokens']
    lines = ['# Training mixture audit: Qwen3-1.7B Repeat2 token model v5', '',
             'This audit reads the exact 20,000-document training JSONL and applies the trainer’s Qwen tokenizer, offset labels, 512-source-token windows, and 256-token stride. The trainer samples shuffled **windows** uniformly; Repeat2 first-copy labels are masked, so supervised-token counts refer to the second copy only. Raw training text is not included here.', '',
             '## Two training stages', '',
             f"1. **Initialization adapter:** a binary sequence classifier trained from the diverse pyramid on {len(stage1['label']):,} documents (5,000 human and 5,000 AI), with a 512-token truncation. Its configuration sets {config1['epochs']:g} epochs but `max_steps={config1['max_steps']:,}` overrides that; {config1['max_steps']*config1['effective_batch_size']:,} example draws equal about {config1['max_steps']*config1['effective_batch_size']/len(stage1['label']):.2f} passes over this corpus. Only its LoRA backbone weights were transferred to the token model; the token head was newly initialized.",
             f"2. **Token training:** {total['documents']:,} documents became {total['windows']:,} windows. {config2['max_steps']:,} optimizer steps × effective batch {config2['effective_batch_size']} = {config2['max_steps']*config2['effective_batch_size']:,} window draws, approximately one pass over the {total['windows']:,} windows. The model processed {total['processed_tokens_repeat2']:,} Repeat2 tokens and supervised {labeled:,} second-copy token positions.", '',
             'These stage proportions cannot be combined into one percentage: the stages use different objectives and the first stage truncates each document to 512 tokens.', '',
             f"**Underlying-source reuse:** all {overlap['v4_documents_with_all_groups_in_initialization']:,} earlier-v4 token-stage composites reference only source groups already present in initialization training ({overlap['v4_source_groups_in_initialization']:,}/{overlap['v4_unique_source_groups']:,} distinct groups). Only {overlap['v4_exact_text_hashes_in_initialization']} composite is an exact full-text match; the others are new joins or edits of previously seen source material. Thus these 4,964 rows add span supervision and new combinations, but little new underlying source diversity.", '',
             '## Token-stage source concentration', '',
             '| Source family | Documents | Share | Windows | Share | Supervised token positions | Share |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, group in [('LLMTrace',ll),('Earlier v4 composites',v4)]:
        tokens=group['supervised_human_tokens']+group['supervised_ai_tokens']
        lines.append(f"| {name} | {group['documents']:,} | {ratio(group['documents'],total['documents']):.1f}% | "
                     f"{group['windows']:,} | {ratio(group['windows'],total['windows']):.1f}% | "
                     f"{tokens:,} | {ratio(tokens,labeled):.1f}% |")
    lines += ['', f"The token labels are {total['supervised_human_tokens']:,} human ({ratio(total['supervised_human_tokens'],labeled):.1f}%) and {total['supervised_ai_tokens']:,} AI ({ratio(total['supervised_ai_tokens'],labeled):.1f}%). This class balance is reasonable; **source balance is the main problem**. The 4,964 earlier-v4 rows are themselves composites of 16 named sources, so each contributes only a small slice of token training.", '',
              f"There are only {groups['source_kind']['editlens:news / human']['documents'] + groups['source_kind']['mage:xsum / human']['documents']} pure-human documents from the two non-LLMTrace news sources (`editlens:news` and `mage:xsum`). LLMTrace contributes {groups['source_kind']['LLMTrace_detection / human']['documents']:,} pure-human documents overall, but its shared construction pipeline dominates their examples.", '',
              '## Detailed source mix in the token stage', '',
              '| Source | Documents | Windows | Window share | Supervised AI share within source |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, group in sorted(groups['source'].items(),key=lambda item:-item[1]['windows']):
        own=group['supervised_human_tokens']+group['supervised_ai_tokens']
        lines.append(f"| {name} | {group['documents']:,} | {group['windows']:,} | "
                     f"{ratio(group['windows'],total['windows']):.2f}% | {ratio(group['supervised_ai_tokens'],own):.1f}% |")
    lines += ['', '## Domains and construction patterns in the token stage', '',
              'The domain labels are broad. For example, most `news` and all `article` rows here are LLMTrace records; the `news` label alone does not establish publisher-style coverage.', '',
              '| Domain | Documents | Windows | Window share |',
              '| --- | ---: | ---: | ---: |']
    for name,group in sorted(groups['domain'].items(),key=lambda item:-item[1]['windows']):
        lines.append(f"| {name} | {group['documents']:,} | {group['windows']:,} | {ratio(group['windows'],total['windows']):.1f}% |")
    lines += ['', '| Construction | Documents | Windows | Window share |',
              '| --- | ---: | ---: | ---: |']
    for name,group in sorted(groups['construction'].items(),key=lambda item:-item[1]['windows']):
        lines.append(f"| {name} | {group['documents']:,} | {group['windows']:,} | {ratio(group['windows'],total['windows']):.1f}% |")
    lines += ['', '## Initialization-stage mix', '',
              '| Domain | Documents | Share |', '| --- | ---: | ---: |']
    for name,count in Counter(stage1['domain']).most_common():
        lines.append(f'| {name} | {count:,} | {ratio(count,len(stage1["domain"])):.1f}% |')
    lines += ['', 'The initialization stage contained no LLMTrace, but it was paper-heavy (35%) and news-light (5%). The token stage then shifted sharply toward LLMTrace. Its category labels do not represent independent source diversity.', '',
              '## What this explains, and what it does not', '',
              'The source concentration makes the excellent LLMTrace holdout scores a narrow result. It plausibly encouraged reliance on patterns of that dataset and left the calibration set without published nonfiction. This is an evidence-backed hypothesis, not proof of a particular learned shortcut. The external article stress test found 74/150 human-document false alarms, spread across publications; the long-document length effect alone cannot explain its 18.4% human-token FPR.', '',
              'For the next training recipe, set **per-source and per-domain window quotas** before training, report both document and window/token shares, and keep independent human nonfiction articles in training and calibration. Preserve the 300 external stress articles as a holdout. Evaluate by source after every run so a high score on one construction pipeline cannot dominate the headline.', '',
              'The exact per-group counts, including source × kind and source-family × domain, are in [dataset_mixture_audit_v5.json](dataset_mixture_audit_v5.json).', '']
    (OUT/'dataset_mixture_audit_v5.md').write_text('\n'.join(lines))


def render(audit, stage1):
    groups=audit['groups'];total=audit['total']
    figs=[]
    fig,ax=plt.subplots(figsize=(10,4.7))
    names=['Documents','Windows','Supervised tokens']
    ll=groups['family']['LLMTrace']
    allvals=[ratio(ll['documents'],total['documents']),ratio(ll['windows'],total['windows']),
             ratio(ll['supervised_human_tokens']+ll['supervised_ai_tokens'],
                   total['supervised_human_tokens']+total['supervised_ai_tokens'])]
    y=np.arange(3)
    ax.barh(y,allvals,color='#075383',height=.55,label='LLMTrace')
    ax.barh(y,[100-value for value in allvals],left=allvals,color='#81b4c7',height=.55,label='16 other named sources')
    for i,value in enumerate(allvals):
        ax.text(value/2,i,f'{value:.1f}%',ha='center',va='center',color='white',fontsize=12)
    ax.set(xlim=(0,100),yticks=y,yticklabels=names,xlabel='Share (%)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Token-stage concentration: one source dominates',fontsize=15)
    fig.text(.5,.02,'Dark blue: LLMTrace    Light blue: 16 other named sources',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,.92))
    figs.append(('concentration',fig))

    sources=sorted(groups['source'].items(),key=lambda x:-x[1]['windows'])
    fig,ax=plt.subplots(figsize=(11,7.6))
    y=np.arange(len(sources))
    values=[ratio(v['windows'],total['windows']) for _,v in sources]
    ax.barh(y,values,color=['#075383' if i==0 else '#81b4c7' for i in range(len(sources))],height=.72)
    ax.set_yticks(y,[k for k,_ in sources]);ax.invert_yaxis()
    ax.set(xlabel='Training-window share (%)',title='The 16 other sources each contribute less than 5% of windows',xlim=(0,80))
    for i,value in enumerate(values):ax.text(value+.6,i,f'{value:.2f}%',va='center',fontsize=8)
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();figs.append(('sources',fig))

    first=Counter(stage1['domain'])
    second={k:ratio(v['windows'],total['windows']) for k,v in groups['domain'].items()}
    names=sorted(set(first)|set(second))
    fig,ax=plt.subplots(figsize=(11,7))
    x=np.arange(len(names));width=.38
    ax.bar(x-width/2,[ratio(first[k],len(stage1['domain'])) for k in names],width,color='#9c7a3c',label='Initialization: document share')
    ax.bar(x+width/2,[second.get(k,0) for k in names],width,color='#075383',label='Token stage: window share')
    ax.set_xticks(x,names,rotation=55,ha='right');ax.set(ylabel='Share (%)',title='Category shift between the two training stages')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    fig.tight_layout();figs.append(('domains',fig))
    with PdfPages(OUT/'dataset_mixture_audit_v5.pdf') as pdf:
        for name,fig in figs:
            pdf.savefig(fig)
            fig.savefig(OUT/f'dataset_mixture_audit_v5_{name}.png',dpi=170)
            plt.close(fig)


if __name__=='__main__':
    audit=json.loads(AUDIT.read_text())
    config1=json.loads(RUN1.read_text());config2=json.loads(RUN2.read_text())
    stage1=pq.read_table(STAGE1,columns=['label','domain']).to_pydict()
    write_markdown(audit,stage1,config1,config2)
    render(audit,stage1)
