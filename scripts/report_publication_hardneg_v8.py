"""Compare publication pilot v8 with balanced v6 and Pangram baselines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/f/pangram-at-home')
OUT=HERE.parent/'reports'
RUN='qwen3_token_repeat2_publication_v8_20k'
COLORS=['#169c86','#d46b28','#735a9e','#9b68a6']
NAMES=['Qwen balanced v6','Qwen publication v8','Pangram RoBERTa','Pangram Llama']


def import_comparison():
    spec=importlib.util.spec_from_file_location('compare_v6',HERE/'report_balanced_retrain_v6.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load():
    m=import_comparison();d=m.d
    baseline_pure,baseline_span=d.thresholds()
    old_pure,old_mixed,v6_pure,v6_mixed,_,v6_threshold=m.load()
    v8_threshold=json.loads((ROOT/'runs'/RUN/'v8_human_calibration.json').read_text())['threshold']
    d.MODELS[0]=('Our Qwen publication v8',RUN,COLORS[1])
    pure_map={'v6_llmtrace_heldout':'v8_llmtrace_heldout',
              'v6_synthetic_v4_val':'v8_synthetic_v4_val',
              'v6_external_articles':'v8_external_articles',
              'v6_human_locked_test':'v8_human_locked_test',
              'v6_aitdna':'v8_aitdna'}
    mixed_map={'v6_llmtrace_heldout':'v8_llmtrace_heldout',
               'v6_synthetic_v4_val':'v8_synthetic_v4_val',
               'v6_aitdna':'v8_aitdna','v6_coauthor':'v8_coauthor'}
    d.PURE=[(name,path,pure_map[file],baseline) for name,path,file,baseline in d.PURE]
    d.MIXED=[(name,path,mixed_map[file],baseline) for name,path,file,baseline in d.MIXED]
    v8_pure=d.load_pure([v8_threshold,*baseline_pure[1:]],baseline_span)
    v8_mixed=d.load_mixed([v8_threshold,*baseline_span[1:]])
    return d,old_pure,old_mixed,v6_pure,v6_mixed,v8_pure,v8_mixed,v6_threshold,v8_threshold


def models(v6,v8,source):
    return [v6[source]['models'][0],v8[source]['models'][0],*v6[source]['models'][1:]]


def bars(ax,categories,values,ylabel,lower=False):
    x=np.arange(len(categories));w=.18
    for i in range(4):
        heights=np.array(values[i])*100
        ax.bar(x+(i-1.5)*w,heights,w,color=COLORS[i],label=NAMES[i])
        for j,height in enumerate(heights):
            if height>=.3:
                ax.text(x[j]+(i-1.5)*w,height+1.1,f'{height:.0f}' if height>=10 else f'{height:.1f}',
                        ha='center',va='bottom',fontsize=6.5)
    ax.set_xticks(x,categories);ax.set_ylim(0,105);ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    if lower:ax.text(.99,.98,'LOWER IS BETTER',transform=ax.transAxes,ha='right',va='top',fontsize=8)


def doc_scores(name):
    with np.load(ROOT/'runs'/RUN/f'{name}_scores.npz') as data:
        scores=data['score'];offsets=data['document_offsets']
        return np.array([scores[offsets[i]:offsets[i+1]].max() for i in range(len(offsets)-1)])


def article_highlight_share(run,stem,threshold):
    with np.load(ROOT/'runs'/run/f'{stem}_scores.npz') as data:
        score=data['score'];offsets=data['document_offsets']
        return np.array([(score[offsets[i]:offsets[i+1]]>=threshold).mean()
                         for i in range(len(offsets)-1)])


def mixed_span_lengths(folder,filename):
    lengths=[]
    for line in (ROOT/'data'/folder/filename).open():
        row=json.loads(line)
        spans=row['spans']
        if not any(span['label']==0 for span in spans) or not any(span['label']==1 for span in spans):
            continue
        lengths.extend(len(re.findall(r'\S+',row['text'][span['start']:span['end']]))
                       for span in spans if span['label']==1)
    return len(lengths),float(np.median(lengths))


def publisher_holdout(rows,scores,target_fpr):
    """Exploratory calibration: each publisher is scored without its own humans."""
    publisher=["Reader's Digest" if row['publication']=='Readers Digest'
               else row['publication'] for row in rows]
    names=set(publisher)
    false_alarms=caught=0
    for name in names:
        controls=np.array([scores[i] for i,row in enumerate(rows)
                           if publisher[i]!=name and row['kind']=='human'])
        index=int(target_fpr*len(controls))
        threshold=float(np.nextafter(np.sort(controls)[::-1][index],np.inf))
        false_alarms+=sum(scores[i]>=threshold for i,row in enumerate(rows)
                          if publisher[i]==name and row['kind']=='human')
        caught+=sum(scores[i]>=threshold for i,row in enumerate(rows)
                    if publisher[i]==name and row['kind']=='ai')
    return false_alarms,caught


def source_aware(threshold):
    cal=doc_scores('v8_commonpile_calibration')
    own=float(np.nextafter(np.sort(cal)[::-1][int(.05*len(cal))],np.inf))
    combined=max(threshold,own)
    external=doc_scores('v8_external_articles')
    rows=[json.loads(line) for line in (ROOT/'data/span_ai_eval_candidate_v1/test.jsonl').open()]
    labels=np.array([row['kind']=='ai' for row in rows])
    heldout=doc_scores('v8_commonpile_locked_test')
    return {'original_threshold':threshold,'commonpile_5pct_threshold':own,
            'source_aware_threshold':combined,'at_source_aware':{
                'external_human_false_alarms':int(((external>=combined)&~labels).sum()),
                'external_human_total':int((~labels).sum()),
                'external_ai_caught':int(((external>=combined)&labels).sum()),
                'external_ai_total':int(labels.sum()),
                'commonpile_human_false_alarms':int((heldout>=combined).sum()),
                'commonpile_human_total':len(heldout)},
            'at_original':{'commonpile_human_false_alarms':int((heldout>=threshold).sum()),
                           'commonpile_human_total':len(heldout)}}


def main():
    d,old_pure,old_mix,v6_pure,v6_mix,v8_pure,v8_mix,v6_thr,v8_thr=load()
    adjusted=source_aware(v8_thr)
    article_v6=v6_pure['External articles']['models'][0]
    article_v8=v8_pure['External articles']['models'][0]
    article_fpr=article_v8['fp']/article_v8['human']
    article_recall=article_v8['tp']/article_v8['ai']
    article_gate=article_fpr<=.10 and article_recall>=.95
    article_rows=v8_pure['External articles']['rows']
    human_index=np.array([i for i,row in enumerate(article_rows) if row['kind']=='human'])
    substantial_v6=int((article_highlight_share('qwen3_token_repeat2_balanced_v6_20k',
                       'v6_external_articles',v6_thr)[human_index]>=.10).sum())
    substantial_v8=int((article_highlight_share(RUN,'v8_external_articles',v8_thr)
                        [human_index]>=.10).sum())
    mixed_gate=(v8_mix['LLMTrace test']['models'][0]['ai_recall']>=.75 and
                v8_mix['AITDNA collaboration']['models'][0]['human_fpr']<=.05 and
                v8_mix['CoAuthor collaboration']['models'][0]['ai_recall']>=.50)
    article_verdict='PASS' if article_gate else 'FAIL'
    mixed_verdict='PASS' if mixed_gate else 'FAIL'
    (ROOT/'runs'/RUN/'v8_source_aware_threshold_analysis.json').write_text(json.dumps(adjusted,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    pdf=OUT/'publication_hardneg_v8.pdf'
    with PdfPages(pdf) as pages:
        fig=plt.figure(figsize=(11.7,8.3))
        fig.suptitle('Publication retrain: the verdict',fontsize=21,weight='bold',y=.94)
        fig.text(.07,.86,'Fixed operating rule: 5% false alarms on 1,120 separate human controls.',fontsize=11)
        fig.text(.07,.76,f'PUBLISHED HUMAN ARTICLES  |  {article_verdict}',fontsize=16,weight='bold',
                 color='#16806d' if article_gate else '#a9403b')
        fig.text(.08,.70,f'False alarms: v6 {article_v6["fp"]}/150  →  v8 {article_v8["fp"]}/150 '
                 f'({100*article_fpr:.1f}%). Goal: at most 15/150 (10%).',fontsize=13)
        fig.text(.08,.64,f'AI articles caught: {article_v8["tp"]}/150 '
                 f'({100*article_recall:.1f}%). Goal: at least 95%.',fontsize=13)
        fig.text(.08,.59,f'Human articles with ≥10% of tokens wrongly highlighted: '
                 f'v6 {substantial_v6}/150 → v8 {substantial_v8}/150.',fontsize=11)
        fig.text(.07,.52,f'MIXED TEXT HIGHLIGHTING  |  {mixed_verdict}',fontsize=16,weight='bold',
                 color='#16806d' if mixed_gate else '#a9403b')
        fig.text(.08,.46,'AI tokens highlighted / human tokens falsely marked:',fontsize=12)
        for j,source in enumerate(['LLMTrace test','Synthetic v4 validation','AITDNA collaboration','CoAuthor collaboration']):
            m=v8_mix[source]['models'][0]
            fig.text(.09,.40-j*.055,f'{source}: {100*m["ai_recall"]:.1f}% / '
                     f'{100*m["human_fpr"]:.1f}%',fontsize=11)
        fig.text(.07,.10,'Mixed-text verdict requires ≥75% AI-token recall on LLMTrace, '
                 '≤5% human-token false marks on AITDNA, and ≥50% AI-token recall on CoAuthor.',fontsize=9)
        fig.text(.07,.055,'The external article set has attributed authors, but independent '
                 'verification of AI-free workflows is unavailable.',fontsize=9)
        pages.savefig(fig);plt.close(fig)

        manifest=json.loads((ROOT/'data/span_publication_hardneg_v8/manifest.json').read_text())
        audit=json.loads((ROOT/'data/span_publication_hardneg_v8/exposure_audit.json').read_text())
        names=['DAMASHA','mage:sci','editlens:fineweb_edu',
               'editlens:reddit_writing_prompts','paper:acl_anthology','paper:pmc_oa',
               'mage:wp','LLMTrace']
        docs=manifest['source_documents']
        tokens=audit['source_supervised_token_fraction']
        labels=[*names,'All other sources']
        doc_share=[docs[name]/manifest['documents'] for name in names]
        token_share=[tokens[name] for name in names]
        doc_share.append(1-sum(doc_share));token_share.append(1-sum(token_share))
        fig,axes=plt.subplots(1,2,figsize=(11.7,8.3),sharey=True)
        fig.suptitle('Training mix: no one source dominates',fontsize=18,weight='bold',y=.97)
        y=np.arange(len(labels))
        colors=['#d46b28' if name=='LLMTrace' else '#169c86' for name in labels]
        for ax,values,title in zip(axes,[doc_share,token_share],
                                   ['Share of 20,000 documents','Share of supervised token positions']):
            ax.barh(y,np.array(values)*100,color=colors)
            ax.set_xlim(0,50);ax.set_yticks(y,labels);ax.invert_yaxis()
            ax.set_xlabel('Share (%)');ax.set_title(title)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            for j,value in enumerate(values):
                ax.text(value*100+.5,j,f'{100*value:.1f}%',va='center',fontsize=8)
        fig.text(.06,.08,'LLMTrace: 600 documents = 3.0%; 0.92% of supervised token positions. '
                 'The remaining token labels are 57.1% human and 42.9% AI.',fontsize=9)
        fig.tight_layout(rect=(.03,.12,.98,.91));pages.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Did publication hard negatives lower false alarms?',fontsize=19,weight='bold',y=.97)
        fig.text(.055,.91,'Each Qwen threshold was calibrated to 5% false alarms on the same separate 1,120 human controls.',fontsize=9)
        sources=['LLMTrace test','External articles','Locked human test','AITDNA human controls']
        bundles=[models(v6_pure,v8_pure,s) for s in sources]
        bars(axes[0],['LLMTrace','Published articles','Other human','AITDNA controls'],
             [[b[i]['fp']/b[i]['human'] for b in bundles] for i in range(4)],
             'Human document false alarms (%)',True)
        sources=['LLMTrace test','Synthetic v4 validation','External articles']
        bundles=[models(v6_pure,v8_pure,s) for s in sources]
        bars(axes[1],['LLMTrace AI','Synthetic AI','Article AI'],
             [[b[i]['ai_recall'] for b in bundles] for i in range(4)],'AI documents caught (%)')
        fig.legend([plt.Rectangle((0,0),1,1,color=color) for color in COLORS],NAMES,
                   loc='upper left',bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pages.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        fig.suptitle('Mixed documents: passage detection and false marks',fontsize=18,weight='bold',y=.97)
        sources=['LLMTrace test','Synthetic v4 validation','AITDNA collaboration','CoAuthor collaboration']
        bundles=[models(v6_mix,v8_mix,s) for s in sources]
        labels=['LLMTrace','Synthetic','AITDNA','CoAuthor']
        bars(axes[0],labels,[[b[i]['ai_recall'] for b in bundles] for i in range(4)],
             'AI tokens highlighted (%)')
        bars(axes[1],labels,[[b[i]['human_fpr'] for b in bundles] for i in range(4)],
             'Human tokens falsely marked (%)',True)
        fig.legend([plt.Rectangle((0,0),1,1,color=color) for color in COLORS],NAMES,
                   loc='upper left',bbox_to_anchor=(.055,.885),ncol=4,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.85));pages.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.7,8.3))
        fig.suptitle('Published article ROC: ranking across thresholds',fontsize=18,weight='bold',y=.97)
        rows=v6_pure['External articles']['rows'];y=np.array([row['kind']=='ai' for row in rows])
        for i,m in enumerate(models(v6_pure,v8_pure,'External articles')):
            fpr,tpr,_=roc_curve(y,m['score'])
            ax.plot(fpr,tpr,color=COLORS[i],lw=2,label=f'{NAMES[i]} · AUC {m["auroc"]:.3f}')
            ax.scatter(m['fp']/m['human'],m['tp']/m['ai'],s=60,color=COLORS[i],zorder=4)
        ax.plot([0,1],[0,1],'--',color='#94a3b8')
        ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='False-positive rate on human articles',
               ylabel='Recall on AI articles')
        ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.9));pages.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.7,8.3))
        fig.suptitle('Which human publications are falsely flagged?',fontsize=18,weight='bold',y=.97)
        rows=v8_pure['External articles']['rows']
        names=['Associated Press','Discover','National Geographic','New York Times',
               "Reader's Digest",'Scientific American','Smithsonian Magazine','Wall Street Journal']
        aliases={'Readers Digest':"Reader's Digest"}
        groups=[[j for j,row in enumerate(rows) if row['kind']=='human' and
                 aliases.get(row.get('publication'),row.get('publication'))==name] for name in names]
        comparison=models(v6_pure,v8_pure,'External articles')
        y=np.arange(len(names));width=.17
        for i,model in enumerate(comparison):
            values=[100*sum(model['score'][j]>=model['threshold'] for j in group)/len(group)
                    for group in groups]
            ax.barh(y+(i-1.5)*width,values,width,color=COLORS[i],label=NAMES[i])
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('Attributed-human articles falsely flagged (%)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.legend(loc='lower right',ncol=2,frameon=False)
        fig.tight_layout(rect=(.04,.04,.98,.91));pages.savefig(fig);plt.close(fig)

    lines=['# Publication hard-negative pilot v8','',
           'The v8 model changes 500 of 20,000 training documents from DAMASHA mixed examples to dated, attributed human publication prose from five CC BY publishers. The architecture, tuned hyperparameters, and initialization match v6. Original article stress results have guided this experiment and are now development evidence, not an untouched final test.','',
           'LLMTrace is 600/20,000 documents (3.0%), 2.32% of training windows, and 0.92% of supervised token positions. DAMASHA is the largest source at 24.5% of documents and 27.1% of supervised positions. The final token supervision is 57.1% human and 42.9% AI.','',
           '## Plain-language verdict','',
           f'**Published articles: {article_verdict}.** At the separately calibrated threshold, v8 falsely flags {article_v8["fp"]}/150 attributed-human articles ({100*article_fpr:.1f}%), versus {article_v6["fp"]}/150 for v6. It catches {article_v8["tp"]}/150 AI articles ({100*article_recall:.1f}%). The stated goal is at most 10% human false alarms with at least 95% AI recall.','',
           f'The any-highlight document rule includes small isolated errors. A stricter descriptive view counts articles with at least 10% of tokens falsely highlighted: v6 {substantial_v6}/150; v8 {substantial_v8}/150. This is not a recalibrated operating threshold.','',
           f'**Mixed documents: {mixed_verdict}.** A useful passage highlighter should both find AI passages and leave human passages unmarked across datasets. The current check requires ≥75% AI-token recall on LLMTrace, ≤5% human-token false marks on AITDNA, and ≥50% AI-token recall on CoAuthor. This is a practical gate, not a published benchmark standard.','',
           '“AI-token recall” means the fraction of truly AI-written word/token positions the model highlights. “Human-token false marks” means the fraction of human-written positions it incorrectly highlights. “AI spans half covered” counts an AI passage only if the model highlights at least half of it; this is stricter than merely touching its edge.','',
           f'Qwen v6 threshold: `{v6_thr:.4f}`; v8 threshold: `{v8_thr:.4f}`. Both use the same independent 1,120-document calibration protocol.','',
           '## Fully human and fully AI documents','',
           '| Dataset | Model | Human false alarms | AI caught | Document AUROC |',
           '| --- | --- | ---: | ---: | ---: |']
    for source in v6_pure:
        for name,m in zip(NAMES,models(v6_pure,v8_pure,source)):
            ai=f'{m["tp"]}/{m["ai"]}' if m['ai'] else '—'
            auc=f'{m["auroc"]:.4f}' if m['auroc'] is not None else '—'
            lines.append(f'| {source} | {name} | {m["fp"]}/{m["human"]} | {ai} | {auc} |')
    lines+=['','## Mixed documents','',
            '| Dataset | Model | AI-token recall | Human-token false marks | AI spans half covered |',
            '| --- | --- | ---: | ---: | ---: |']
    for source in v6_mix:
        for name,m in zip(NAMES,models(v6_mix,v8_mix,source)):
            lines.append(f'| {source} | {name} | {100*m["ai_recall"]:.1f}% | '
                         f'{100*m["human_fpr"]:.2f}% | {100*m["span_recall_half_covered"]:.1f}% |')
    lines+=['','AI-token recall weighs long passages more heavily. The half-covered span rate '
            'gives each AI passage one vote, including short insertions. These sets differ '
            'substantially in passage length:','',
            '| Mixed set | AI passages | Median AI passage length |',
            '| --- | ---: | ---: |']
    for label,folder,filename in (
        ('LLMTrace','span_size_curve_v5/size_20000','test_llmtrace.jsonl'),
        ('Synthetic v4','span_training_v4','val.jsonl'),
        ('AITDNA','span_sources_v5/normalized_aitdna_real','locked_test.jsonl'),
        ('CoAuthor','span_realistic_eval_v1','test.jsonl')):
        n,median=mixed_span_lengths(folder,filename)
        lines.append(f'| {label} | {n} | {median:g} {"word" if median==1 else "words"} |')
    lines+=['','## Human article false alarms by publication','',
            '| Publication | Human articles | Qwen v6 | Qwen v8 | Pangram RoBERTa | Pangram Llama |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    rows=v8_pure['External articles']['rows']
    comparison=models(v6_pure,v8_pure,'External articles')
    aliases={'Readers Digest':"Reader's Digest"}
    for name in ['Associated Press','Discover','National Geographic','New York Times',
                 "Reader's Digest",'Scientific American','Smithsonian Magazine','Wall Street Journal']:
        group=[j for j,row in enumerate(rows) if row['kind']=='human' and
               aliases.get(row.get('publication'),row.get('publication'))==name]
        counts=[sum(m['score'][j]>=m['threshold'] for j in group) for m in comparison]
        lines.append(f'| {name} | {len(group)} | '+' | '.join(f'{n}/{len(group)}' for n in counts)+' |')
    lines+=['','## Separate publication controls','',
            '| Human-only source | Qwen v6 false alarms | Qwen v8 false alarms |',
            '| --- | ---: | ---: |']
    for label,v6_file,v8_file,total in (
        ('PMC pre-2023','v6_pmc_article_test','v8_pmc_article_test',346),
        ('CNN/Daily Mail historical','v6_cnn_article_test','v8_cnn_article_test',500)):
        v6=json.loads((ROOT/'runs/qwen3_token_repeat2_balanced_v6_20k'/f'{v6_file}.json').read_text())
        v8=json.loads((ROOT/'runs'/RUN/f'{v8_file}.json').read_text())
        lines.append(f'| {label} | {v6["overall"]["pure_human_documents_with_false_highlight"]}/{total} | '
                     f'{v8["overall"]["pure_human_documents_with_false_highlight"]}/{total} |')
    common=json.loads((ROOT/'runs'/RUN/'v8_commonpile_locked_test.json').read_text())
    lines.append(f'| Common Pile four unseen publishers | — | {common["overall"]["pure_human_documents_with_false_highlight"]}/200 |')
    lines+=['','## Exploratory publisher-transfer calibration','',
            'This diagnostic uses the already-inspected external article set. For each publisher, '
            'it chooses a threshold from *human articles at the other publishers*, then scores '
            'the held-out publisher. Reader’s Digest spelling variants are grouped. It is useful '
            'evidence about threshold transfer, but the data and target rates have been inspected '
            'during development, so these numbers are **not a new independent test or a deployable threshold**.','',
            '| Calibration human-FPR target | Model | Held-publisher human false alarms | Held-publisher AI caught |',
            '| ---: | --- | ---: | ---: |']
    for rate in (.05,.10):
        for name,model in [('Qwen v6',article_v6),('Qwen v8',article_v8)]:
            fp,tp=publisher_holdout(article_rows,model['score'],rate)
            lines.append(f'| {100*rate:.0f}% | {name} | {fp}/150 | {tp}/150 |')
    lines+=['','The result motivates collecting independent, pre-2023 science/magazine-style '
            'human articles for calibration and a separate locked test, with matched AI '
            'articles to measure the recall cost. Generic historical news and PMC controls '
            'were too easy and did not raise the threshold.','']
    adj=adjusted['at_source_aware']
    lines+=['','## Independent publication threshold check','',
            f'Common Pile human calibration threshold for 5% FPR: `{adjusted["commonpile_5pct_threshold"]:.4f}`; '
            f'source-aware maximum with original controls: `{adjusted["source_aware_threshold"]:.4f}`.','',
            f'At that source-aware threshold: external human false alarms {adj["external_human_false_alarms"]}/150; '
            f'external AI caught {adj["external_ai_caught"]}/150; '
            f'new held-out Common Pile human false alarms {adj["commonpile_human_false_alarms"]}/200.','',
            'Common Pile calibration and locked-test articles come from publishers absent in v8 training. Dates/bylines provide strong but not absolute authorship evidence. The external Human Detectors set has attributed authors but AI-free workflows were not independently verified. EditLens has coarse broadcast window scores for mixed text.']
    (OUT/'publication_hardneg_v8.md').write_text('\n'.join(lines)+'\n')
    print(pdf)


if __name__=='__main__':main()
