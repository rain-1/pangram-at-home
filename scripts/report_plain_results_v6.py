"""Plain-language scorecard from the frozen v5 model/baseline predictions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dashboard_v5', HERE/'report_evaluation_dashboard_v5.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
OUT = HERE.parent/'reports/plain_results_v6.pdf'
COLORS = [x[2] for x in d.MODELS]
NAMES = ['Our Qwen 20k', 'Pangram RoBERTa', 'Pangram Llama']


def title(fig, heading, explanation):
    fig.text(.055, .955, heading, fontsize=19, weight='bold', va='top')
    fig.text(.055, .91, explanation, fontsize=10, va='top', color='#334155')


def grouped(ax, categories, values, ylabel, ylim=(0, 105), lower_better=False):
    x = np.arange(len(categories)); width = .24
    for i, name in enumerate(NAMES):
        bars = ax.bar(x+(i-1)*width, values[i], width, label=name, color=COLORS[i])
        for bar, value in zip(bars, values[i]):
            ax.text(bar.get_x()+bar.get_width()/2, min(value+2, 103), f'{value:.0f}',
                    ha='center', fontsize=7, rotation=0)
    ax.set_xticks(x, categories)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    if lower_better:
        ax.text(.99, .99, 'LOWER IS BETTER', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#475569')


def main():
    pure_thresholds, span_thresholds = d.thresholds()
    pure = d.load_pure(pure_thresholds, span_thresholds)
    mixed = d.load_mixed(span_thresholds)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9})
    with PdfPages(OUT) as pdf:
        fig, axes = plt.subplots(2,1,figsize=(11.7,8.3))
        title(fig, '1. Whole-document decisions',
              'At the frozen threshold. A false alarm means a fully human document was flagged as AI.')
        human_sets = ['LLMTrace test','Synthetic v4 validation','External articles','Locked human test','AITDNA human controls']
        labels = ['LLMTrace\n720 human','Synthetic\n152 human','Published articles\n150 human',
                  'Other human\n3,579','AITDNA controls\n103']
        grouped(axes[0],labels,[[100*pure[s]['models'][i]['fp']/pure[s]['models'][i]['human'] for s in human_sets]
                                 for i in range(3)], 'Human false alarms (%)', lower_better=True)
        ai_sets = ['LLMTrace test','Synthetic v4 validation','External articles']
        grouped(axes[1],['LLMTrace\n516 AI','Synthetic\n148 AI','Article AI\n150'],
                [[100*pure[s]['models'][i]['ai_recall'] for s in ai_sets] for i in range(3)],
                'AI documents caught (%)')
        fig.legend([Patch(facecolor=c) for c in COLORS],NAMES,loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=3,frameon=False,fontsize=9)
        fig.text(.055,.025,'Verdict: our Qwen catches article AI, but flags 74 of 150 human articles. That false-alarm rate is unacceptable.',fontsize=10,weight='bold')
        fig.tight_layout(rect=(.04,.055,.98,.86));pdf.savefig(fig);plt.close(fig)

        fig, axes = plt.subplots(3,1,figsize=(11.7,8.3))
        title(fig,'2. Can the models highlight AI passages in mixed writing?',
              'Read each row as “out of 100”. AITDNA is real collaboration; LLMTrace/synthetic are constructed; CoAuthor has short insertions.')
        sets = ['LLMTrace test','Synthetic v4 validation','AITDNA collaboration','CoAuthor collaboration']
        labels = ['LLMTrace','Synthetic','AITDNA','CoAuthor']
        grouped(axes[0],labels,[[100*mixed[s]['models'][i]['ai_recall'] for s in sets] for i in range(3)],
                'AI tokens highlighted (%)')
        grouped(axes[1],labels,[[100*mixed[s]['models'][i]['human_fpr'] for s in sets] for i in range(3)],
                'Human tokens falsely marked (%)',lower_better=True)
        grouped(axes[2],labels,[[100*mixed[s]['models'][i]['span_recall_half_covered'] for s in sets] for i in range(3)],
                'AI spans half covered (%)')
        fig.legend([Patch(facecolor=c) for c in COLORS],NAMES,loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=3,frameon=False,fontsize=9)
        fig.text(.055,.025,'Verdict: Qwen is useful on AITDNA, misses most LLMTrace AI text, and misses all CoAuthor AI at this threshold.',fontsize=10,weight='bold')
        fig.tight_layout(rect=(.04,.055,.98,.86));pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11.7,8.3))
        title(fig,'3. Threshold-free document ROC curves',
              'A curve near the upper-left is better. The dots mark the frozen operating threshold used on pages 1–2.')
        for ax, source in zip(axes,['LLMTrace test','External articles']):
            bundle=pure[source];labels=np.array([r['kind']=='ai' for r in bundle['rows']])
            for i,model in enumerate(bundle['models']):
                fpr,tpr,_=roc_curve(labels,model['score'])
                ax.plot(fpr,tpr,color=COLORS[i],lw=2,label=f"{NAMES[i]} · AUC {roc_auc_score(labels,model['score']):.3f}")
                ax.scatter(model['fp']/model['human'],model['tp']/model['ai'],color=COLORS[i],s=55,zorder=4)
            ax.plot([0,1],[0,1],ls='--',color='#94a3b8',lw=1)
            ax.set_xlim(0,1);ax.set_ylim(0,1.03);ax.set_xlabel('False-positive rate on human documents')
            ax.set_ylabel('Recall on AI documents');ax.set_title(source,fontsize=13)
            ax.legend(loc='lower right',fontsize=8,frameon=False)
            ax.grid(alpha=.15)
        fig.text(.055,.07,'The article ROC also shows ranking weakness: AUC 0.937 for Qwen versus 0.999/1.000 for the baselines.\n'
                 'Moving the threshold alone will trade away AI recall; source-balanced retraining is the next test.',fontsize=10)
        fig.tight_layout(rect=(.04,.12,.98,.86));pdf.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(11.7,8.3))
        title(fig,'4. Human-publication false alarms by source',
              'Share of attributed-human articles falsely flagged by each model.')
        names=['Associated Press','Discover','National Geographic','New York Times',
               "Reader’s Digest",'Scientific American','Smithsonian','Wall Street Journal']
        totals=[15,20,25,20,15,15,25,15]
        errors=[[4,8,17,7,10,5,15,8],
                [0,0,1,0,0,0,0,0],
                [0,1,4,0,1,0,0,1]]
        y=np.arange(len(names));height=.22
        for i in range(3):
            vals=[100*a/b for a,b in zip(errors[i],totals)]
            ax.barh(y+(i-1)*height,vals,color=COLORS[i],height=height,label=NAMES[i])
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('Human articles falsely flagged (%)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.legend(frameon=False,ncol=3,loc='lower right')
        fig.text(.055,.075,'These articles have named human authors; their production workflows were not independently verified as AI-free.\n'
                 'This 150-article set stays frozen for evaluation. It will not be used to set a new threshold or as training text.',fontsize=10)
        fig.tight_layout(rect=(.04,.14,.98,.86));pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3))
        title(fig,'5. Which mixed-text categories fail?',
              'LLMTrace held-out source categories. In each row, AI recall is the share of AI tokens found; false marks are human tokens wrongly highlighted.')
        llmt=mixed['LLMTrace test'];categories=sorted({r['domain'] for r in llmt['rows'] if r['kind']=='mixed'})
        ai_values=[[],[],[]];human_values=[[],[],[]]
        for category in categories:
            indices=[j for j,row in enumerate(llmt['rows']) if row['kind']=='mixed' and row['domain']==category]
            rows=[llmt['rows'][j] for j in indices]
            for i in range(3):
                model=llmt['models'][i]
                tokens=[model['tokens'][j] for j in indices]
                stats=d.mixed_stats(rows,tokens,model['threshold'])
                ai_values[i].append(100*stats['ai_recall'])
                human_values[i].append(100*stats['human_fpr'])
        x=np.arange(len(categories));width=.24
        for ax,values,ylabel in [(axes[0],ai_values,'AI tokens found (%)'),
                                  (axes[1],human_values,'Human tokens falsely marked (%)')]:
            for i in range(3):ax.bar(x+(i-1)*width,values[i],width,color=COLORS[i],label=NAMES[i])
            ax.set_xticks(x,[v.replace('_','\n') for v in categories],fontsize=8)
            ax.set_ylim(0,105 if ax is axes[0] else 30)
            ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.legend([Patch(facecolor=c) for c in COLORS],NAMES,loc='upper left',
                   bbox_to_anchor=(.055,.885),ncol=3,frameon=False,fontsize=9)
        fig.text(.055,.025,'Our Qwen misses especially many AI tokens in questions, reviews, and stories. Its very low false-mark rate here comes with that low recall.',fontsize=10)
        fig.tight_layout(rect=(.04,.055,.98,.86));pdf.savefig(fig);plt.close(fig)
    print(OUT)


if __name__=='__main__':
    main()
