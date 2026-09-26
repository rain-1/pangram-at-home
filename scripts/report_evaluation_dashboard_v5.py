"""Unified, source-aware evaluation of Qwen and the two open EditLens models.

All predictions are cached from prior frozen-threshold evaluations. No fitting,
threshold selection, or new inference occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path('/mnt/f/pangram-at-home')
RUNS = ROOT / 'runs'
OUT = Path(__file__).resolve().parents[1] / 'reports'
MODELS = [
    ('Our Qwen 20k', 'qwen3_token_repeat2_v5_20k_e1_local', '#075383'),
    ('EditLens RoBERTa', 'open_pangram_editlens_roberta_v5', '#735a9e'),
    ('EditLens Llama', 'open_pangram_editlens_llama_v5', '#9b68a6'),
]
PURE = [
    ('LLMTrace test', 'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
     'v5_llmtrace_heldout', 'llmtrace_pure'),
    ('Synthetic v4 validation', 'span_training_v4/val.jsonl',
     'v5_prior_synthetic_val', 'synthetic_v4_pure'),
    ('External articles', 'span_ai_eval_candidate_v1/test.jsonl',
     'v5_external_human_detectors', 'external_human_detectors'),
    ('Locked human test', 'span_human_eval_v2/test.jsonl',
     'v5_human_locked_test', 'locked_human'),
    ('AITDNA human controls', 'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
     'v5_aitdna', 'aitdna'),
]
MIXED = [
    ('LLMTrace test', 'span_size_curve_v5/size_20000/test_llmtrace.jsonl',
     'v5_llmtrace_heldout', 'llmtrace_heldout'),
    ('Synthetic v4 validation', 'span_training_v4/val.jsonl',
     'v5_prior_synthetic_val', 'synthetic_v4_val'),
    ('AITDNA collaboration', 'span_sources_v5/normalized_aitdna_real/locked_test.jsonl',
     'v5_aitdna', 'aitdna'),
    ('CoAuthor collaboration', 'span_realistic_eval_v1/test.jsonl',
     'v5_coauthor', 'coauthor'),
]


def read_rows(relative):
    return [json.loads(line) for line in (ROOT / 'data' / relative).open()]


def qwen_tokens(file, rows):
    path = RUNS / MODELS[0][1] / f'{file}_scores.npz'
    with np.load(path) as data:
        scores, labels = data['score'], data['label']
        offsets, ids = data['document_offsets'], data['document_ids']
        assert len(rows) == len(ids)
        assert all(row['id'] == str(id_) for row, id_ in zip(rows, ids))
        return [(scores[offsets[i]:offsets[i+1]].astype(np.float64),
                 labels[offsets[i]:offsets[i+1]].astype(np.int8))
                for i in range(len(rows))]


def span_tokens(model_index, file, rows):
    model = ('roberta', 'llama')[model_index - 1]
    path = RUNS / f'open_pangram_editlens_{model}_span_v5' / f'{file}_scores.npz'
    with np.load(path) as data:
        scores, labels = data['score'], data['label']
        offsets, ids = data['document_offsets'], data['document_ids']
        assert len(rows) == len(ids)
        assert all(row['id'] == str(id_) for row, id_ in zip(rows, ids))
        return [(scores[offsets[i]:offsets[i+1]].astype(np.float64),
                 labels[offsets[i]:offsets[i+1]].astype(np.int8))
                for i in range(len(rows))]


def thresholds():
    qwen = json.loads((RUNS / MODELS[0][1] / 'v5_llmtrace_heldout.json').read_text())['threshold']
    pure = [qwen]
    span = [qwen]
    for name in ('roberta', 'llama'):
        pure.append(json.loads((RUNS / f'open_pangram_editlens_{name}_v5' / 'summary.json').read_text())['threshold'])
        span.append(json.loads((RUNS / f'open_pangram_editlens_{name}_span_v5' / 'summary.json').read_text())['threshold'])
    return pure, span


def load_pure(pure_thresholds, span_thresholds):
    output = {}
    for dataset, relative, qwen_file, baseline_file in PURE:
        source_rows = read_rows(relative)
        allowed = ('human',) if dataset == 'AITDNA human controls' else ('human', 'ai')
        rows = [row for row in source_rows if row['kind'] in allowed]
        qwen = qwen_tokens(qwen_file, source_rows)
        qwen_scores = [float(scores.max()) for row, (scores, _) in zip(source_rows, qwen)
                       if row['kind'] in allowed]
        models = [{'name': MODELS[0][0], 'color': MODELS[0][2],
                   'score': np.array(qwen_scores), 'threshold': pure_thresholds[0]}]
        for index, model in enumerate(('roberta', 'llama'), start=1):
            if dataset == 'AITDNA human controls':
                all_scores = span_tokens(index, baseline_file, source_rows)
                scores = [float(score.max()) for row,(score,_) in zip(source_rows,all_scores)
                          if row['kind'] in allowed]
                threshold = span_thresholds[index]
            else:
                path = RUNS / f'open_pangram_editlens_{model}_v5' / f'{baseline_file}.jsonl'
                data = [json.loads(line) for line in path.open()]
                assert len(data) == len(rows)
                assert all(a['id'] == b['id'] and a['kind'] == b['kind']
                           for a, b in zip(rows, data))
                scores = [r['score'] for r in data]
                threshold = pure_thresholds[index]
            models.append({'name': MODELS[index][0], 'color': MODELS[index][2],
                           'score': np.array(scores), 'threshold': threshold})
        for model in models:
            label = np.array([row['kind'] == 'ai' for row in rows])
            pred = model['score'] >= model['threshold']
            h, a = int((~label).sum()), int(label.sum())
            fp, tp = int((pred & ~label).sum()), int((pred & label).sum())
            model.update({'human': h, 'ai': a, 'fp': fp, 'tp': tp,
                          'accuracy': (tp + h - fp) / (h + a),
                          'human_specificity': (h-fp)/h if h else None,
                          'ai_recall': tp/a if a else None,
                          'auroc': roc_auc_score(label, model['score']) if h and a else None})
        output[dataset] = {'rows': rows, 'models': models}
    return output


def run_lengths(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    delta = np.diff(padded)
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)))


def mixed_stats(rows, token_rows, threshold):
    selected = [(row, scores, labels) for row, (scores, labels)
                in zip(rows, token_rows) if row['kind'] == 'mixed']
    scores = np.concatenate([x[1] for x in selected])
    labels = np.concatenate([x[2] for x in selected])
    assert set(np.unique(labels)) <= {0, 1}
    pred = scores >= threshold
    ai = labels == 1
    tp = int((pred & ai).sum()); fn = int((~pred & ai).sum())
    fp = int((pred & ~ai).sum()); tn = int((~pred & ~ai).sum())
    spans = []
    detected_docs = 0
    for _, doc_scores, doc_labels in selected:
        doc_pred = doc_scores >= threshold
        doc_ai = doc_labels == 1
        detected_docs += bool((doc_pred & doc_ai).any())
        for start, end in run_lengths(doc_ai):
            spans.append(float(np.mean(doc_pred[start:end])))
    return {'docs': len(selected), 'ai_tokens': tp+fn, 'human_tokens': tn+fp,
            'ai_recall': tp/(tp+fn), 'human_fpr': fp/(tn+fp),
            'token_accuracy': (tp+tn)/(tp+tn+fp+fn),
            'balanced_accuracy': .5*(tp/(tp+fn)+tn/(tn+fp)),
            'precision': tp/(tp+fp) if tp+fp else 0,
            'f1': 2*tp/(2*tp+fp+fn) if tp else 0,
            'auroc': roc_auc_score(ai, scores),
            'mixed_docs_with_correct_ai_highlight': detected_docs,
            'ai_spans': len(spans),
            'span_recall_half_covered': float(np.mean(np.array(spans) >= .5)) if spans else None,
            'score': scores, 'label': ai}


def load_mixed(span_thresholds):
    output = {}
    for dataset, relative, qwen_file, baseline_file in MIXED:
        rows = read_rows(relative)
        qwen = qwen_tokens(qwen_file, rows)
        models = []
        for index in range(3):
            token_rows = qwen if index == 0 else span_tokens(index, baseline_file, rows)
            assert all(np.array_equal(a[1], b[1]) for a, b in zip(qwen, token_rows))
            stats = mixed_stats(rows, token_rows, span_thresholds[index])
            stats.update({'name': MODELS[index][0], 'color': MODELS[index][2],
                          'threshold': span_thresholds[index], 'tokens': token_rows})
            models.append(stats)
        output[dataset] = {'rows': rows, 'models': models}
    return output


def pct(value, digits=1):
    return '—' if value is None else f'{100*value:.{digits}f}%'


def write_markdown(pure, mixed):
    lines = ['# Model evaluation dashboard: Qwen 20k versus Open Pangram', '',
             '**Read the source rows before the averages.** The LLMTrace test is disjoint from training by text and group ID, but 15,036 of our 20,000 training documents came from other LLMTrace records. Synthetic v4 is a development validation set. External articles, AITDNA, CoAuthor, and the locked human test probe different sources.', '',
             'Document decision: our Qwen model is positive if any token crosses its frozen threshold; EditLens uses its mean native-window score on the main pure-document sets. For AITDNA human controls, all models use the span-window decoder and its document-any threshold. EditLens has no native token head, so mixed-document token scores broadcast each EditLens window prediction across its source tokens. Every threshold was calibrated on the same separate 1,120 pure-human documents to allow at most 5% document-any false alarms. This calibration set has social Q&A, professional finance, and creative writing, with no published nonfiction articles.', '',
             '## Source-balanced overview', '',
             'These are unweighted averages of the named source rows below, so the large locked human set cannot conceal a failure on external articles. They describe this test collection, not deployment prevalence.', '',
             '| Model | Mean pure-human accuracy across 5 sets | Mean pure-AI accuracy across 3 sets | Mean mixed-token balanced accuracy across 4 sets |',
             '| --- | ---: | ---: | ---: |']
    for index,(name,_,_) in enumerate(MODELS):
        human=np.mean([bundle['models'][index]['human_specificity'] for bundle in pure.values()])
        ai=np.mean([bundle['models'][index]['ai_recall'] for bundle in pure.values()
                    if bundle['models'][index]['ai_recall'] is not None])
        mixed_bal=np.mean([bundle['models'][index]['balanced_accuracy'] for bundle in mixed.values()])
        lines.append(f'| {name} | {pct(human)} | {pct(ai)} | {pct(mixed_bal)} |')
    lines += ['', '## 1. Fully human and fully AI documents', '',
             'Human-only accuracy is **specificity** (`1 − FPR`); AI-only accuracy is **recall** (`1 − FNR`). “FPR on AI-only text” is undefined because false positives require human examples. Combined accuracy is shown only where both classes exist.', '',
             '| Dataset | Model | Human correct / total | Human FPR | AI correct / total | AI FNR | Combined accuracy | AUROC |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for dataset, bundle in pure.items():
        for model in bundle['models']:
            h,a,fp,tp = (model[k] for k in ('human','ai','fp','tp'))
            ai_correct = f'{tp}/{a}' if a else '—'
            ai_fnr = pct(1-tp/a) if a else '—'
            accuracy = pct(model['accuracy']) if a else '—'
            auc = f"{model['auroc']:.4f}" if a else '—'
            lines.append(f"| {dataset} | {model['name']} | {h-fp}/{h} | {pct(fp/h)} | "
                         f"{ai_correct} | {ai_fnr} | {accuracy} | {auc} |")
    lines += ['', '## 2. Mixed documents: token and span localization', '',
              'These rows contain both human and AI text. “Mixed documents found” requires at least one correctly highlighted AI token. Human FPR is the fraction of human **tokens within mixed documents** falsely highlighted. Token balanced accuracy averages AI-token recall and human-token specificity; it prevents the large human portions from hiding missed AI spans.', '',
              '| Dataset | Model | Mixed documents found | Token balanced accuracy | AI-token recall | Human-token FPR | Token F1 | Token AUROC | AI spans ≥50% covered |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for dataset, bundle in mixed.items():
        for model in bundle['models']:
            lines.append(f"| {dataset} | {model['name']} | {model['mixed_docs_with_correct_ai_highlight']}/{model['docs']} | "
                         f"{pct(model['balanced_accuracy'])} | {pct(model['ai_recall'])} | "
                         f"{pct(model['human_fpr'],2)} | {model['f1']:.3f} | {model['auroc']:.3f} | "
                         f"{pct(model['span_recall_half_covered'])} ({model['ai_spans']} spans) |")
    lines += ['', 'A model that predicts every token as human can have high raw token accuracy on human-heavy documents. CoAuthor exposes that failure: all three models have 0% AI-token recall at the frozen thresholds. EditLens window spans are deliberately coarse, so compare their localization with that limitation in mind.', '',
              '## 3. Category breakdown: LLMTrace pure documents', '',
              '| Domain | Model | Human FPR | AI recall |', '| --- | --- | ---: | ---: |']
    ll = pure['LLMTrace test']
    for domain in sorted({row['domain'] for row in ll['rows']}):
        ix = [i for i,row in enumerate(ll['rows']) if row['domain'] == domain]
        labels = np.array([ll['rows'][i]['kind'] == 'ai' for i in ix])
        for model in ll['models']:
            pred = model['score'][ix] >= model['threshold']
            h,a = (~labels).sum(),labels.sum()
            lines.append(f"| {domain} ({len(ix)}) | {model['name']} | "
                         f"{pct(float(np.mean(pred[~labels])) if h else None,2)} | "
                         f"{pct(float(np.mean(pred[labels])) if a else None)} |")
    lines += ['', '## 4. Category breakdown: LLMTrace mixed documents', '',
              '| Domain | Model | AI-token recall | Human-token FPR | Token balanced accuracy |',
              '| --- | --- | ---: | ---: | ---: |']
    llm = mixed['LLMTrace test']
    for domain in sorted({row['domain'] for row in llm['rows'] if row['kind'] == 'mixed'}):
        ix = [i for i,row in enumerate(llm['rows']) if row['kind'] == 'mixed' and row['domain'] == domain]
        for model in llm['models']:
            selected = [(llm['rows'][i], model['tokens'][i]) for i in ix]
            stat = mixed_stats([row for row,_ in selected], [tokens for _,tokens in selected], model['threshold'])
            lines.append(f"| {domain} ({len(ix)}) | {model['name']} | {pct(stat['ai_recall'])} | "
                         f"{pct(stat['human_fpr'],2)} | {pct(stat['balanced_accuracy'])} |")
    lines += ['', '## 5. Real collaboration categories', '',
              '| Dataset and category | Model | AI-token recall | Human-token FPR | Token balanced accuracy |',
              '| --- | --- | ---: | ---: | ---: |']
    for dataset in ('AITDNA collaboration','CoAuthor collaboration'):
        bundle=mixed[dataset]
        for domain in sorted({row['domain'] for row in bundle['rows'] if row['kind']=='mixed'}):
            ix=[i for i,row in enumerate(bundle['rows']) if row['kind']=='mixed' and row['domain']==domain]
            for model in bundle['models']:
                selected=[(bundle['rows'][i],model['tokens'][i]) for i in ix]
                stat=mixed_stats([row for row,_ in selected],[tokens for _,tokens in selected],model['threshold'])
                lines.append(f"| {dataset}: {domain} ({len(ix)}) | {model['name']} | "
                             f"{pct(stat['ai_recall'])} | {pct(stat['human_fpr'],2)} | "
                             f"{pct(stat['balanced_accuracy'])} |")
    lines += ['', '## 6. External article categories', '',
              'The [separate external article report](external_articles_v5.md) includes generator and publication charts. The same breakdown is tabulated here.', '',
              '| AI generator | Our Qwen 20k | EditLens RoBERTa | EditLens Llama |',
              '| --- | ---: | ---: | ---: |']
    external=pure['External articles'];rows=external['rows']
    for generator in sorted({row['generator'] for row in rows if row['kind']=='ai'}):
        ix=[i for i,row in enumerate(rows) if row['kind']=='ai' and row['generator']==generator]
        cells=[f"{sum(model['score'][ix]>=model['threshold'])}/{len(ix)}" for model in external['models']]
        lines.append(f"| {generator} | {' | '.join(cells)} |")
    lines += ['', '| Human publication | Our Qwen FPR | EditLens RoBERTa FPR | EditLens Llama FPR |',
              '| --- | ---: | ---: | ---: |']
    for publication in sorted({row['publication'] for row in rows if row['kind']=='human'}):
        ix=[i for i,row in enumerate(rows) if row['kind']=='human' and row['publication']==publication]
        cells=[f"{sum(model['score'][ix]>=model['threshold'])}/{len(ix)}" for model in external['models']]
        lines.append(f"| {publication} | {' | '.join(cells)} |")
    lines += ['', 'Our Qwen model flags 74/150 attributed-human articles, versus 1/150 for EditLens RoBERTa and 7/150 for EditLens Llama. Its human-token FPR is 18.4%; median highlighted share among falsely flagged human articles is 31.7%.', '',
              '## What failed', '',
              'Our 20k mixture contains 15,036 LLMTrace documents (75.2%). It has only 20 pure-human `editlens:news` documents and 28 pure-human `mage:xsum` documents; 548 LLMTrace human documents carry the `news` domain label. The 1,120-document human calibration set has no published nonfiction articles. External article human FPR remains about 50% across 500–1,000-word bands, while the similarly long calibration documents are near the intended 5% rate. Length and the “any token” document rule contribute, but cannot explain the 18.4% human-token FPR or near-whole-article false highlights. The evidence supports a source shift and inadequate human coverage/calibration; the specific learned shortcut is not yet identified.', '',
              'Next evaluation gate: use an external, source-balanced calibration set assembled without these 300 stress articles; add separate human nonfiction to training, preserve this article set as a holdout, and report these source rows and mixed-span metrics for every future checkpoint. Also retain CoAuthor as a short-insertion challenge.', '',
              'Sources: [Human Detectors](https://github.com/jenna-russell/human_detectors), [EditLens RoBERTa](https://huggingface.co/pangram/editlens_roberta-large), [EditLens Llama](https://huggingface.co/pangram/editlens_Llama-3.2-3B).', '']
    (OUT / 'evaluation_dashboard_v5.md').write_text('\n'.join(lines))


def draw_bars(ax, values, names, colors, title, xlabel, limit=105, digits=1):
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=.62)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    for i,v in enumerate(values):
        ax.text(v + max(1,limit*.015), i, f'{v:.{digits}f}%', va='center', fontsize=8)


def render(pure, mixed):
    OUT.mkdir(exist_ok=True)
    pages = []
    names = [m[0] for m in MODELS]
    colors = [m[2] for m in MODELS]
    selected = ['LLMTrace test', 'Synthetic v4 validation', 'External articles']
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.2))
    for i,dataset in enumerate(selected):
        bundle = pure[dataset]['models']
        draw_bars(axes[i,0], [100*m['ai_recall'] for m in bundle], names, colors,
                  f'{dataset}: fully AI', 'AI documents detected (%)')
        fp = [100*m['fp']/m['human'] for m in bundle]
        draw_bars(axes[i,1], fp, names, colors,
                  f'{dataset}: fully human', 'Human false alarms (%)', max(10,max(fp)+13))
    fig.suptitle('Pure documents: source shift changes the ranking', fontsize=15)
    fig.text(.5,.012,'Thresholds fixed on separate human calibration. LLMTrace is an in-corpus holdout for our model; external articles are a different source.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.045,1,.95),h_pad=1.3,w_pad=2)
    pages.append(('pure',fig))

    fig, axes = plt.subplots(4, 2, figsize=(11, 11.8))
    for i,(dataset,bundle) in enumerate(mixed.items()):
        models = bundle['models']
        draw_bars(axes[i,0], [100*m['ai_recall'] for m in models], names, colors,
                  f'{dataset}: AI-token recall','AI tokens found (%)')
        fpr = [100*m['human_fpr'] for m in models]
        draw_bars(axes[i,1], fpr,names,colors,
                  f'{dataset}: human-token FPR','Human tokens falsely flagged (%)',max(10,max(fpr)+15),digits=2)
    fig.suptitle('Mixed documents: token-level performance',fontsize=15)
    fig.text(.5,.009,'EditLens predictions are broadcast window scores; their spans are coarse. LLMTrace is an in-corpus holdout for our model.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.04,1,.965),h_pad=1.2,w_pad=2)
    pages.append(('mixed',fig))

    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,dataset in zip(axes,selected):
        bundle=pure[dataset]
        labels=np.array([row['kind']=='ai' for row in bundle['rows']])
        for model in bundle['models']:
            fpr,tpr,_=roc_curve(labels,model['score'])
            ax.plot(100*fpr,100*tpr,color=model['color'],label=f"{model['name']} ({model['auroc']:.3f})")
        ax.plot([0,100],[0,100],':',color='#999999')
        ax.set(xlim=(0,100),ylim=(0,100),title=dataset,xlabel='Human FPR (%)',ylabel='AI recall (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Document ROC curves (AUROC in legend)',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93))
    pages.append(('document_roc',fig))

    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for ax,(dataset,bundle) in zip(axes.ravel(),mixed.items()):
        for model in bundle['models']:
            fpr,tpr,_=roc_curve(model['label'],model['score'])
            ax.plot(100*fpr,100*tpr,color=model['color'],label=f"{model['name']} ({model['auroc']:.3f})")
        ax.plot([0,100],[0,100],':',color='#999999')
        ax.set(xlim=(0,100),ylim=(0,100),title=dataset,xlabel='Human-token FPR (%)',ylabel='AI-token recall (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Mixed-document token ROC curves (AUROC in legend)',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.945))
    pages.append(('token_roc',fig))
    with PdfPages(OUT / 'evaluation_dashboard_v5.pdf') as pdf:
        for name,fig in pages:
            pdf.savefig(fig)
            fig.savefig(OUT / f'evaluation_dashboard_v5_{name}.png',dpi=170)
            plt.close(fig)


if __name__ == '__main__':
    pure_thresholds, span_thresholds = thresholds()
    pure = load_pure(pure_thresholds, span_thresholds)
    mixed = load_mixed(span_thresholds)
    write_markdown(pure,mixed)
    render(pure,mixed)
