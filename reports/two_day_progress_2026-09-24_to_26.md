---
title: "Pangram at Home: two-day research report"
subtitle: "Data, baselines, tuning, and span detection"
date: "26 September 2026"
geometry: margin=22mm
colorlinks: true
linkcolor: blue
---

# Executive summary

Between the evening of **24 September and 26 September 2026**, we turned an
initial paper-heavy AI-text detector into a broader research pipeline. We
built balanced data pyramids, trained and compared classical and neural
baselines, tuned a Qwen3-1.7B passage classifier, investigated Repeat2, and
implemented a binary token-level span model that can highlight regions of a
long document. We also collected new human-fiction, real-collaboration, and
attribution datasets for future work.

The main result is **promising but conditional**. Our selected Vast passage
checkpoint catches **93.4%** of AI passages on the 1,000-row diverse test at
**2.2%** human false positives, and **85.8%** on the independent-source RAID
test at **0.8%** human false positives. It catches only **37.0%** of the
paraphrased-AI challenge set. The current token model improves substantially
on constructed mixed documents, but realistic collaborations expose a major
calibration and task-transfer gap: on AITDNA, it catches **88.4%** of AI tokens
while falsely highlighting **18.0%** of human tokens *inside mixed documents*.
CoAuthor tests mostly short sentence suggestions outside our primary target;
its low recall is reported separately.

These figures come from different tasks and thresholds. Passage results are
one decision per 512-token input. Span results are aggregated token labels
over overlapping windows. They should not be put on one leaderboard.

# 1. What we built

## Data and evaluation

Our first frozen pyramids included a **37,568-row general EditLens train tier**,
a **1,542-row PMC/ACL paper train tier**, and a **4,404-row mixed train tier**.
Each had nested smaller train, validation, and test tiers balanced 50/50 human
and AI. We checked exact-text, work, and split separation. The paper examples
were chiefly **abstracts**; full PMC articles were downloaded for human-only
body audits, not balanced full-paper training. The early paper generators were
two small local models given titles, which made that first problem narrower
than our intended deployment task. [Initial data and baselines](first-data-and-baselines.md).

We then rebuilt the active passage mixture as a **10,000-row balanced train
set**, with nested 200, 1,000, and 4,000-row train tiers, 800 validation rows,
and 1,000 test rows. Its requested shares were **35% scientific/paper, 20%
reference/education, 20% creative, 10% social/Q&A, 10% reviews, and 5% news**.
Sixteen source/domain entries contributed. The mix included MAGE, EditLens,
and locally paired PMC/ACL papers. Human-only checks covered classic fiction,
student essays, Federal Reserve prose, Writers Stack Exchange, and PMC body
text. RAID supplied a separate source-family test across eight domains and
eleven generators. [Diverse data manifest and limits](diverse-data-v1.md).

We learned that **balanced labels and disjoint row IDs are necessary but not
sufficient**. The diverse train and test share several source families;
source-specific style, prompts, and generation pipelines can make a ROC curve
look sharper than real transfer warrants. A small number of high-similarity
MAGE contexts remained even after split checks. We therefore kept RAID,
paraphrase challenges, human-only audits, and per-source results visible.

## Baselines and passage model

We trained character and word TF-IDF logistic classifiers and a MiniLM
embedding classifier, and evaluated local EditLens RoBERTa and Llama
checkpoints. We also tested the Load Bearing vocabulary cluster as an
exploratory transfer baseline; it is not a purpose-trained authorship
detector. All operating thresholds came from the same diverse validation set
for a nominal 2% human false-positive target. [Baseline table](diverse-baseline-results-v1.md),
[readable comparison PDF](readable_model_comparison_v1.pdf).

| Model, averaged equally over diverse test and RAID | AI recall | Human FPR |
| --- | ---: | ---: |
| Our first diverse Qwen3 passage model | **83.0%** | 1.3% |
| Character TF-IDF | 35.8% | 1.3% |
| Word TF-IDF | 32.9% | 0.8% |
| MiniLM + logistic | 8.5% | 1.1% |
| EditLens RoBERTa | 44.3% | 2.2% |
| EditLens Llama | 58.0% | 0.8% |
| Load Bearing vocabulary probe | 4.1% | 5.0% |

The average is a compact summary of **two fixed test sets**, not an estimate
of performance on all writing. The first paper-heavy adapter made this point
especially clearly: it achieved only **0.633 AUROC and 16.3% AI recall** on
the broader 4,000-row MAGE test, despite much stronger in-family results.

# 2. Tuning and compute experiments

On Vast, we ran **24 Ray Tune trials** over learning rate, effective batch,
LoRA rank, and dropout. Seven reached the full budget; ASHA pruned 17. Ray
reported no trial failures. Some pruned W&B runs appeared as “crashed” because
of the old shutdown path; that logging path was fixed. The selected setting
was **LR 7.61e-5, effective batch 8, rank 32, alpha 64, dropout 0.0688**,
BF16 LoRA targeting attention and feed-forward projections, with 512 source
tokens. Its full-budget validation AUROC was **0.9946**, partial AUROC in the
low-FPR region **0.9681**, and AI recall at the validation 2% FPR point
**93.5%**. Each combination ran once on the same 800 validation passages, so
this selected score is a development result, not an independent gain estimate.
[Tuning report](hpo_results_v3.md), [W&B project](https://wandb.ai/eac-adsf/pangram-at-home).

Nine data-mixture ablations then used that setting with equal example
budgets. Removing paper data lowered validation partial AUROC by **4.88
percentage points**; removing creative writing lowered it by **4.53 points**.
Changing paper share from 35% to 20% or 50% moved partial AUROC by less than
0.2 points. This supports keeping both scientific and creative writing, but
one seed and one repeatedly examined validation set do not establish a final
optimal percentage. [Mixture ablations](ablation_results_v3.md).

On the local RTX 4080, we compared a single-copy classifier with a Repeat2
classifier. We initially changed microbatch from Vast's **2 × accumulation 4**
to **1 × accumulation 8** without a memory measurement. Effective batch stayed
at eight, but this prevented exact reproduction of the Vast winner. A later
microbatch-2 backward/optimizer probe fit for both single-copy and Repeat2
inputs, so that change was not justified by measured memory need. The local
models also used BF16, while an earlier diverse QLoRA run used NF4; these runs
should not be described as differing in only one variable.

At frozen thresholds, the selected Vast passage checkpoint was at least as
convincing as the local controls on broad transfer: **93.4% AI recall / 2.2%
human FPR** on diverse test, **85.8% / 0.8%** on RAID, and **37.0% recall**
on paraphrased AI. The local passage Repeat2 result had **93.8% / 5.2%** on
diverse test and **28.7%** paraphrase recall. Thus a slight development
validation improvement did not translate into the desired FPR/recall tradeoff.
[Wider controlled comparison](comparative_models_v1.md).

For long scientific documents, a source-disjoint human-only audit of **100
PMC papers and 685,096 source tokens** found the Vast passage model falsely
highlighted **0.8% of tokens**, with at least one false-highlight window in
**9/100 papers**. A low token FPR can coexist with a consequential
document-level alert rate when many windows are tested.

# 3. Moving from passages to spans

We confirmed that the original passage classifier produced **one label per
window**, so sliding it over a document only gave a coarse heatmap. We added
a binary human/AI token head, exact character-offset labels, overlapping
**512-source-token windows with stride 256**, and a span inference script.
For Repeat2, the input is two copies of a source window and **the entire first
copy's token loss is masked**; predictions and supervision use the second
copy. We have **not** trained an AI-assisted third class, mixed-document
auxiliary head, or humanizer head. [Architecture and scope](span-detection-roadmap-v1.md).

The first token pilot trained on **2,400 synthetic mixed documents** and used
a separate 500-document source-group-disjoint confirmation set. At a
document-aware development threshold, Repeat2 reached **84.2% AI-token
recall** on confirmation, compared with **50.8%** for the single-copy token
model; **7.1%** versus **6.0%** of wholly human confirmation documents received
some false highlight. This established that the Repeat2 token objective was
worth pursuing. It did **not** establish realistic editing performance:
synthetic joins can expose source and formatting cues, and the confirmation
parent test had been examined previously. [Token pilot report](token_span_pilot_v3.md).

We next built **5,000 training documents** from the broader source pool,
yielding **7,600 windows** and roughly balanced labeled human and AI tokens.
Documents included intact passages, same-class joins, and mixed joins, with
short and long variants. Many long examples are unrelated excerpts joined
together; the 5,000 composites do not represent 5,000 independently authored
works. [Span data v4](span-training-data-v4.md).

The Vast v4 Repeat2 run completed, its adapter and scores were verified on
the external drive, and its instance was destroyed. Estimated rental cost
for **that run** was **$0.508**; this is not a total project-spend figure.
The v4 model reached **73.6% AI-token recall** on its constructed validation,
versus **18.0%** for the previous v3 checkpoint at separately frozen
human-document-safe thresholds. On a **3,579-document human-only** test,
**0.7%** of documents had any false highlight. These are different operating
points from the first token pilot and should not be numerically compared
across tables without their threshold definitions. [V4 comparison](span_v4_comparison.md),
[W&B run](https://wandb.ai/eac-adsf/pangram-at-home/runs/stim7kc8).

# 4. What real collaboration taught us

We evaluated v3 and v4 on **AITDNA**, which records surviving human and model
text from real writing studies. On v4's frozen threshold, its **258 mixed
documents** showed **88.4% AI-token recall**, **18.0% human-token FPR within
mixed documents**, and **0.918 token AUROC**. Of **103 wholly human final
documents**, one received a false highlight. V3 had lower mixed-document
AI recall (**39.1%**) and lower mixed-document human FPR (**2.5%**), while
whole-set AUROC was nearly unchanged (**0.958** for v3 versus **0.956** for
v4). The v4 gain is therefore partly a shift along a recall/FPR tradeoff,
not an unambiguous improvement in ranking. [AITDNA audit](span_external_sources_v5.md).

The CoAuthor subset gave **3.3% AI-token recall** and **0.4% human-token FPR**
on **119 mixed documents** at the same frozen v4 threshold, with AUROC
**0.676**. Only **51.3%** of its source tokens have usable labels in our
conservative reconstruction. At a retrospective 5% FPR chosen *within that
benchmark*, recall is still only **21.7%**, so threshold conservatism alone
cannot explain the poor ranking. Yet CoAuthor's GPT-3 interface offered short
sentence suggestions: only **4 of 407** surviving AI spans in our subset
reach 320 characters. We now treat it as a **separate short-completion stress
test**, not the headline benchmark for substantial open-ended AI prose.
[Side-by-side report and ROC](real_collaboration_v4.md),
[CoAuthor paper](https://cs.stanford.edu/~minalee/pdf/chi2022-coauthor.pdf).

The crucial lesson is to specify the target **before** averaging results.
Token recall weights long AI spans heavily: **81.5% of AITDNA AI characters**
lie in spans at least 320 characters long, versus **7.4% for CoAuthor**. We
need separate scores for substantial generated passages, local rewrites, and
short suggestions, and separate rates for pure-human documents and human
regions embedded within assisted documents.

# 5. New data acquired, but not yet part of the detector

We normalized three external span sources on the external drive:
**LLMTrace** has **27,754 English train, 5,534 validation, and 7,012 test
documents** across nine domains; **AITDNA** has **362 unique documents from
99 writers** and is reserved for evaluation; **DAMASHA clean** has **96,119
parsed unique candidates**. AITDNA's multiple published views are projections
of the *same* 362 texts, not thousands of independent documents. DAMASHA's
aggregate mixes new records with TriBERT/M4GT and lacks upstream source IDs;
our overlap audit found **735 candidate records** sharing sampled long phrases
with the protected human test. We should not merge it wholesale. Five
LLMTrace topic groups also touch the frozen diverse test and must be excluded
before training. None of these sources entered the completed v4 run.
[External span source audit](span_external_sources_v5.md).

For creative writing, we collected **1,849 candidate records / 4.38 million
words** from official PAN fanfiction benchmarks, a small AO3 mirror pilot,
original fantasy stories, a controlled volunteer story study, and historical
ebooks. These are candidate human-source records, **not** an already balanced
AI/human training split or proof that every page was unassisted. AO3's public
archive alone cannot guarantee human authorship. The source texts remain off
Git; source/version/rights caveats are recorded. [Fiction collection](fiction-data-v1.md).

For later **frozen-head attribution probes**, we gathered **299 essays**
attributed to Gwern, Paul Graham, Scott Alexander, and Eliezer Yudkowsky,
with work-level splits, plus **11,864 AI responses** from two prompt-grouped
generator datasets. No attribution head has been trained, and these texts
have not been added to the detector. The linked SynthPrompts data turned out
to be *prompts* written by one model, not labeled answers from many models;
we kept a small prompt-only pool separately. [Attribution data](attribution-data-v1.md).

# 6. Decisions and next experiments

1. **Keep the task narrow enough to measure:** substantial original prose
   from open-ended generation is the main target. Report short completion,
   proofreading, and humanized/paraphrased text as separate challenges.
2. **Build the next span pyramid from decontaminated LLMTrace groups**, balanced
   across domains, length, and human/AI/mixed content. Keep AITDNA reserved;
   defer DAMASHA until its source lineage can be reconstructed or screened.
3. **Add coherent, realistic mixed writing and independent long AI documents.**
   Synthetic joins remain useful training controls but should not be the
   principal evidence of boundary quality. A prepared 300-article modern
   generator candidate has not yet been scored; its human-byline and rights
   caveats must accompany any result. [Next AI evaluation](span-ai-eval-next.md).
4. **Calibrate the whole-document output**, including window aggregation and
   document-any-highlight rate. Report FPR on both pure-human documents and
   human regions of mixed documents. Review false positives and false
   negatives by source, generator, span length, and paper section.
5. **Confirm improvements on fresh groups**, because the current diverse and
   synthetic confirmation sets have been repeatedly inspected. Avoid
   promoting a development result to a blind-test claim. Once a stable
   baseline is chosen, test hard-negative mining as a separate later study.
6. **Keep attribution optional.** Train probes on frozen representations first;
   only test joint heads if the core detector's FPR and recall remain intact.

All raw text, score caches, and adapters are on `/mnt/f/pangram-at-home`;
Git contains code, hashes, aggregate reports, and small charts. We have
committed and pushed the completed work regularly. The current Vast instance
is closed and no training run is active as of this report.
