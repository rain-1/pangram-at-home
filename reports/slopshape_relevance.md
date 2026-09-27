# SlopShape: relevance to pangram-at-home

Source: [SlopShape v2](https://arxiv.org/html/2609.15369v2) and its
[release README](https://github.com/pulse-energy-eu/slopshape).

## What the study actually shows

The paper pairs 2,250 B2B blog posts from pre-ChatGPT Wayback snapshots across
268 company domains with 11,250 first-shot AI mirror posts from five models. It
splits by company domain, then uses an LLM to score 187 document-structure
features and trains a classifier on them. The structural classifier reaches
98.0% macro-F1 on its company-held-out test, and 98.1% after each AI post is
reworded by its generating model. It also attributes the six author classes
(human plus five AI models) at 79.3% accuracy. Those are impressive scoped
results; the study's word-level ModernBERT and stylometric baselines reach
100.0% macro-F1 on the original unedited corpus.

The most useful qualitative finding is that AI commercial posts often have a
predictable whole-article organization: they announce a payoff and plan early,
then end by restating the point. Structure and word/style features make mostly
different errors in this test. This suggests an optional complementary signal
for long, coherent documents.

## Where it could help us

1. **Better provenance.** Their human material comes from actual pre-2023
   archived snapshots. Our new science-article v9 pool currently has old
   publication dates but contemporary page snapshots; we should locate dated
   archived HTML for at least the locked-test articles before treating them as
   definitive human ground truth.
2. **A new domain.** B2B company blogs are missing from our publication data.
   Collect an independent pre-2023 archive-based B2B pool with company-domain
   splits and matched AI writing tasks. Keep companies disjoint across train,
   calibration and test. Their release includes a sampling ledger, but reuse
   rights must be cleared before using it as data.
3. **A document-level complement.** Independently implement a small set of
   interpretable structure indicators (section count, introductory signposting,
   closing recap, citation/evidence placement) and test whether they explain
   our human publication false positives or add ranking value beyond the Qwen
   detector. Evaluate on science features and independently collected blogs;
   keep the entire EPA pool locked until the experiment is specified.
4. **Rewording stress test.** Compare our detector on original AI features and
   claim-preserving reworded versions. Record false-positive rate on unchanged
   human counterparts at the same threshold. Their robustness result is for
   one specific self-rewording protocol, so test varied rewrites separately.

## Limits for our system

- Their instrument reads whole commercial posts (600–2,500 words) and is
  expensive: the paper reports 148,500 LLM scoring calls for its main corpus.
  It does not directly label AI spans in heterogeneous documents or provide a
  practical 512-token window replacement.
- Mirrors are generated from reverse-engineered briefs, which can omit context
  available to the original human writer. Historical human posts and 2026 AI
  mirrors also confound date with authorship. The authors state both limits.
- Their 98% figure does not establish cross-domain performance on science
  articles, fiction, social media, or real human-AI collaboration. We should
  test any structural signal on our own held-out domains before adopting it.
- The code is PolyForm Noncommercial; the instrument, prompts, and other
  non-code release materials are all-rights-reserved for audit/verification.
  Post-level feature answers and mirror texts are gated under a research
  agreement. We can use the paper to form hypotheses, but should not import
  its code, feature taxonomy, or corpus into this open-source project without
  suitable permission.

**Recommendation:** prioritize archived-snapshot verification and a held-out
B2B blog evaluation. Then run a small, independently implemented structural
feature ablation as an optional document-level second opinion. Continue to
develop the window/span detector for heterogeneous text.
