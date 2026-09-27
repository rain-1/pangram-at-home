# Second Hugging Face and named-writer data intake

Collected 27 September 2026 for the noncommercial, open-research span
attribution project. Text and raw downloads are outside Git under
`/mnt/f/pangram-at-home/data/candidate_span_sources/round2/`. These are
**screened candidates**, not additions to a training run. Existing locked
evaluations remain protected.

## Best additions

| Source | Screened records | Value for the detector | Provenance and limits |
| --- | ---: | --- | --- |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 4,118 longer human responses; 625 creative writing, 688 brainstorming, 1,574 general QA, 1,231 open QA | Distinct modern human prose written for open-ended prompts. Prioritize the 1,313 creative/brainstorming rows; review QA prompts for single-answer tasks before intake. | Databricks says employees wrote responses and were instructed **not to use generative AI**. This is unusually direct provenance, although compliance was not independently observed. CC BY-SA 3.0 with attribution/share-alike. Context text is excluded from our response-only records. |
| [GRADTEX](https://huggingface.co/datasets/elisabeth-pl-pl/GRADTEX) | 5,374 mixed completions plus 5,374 paired human seeds | Adds AI-first and AI-last transitions in reviews, social text, fiction, science, and other domains. Train file has Gemma 4 and Mistral Small 3.2 outputs. | **Inferred** boundary from exact retained context, not published gold spans. 3,062 weak-context rows were rejected. Its seeds derive from MAGE, already in our project, so this is augmentation rather than an independent benchmark. CC BY 4.0 derivative, with MAGE source rights carried through. |
| [Travis ShortStory](https://huggingface.co/datasets/Travis-ML/ShortStory-SFT-jsonl) | 710 historical stories by 26 named authors | Clean, substantial fiction, 300–2,500 words per source card; useful human fiction coverage or author-held-out stress. | Project Gutenberg transcriptions of works asserted public domain. Individual work/edition rights still need review for a release. Prompts are model-generated but excluded; only historical story text is labeled human. Nine duplicate stories were removed. |
| [Stanford Human Preferences](https://huggingface.co/datasets/stanfordnlp/SHP) | 48,483 unique comments from 13,461 posts across AskCulinary, AskHistorians, AskPhilosophy, AskScience, and AskScienceFiction | Substantial pre-2023 social/Q&A prose with comment timestamps and topic variety; useful source-specific stress and controlled training sample. | Reddit user submissions, not verified unaided writing. The dataset publication provides the data but does not establish item-level user-content reuse rights. Filtered to at least 35 words and comment dates before 2023; do not label these as strict gold human text. |
| [Human–AI Story Contrastive v4](https://huggingface.co/datasets/schonsense/human_ai_story_contrastive_v4) | 1,437 matched prompt groups: 1,437 human references and 2,874 GPT-5.6 Sol stories | Valuable *conditional* creative-writing challenge; blind generations and human-conditioned rewrites can be reported separately. | The card gives no dataset license or story-level dates/author permission. It does not supply mixed-span labels. Keep out of strict human training/evaluation until the source rights and provenance are resolved. |

The 5,374 GRADTEX completions come from 2,765 `complete_ending` and 2,609
`complete_beginning` rows. Its original train Parquet had 79,065 document
labels, but the other edit scenarios cannot be treated as gold token labels.
For retained completions, the unchanged prefix or suffix exactly matches at
least 200 characters and at least half the final text; the rest is at least 80
characters. This is conservative provenance reconstruction based on the
dataset's concatenation recipe. It does not make the token-polish and style
rewrite rows span-labeled.

## Additional named human writers

The earlier author-attribution collection already has [Gwern, Paul Graham,
Scott Alexander, and Eliezer Yudkowsky](attribution-data-v1.md). Their names
and publication dates do not by themselves prove that the *current served
version* is AI-free; Scott's source reports later edits, and several Graham
essays lack pre-2023 snapshots.

This pass adds two writers with substantially earlier source dates:

| Writer | Acquired essays | Source and human-origin evidence | Reuse status |
| --- | ---: | --- | --- |
| [Cory Doctorow](https://craphound.com/context/download/) | 56 essays, from *Content* (2008) and *Context* (2011) | Official author-published collections, with the introductions/forewords, interview, blockquotes, and book boilerplate excluded. Pre-LLM book publication is strong human-origin evidence; current server files are not archival captures from publication day. | Both books explicitly carry CC BY-NC-SA terms; attribution and share-alike matter. Quoted third-party text may remain in paragraphs. |
| [Aaron Swartz](http://www.aaronsw.com/weblog/fullarchive) | 94 essays dated 2009–2012 | Official first-party archive, bounded to 120 dated article pages; he died in 2013, well before modern LLM writing tools. This supports the original authorship label, though the current served copy could contain later edits. | No blanket site/article license verified. Keep privately on the external disk and do not redistribute these copies as an open corpus without rights review. |

These 150 records are whole essays with author, date or volume, source URL,
source hash, and work ID. They are useful for a named-author probe and
human-prose false-positive checks. Split by **whole work and author** before
windowing. For the detector, do not let one prolific writer or old era dominate
the human class. A writer-classification result is not evidence of human/AI
detection accuracy.

## Integrity and next experiment

The intake auditor checks full character-span coverage, IDs, group IDs,
normalized exact text, and sampled 24-word overlap with current training,
validation, and protected evaluations. The screened Dolly, GRADTEX, SHP,
Doctorow, and Swartz records have zero flagged matches to those references.
GRADTEX required dropping 349 **whole source groups** with overlap flags;
Dolly dropped two, SHP one. A further two GRADTEX groups overlapped the
previously acquired OpAI abstract intake, and 34 SHP posts overlapped the
previously acquired GEN train split, so those entire groups were excluded.
Duplicate source groups or identical rows were also removed. The
story-contrastive set had no sampled matches to the
screened GRADTEX or Travis sets. A zero sampled-shingle hit is a useful screen,
not proof of independence; final mixes still need source-family grouping and
near-duplicate review.

The next bounded ablation should add **Dolly creative writing/brainstorming**
as stronger human examples and **GRADTEX completions** as limited mixed-span
augmentation, keeping the v10 architecture/evaluation fixed. Cap both by
source group and supervised-token share. Use Travis fiction as a distinct
human stress slice or an author-disjoint training allocation. Try a small SHP
sample only with its rights/provenance status made explicit and matched AI
writing in the same question-answer styles. Preserve the two Pangram
baselines and report human-document FPR, AI-document recall, mixed token
recall, boundary quality, and per-source performance.

The raw files, normalization manifests, screened files, and audit outputs are
in the external `round2/` directory. Rebuild the derived candidates with
`scripts/download_hf_candidate_round2.py` (pinned Hugging Face revisions),
`scripts/prepare_hf_candidate_round2.py`,
`scripts/prepare_doctorow_author_candidate.py`,
`scripts/prepare_aaron_swartz_author_candidate.py`, and
`scripts/screen_hf_candidate_round2.py`. The author collectors cache their
first-party source files and record hashes; reruns should use a fresh versioned
directory if the source changes.
