# Adding rain1's v14 mix and heterogeneous-ai-spans v1.3.0 to woog's MoE training set

Date: 2026-10-06. Read-only analysis. Nothing on the Space was written: base and eval files were copied locally and their checksums verified. The analysis scripts are in `scripts/`. Their intermediate outputs (`work/`: stats, overlap and simulation JSON, plus windowed text) are kept out of git because they contain third-party text. `scripts/build_additions.py` rebuilds the `additions-v1.jsonl.gz` input that `setup_mix.py` reads.

**How the numbers were measured.** Windows are cut exactly as `setup_mix.py` cuts them: consecutive 500-token windows, using the native tokenizer `assets/qwen36-35b-a3b/tokenizer.json`. That file is byte-identical to the 4B tokenizer (sha256 5f9e4d49…). The counts are exact token counts, not a character proxy. Span lengths are in characters.

A span is "clipped" when the window edge cuts it. Clipped pieces are fragments of long spans, not short spans in context, so both clipped and unclipped figures are reported. Base windows are woog's existing ≤510-token crops. Their clipped flag is approximate.

## 1. Composition

| | woog base (stage 2, per epoch) | rain1 v14 train | heterogeneous-ai-spans v1.3.0 train |
|---|---|---|---|
| Rows | 24,000 windows: mirrors 6,000, papers 6,000, gradtex 4,800 (document-only), human 6,000, fullpapers 1,200 | 13,285 rows from 25 `source` values | 3,509 mixed + 3,509 matched controls (same source passage, untouched) |
| Domains | ML/CS papers (paired edits plus full manuscripts); 34-source human pool (fiction, science, reference, reviews, social, web, news, essays, professional); GRADTEX (MAGE-derived seeds) | MAGE (8 domains: cmv, eli5, roct, sci, squad, tldr, wp, xsum, yelp), ACL and PMC papers, Dolly, LLMTrace, ASAP2, PERSUADE, NASA and NOAA science, Writers SE, Common Pile news ×5, GRADTEX, craphound | Gutenberg 1,554, JMLR 2000–2014 1,392, Standard Ebooks 232, WikiText-2 156, Beige Book 173 (4.9%), Hansard 2 (counts are pairs) |
| Generators | Almost all GPT-6 Luna (mirrors, paper edits). Fullpapers also GPT-6(.1) Sol | Modern: LLMTrace 16 (2024–25: gemini-2.5-flash 257, GPT-4.1, o3, Qwen3, …); GRADTEX 2 (gemma-4-31b, mistral-small-3.2). Small or legacy: ACL/PMC/ASAP/science use SmolLM2-1.7B, Qwen2.5-0.5B or Qwen2.5-3B; MAGE ~27 2020–23 models (FLAN-T5 small→xxl, T0, OPT 125m–30b, OPT-IML, GPT-J/NeoX, BLOOM-7B, LLaMA-1, GLM-130B, davinci-002/003, gpt-3.5) | 6 frontier writers: Haiku 4.5 (705), Sonnet 5.5 (621), Opus 5.5 (736, fiction only), Opus 3 (88), GPT-6.1 Sol (677), GPT-6 Luna (682) |
| Human / AI / mixed (documents) | Windows: 37.5% human, 30% pure AI, 12.5% mixed, 20% gradtex document-only | 5,825 human, 5,221 AI, 2,239 mixed (MAGE alone: 2,546 / 3,750 / 1,051) | 50% mixed, 50% pure-human controls |
| Construction | Paired paragraph replacement in real papers; whole-passage mirrors | MAGE/ACL/PMC: unaltered rows plus **synthetic joins of unrelated excerpts** (same-label and mixed short/long joins; v4 recipe). LLMTrace: create/fill_gaps/delete/expand. GRADTEX: preserved-context completion. Science/ASAP: matched human/AI windows (no in-document mixing) | **Coherent in-place replacement**: 2–6-paragraph blocks condensed to a brief, re-expanded and spliced back in, at 55–65% replacement caps |
| Licence flags | Commercial-use-approved pool | NONCOMMERCIAL: PERSUADE 400, craphound 24, Qwen2.5-3B outputs 264 (ASAP 114, science 150). Common Pile CC BY. Dolly CC BY-SA | No NC flags. JMLR "author copyright, paper licence unconfirmed"; WikiText CC-BY-SA version discrepancy |

Beemo and No Robots are **not present** in either candidate set. They appear only in benchmark-v3 as evaluation data. v14 has no EditLens rows in the train parquet.

## 2. Span statistics after windowing (exact Qwen tokens)

Median chars per token is 4.1–5.9 (fiction ~4.2, papers ~5.4, Beige Book 5.8).

| Source (key) | Windows | % human / AI / mixed | AI span chars, all (median; <200 / 200–600 / >600 %) | Unclipped spans (median; % <200) | Spans per mixed window | AI fraction (all; mixed) | Boundary type | Mixed windows with unclipped span <200 |
|---|---:|---|---|---|---:|---|---|---:|
| **Base, 3 epochs (72,000)** | 72,000 | 37.5 / 30.0 / 12.5 | 1,377; 0.1 / 21.0 / 78.9 | 554; 0.0 | 1.00 | 0.34; 0.35 | 100% paragraph | ≈0 |
| base papers | 18,000 | 50 / 0 / 50 | 555; 0.2 / 56.5 / 43.3 | 554; 0.0 | 1.00 | 0.17; 0.35 | paragraph | 0 |
| hetero mixed: JMLR | 2,181 | 18.9 / 11.9 / 69.2 | 1,403; 4.7 / 7.7 / 87.6 | 1,516; 0.0 | 1.00 | 0.53; 0.59 | paragraph (100%) | 0 |
| hetero mixed: Gutenberg | 5,853 | 43.0 / 6.6 / 50.4 | 1,057; 8.2 / 17.8 / 74.0 | 1,323; 0.0 | 1.09 | 0.34; 0.55 | paragraph (97%) | 0 |
| hetero mixed: Standard Ebooks | 1,511 | 58.3 / 4.2 / 37.5 | 1,178; 6.5 / 16.4 / 77.0 | 1,485; 0.0 | 1.05 | 0.24; 0.52 | paragraph | 0 |
| hetero mixed: WikiText | 952 | 63.1 / 0.2 / 36.7 | 1,169; 8.7 / 14.4 / 76.8 | 1,732; 0.0 | 1.05 | 0.20; 0.53 | paragraph | 0 |
| hetero mixed: Beige Book | 534 | 39.5 / 2.8 / 57.7 | 1,211; 9.8 / 13.1 / 77.1 | 1,603; 0.0 | 1.11 | 0.31; 0.48 | paragraph | 0 |
| hetero controls (all) | 10,574 | 100 / 0 / 0 | — | — | — | 0 | — | — |
| v14 LLMTrace | 600 | 32.8 / 23.3 / 43.8 | 143; **59.7** / 31.6 / 8.8 | 143; 59.7 | 1.33 | 0.37; 0.31 | sentence 96%, intra 4% | **219** |
| v14 GRADTEX (spans) | 593 | 10.5 / 5.2 / 84.3 | 363; **19.8** / 57.4 / 22.8 | 351; 20.9 | 1.00 | 0.37; 0.38 | sentence 50%, **intra-sentence 50%** | **98** |
| v14 MAGE (9 sources) | 8,062 | ~38 / ~49 / ~13 | 238–638 by domain; <200 mostly 0–5% (yelp 21%, roct 12%) | — | 2–3 | ~0.55 | joins: paragraph ~65%, sentence ~25%, intra ~10% | 69 |
| v14 ACL / PMC | 1,417 / 981 | 42 / 40 / 18–20 | 920 / 1,042; ~3 / ~12 / ~85 | 956 / 1,097; 0 | 1.2 | 0.50 | paragraph ~63%, **intra-sentence ~27%** (arbitrary cut points) | 0 |
| v14 ASAP2 / science_v9 | 663 / 1,152 | 50 / 50 / 0 | long (whole-window AI) | — | — | 0.5 | none (no mixing) | 0 |
| v14 human-only (Dolly, Common Pile, PERSUADE, Writers SE, craphound) | 2,762 | 100 / 0 / 0 | — | — | — | 0 | — | — |

What the table shows:
- **Heterogeneous-ai-spans does not address the short-span weakness.** None of its unclipped AI spans is under 200 chars; the median is about 1,300–1,700 chars. Every one of its <200-char pieces is a window-edge fragment of a long span. It does address the other weaknesses: 5–6 modern generators instead of one, coherent in-document mixing, and domains outside CS papers (fiction, older ML papers).
- Across all three sources, only **two** supply sentence-scale spans: LLMTrace (219 windows) and v14 GRADTEX (98 windows). The leakage check removes LLMTrace (§3). The proposed mix then gains about 32 short-span windows per epoch. That is a real but negligible increase over the base's ≈0, so **small-edit recall should not be expected to move**. Short in-place edits need new data work. The hetero controls would make a good substrate: 1–3-sentence replacements in Gutenberg and JMLR passages, by the same six writers.

## 3. Redundancy and leakage

Leakage is measured with woog's rule (normalised ≥10-word sentences, identical to `setup_mix.py`). On top of that, rows are flagged as near-duplicates when they share ≥8 normalised 12-word shingles, or when ≥5% shingle coverage includes at least 3 shingles. Protected sets:
- builder: sweep-eval-rows, selection-windows, calibration-windows
- frozen suite: workflow and comparison
- hetero validation and test (the new held-out evaluation)
- benchmark-v3 `examples.parquet` (14,322 rows), checked by text, by group and source keys, and by sentences

Rows that must be dropped (in `leakage-drop-ids.json`, 397 ids, closed over mixed/control pairs):

| Source | Rows dropped | Why |
|---|---:|---|
| hetero Beige Book (mixed / control) | 73 / 73 | 55 docs share ≥10-word sentences with hetero val/test, and 57 with v3 `generate_heterogeneous`. These are boilerplate Beige Book phrasing across districts and reports. The source is excluded anyway |
| hetero WikiText (mixed / control) | 16 / 16 | Sentences shared with v3 `wikitext_detok` (WikiText-2 ⊂ WikiText-103 articles) |
| hetero JMLR (mixed / control) | 1 / 1 | ≥8 shingles shared with hetero test (the same paper's text in another excerpt). 2 more pairs share only "the remainder of the paper is organized as follows" boilerplate and are kept |
| hetero Gutenberg / Standard Ebooks | 0 | 1 pair shares 3 shingles with the `epoch` eval set; kept as boilerplate |
| v14 LLMTrace | 46 | 21 **exact texts** and 42 of 547 topic groups shared with v3 `llmtrace_detection`/`_classification` |
| v14 ASAP2 / PERSUADE | 79 / 77 | Shared sentences with the v3 `persuade` panel (1,500 rows). The frozen-suite ELLIPSE hits (≤5% coverage) are quotes from the prompt source text, not duplicate essays |
| v14 MAGE sci / xsum | 8 / 3 | Real overlap with the ACL papers in the comparison profile (coverage up to 100%), and the BBC footer in `opai` |
| v14 ACL | 3 | Shared sentences with selection/calibration (the builder would drop them too) |
| v14 GRADTEX | 1 | BBC footer shared with `opai` |

Redundancy with the base:
- **GRADTEX.** 86/500 v14 GRADTEX rows have most of their sentences in the base gradtex pool. In addition, 20–54 rows per MAGE source share sentences with base gradtex, which is built on MAGE seeds. v14 GRADTEX re-labels the same human seeds with rain1's own modern completions as **span** labels. That complements the base's document-only GRADTEX; it does not duplicate it.
- **ACL/PMC vs base papers and human pool.** No shared sentences, but both cover the same domain. The base already holds 6,000 paper windows per epoch, plus PMC/ACL/arXiv in the human pool.
- **ASAP/PERSUADE/Writers SE vs base human pool.** No shared sentences in the drawn windows, but the pool already contains asap2 (4,206), persuade (699) and stack_nontech (5,726). Nothing new.
- **Hetero vs base and frozen suite.** JMLR, Gutenberg and Standard Ebooks share no sentences with any base window, the frozen suite or selection/calibration. The hetero val/test splits share no sentences with base training either, so they are a fair held-out set for both arms. JMLR is not ICML/NeurIPS proceedings, so it is a new paper source next to the eval papers.
- **Hetero internal duplication.** 1,697 pure-human windows of mixed docs are byte-identical to windows of their own control. The spec avoids this duplication by keeping only the mixed/AI windows of mixed docs.

### Benchmark-v3 exposure per source (whole-source, not just text)

| Candidate | v3 panel | Exposure | Decision |
|---|---|---|---|
| v14 LLMTrace (train split) | llmtrace_detection 1,478, classification 746 | Exact texts and 42 shared topics. Same generators and construction | **Drop the whole source.** Capping it would still make v3 LLMTrace in-distribution for the treatment arm only |
| hetero (all) | generate_heterogeneous 600 (hetero TEST split) | Group-disjoint (0 shared `het_group`/`het_source` keys), but the same generator pipeline | Keep. **Report v3 generate_heterogeneous and hetero val/test as in-distribution for the treatment arm**, not as generalisation |
| hetero WikiText | wikitext_detok 1,000 eval + 500 calibration | 16 docs share text, and the source is the same | Drop the whole source (only 156 docs) |
| hetero Gutenberg / Standard Ebooks | pg19 1,000 + 500 calibration | No text overlap; same kind of source (pre-1919 Gutenberg books) | Keep. The base human pool already holds 12.5k Gutenberg passages, so both arms are equally exposed |
| v14 ASAP/PERSUADE | persuade 1,000 + 500 calibration | Shared text | Dropped (also redundant and NC). Note the base pool already contains PERSUADE and ASAP; that is a base issue for v3, affecting both arms |
| v14 MAGE | none (MAGE is a v3 *reserve*) | — | Dropped for quality reasons (§4) |
| v14 GRADTEX, science_v9 | none | Clean | Keep |

## 4. Risk assessment

- **Common Pile news: drop.** In v8, swapping in 500 Common Pile documents raised external-article false alarms from 64 to 77 of 150, and the report labelled it a FAIL (`publication_hardneg_v8.md`). Generic news was "too easy and did not raise the threshold".
- **MAGE: drop.** Of 4,801 AI-bearing MAGE rows, 3,843 come from *continuation* generators, and MAGE continuations begin with the human prompt prefix. Here the whole row is labelled AI, so the human first sentence is labelled AI. 176 of those prefixes literally match human rows in v14 or base GRADTEX, which is direct label conflict. 1,876 rows include models under 3B (opt-125m, flan-t5-small, and so on). The text-level "diversity" is therefore 2020–23 artefacts that teach "incoherent = AI". woog's own MAGE subset already excludes continuation setups (`TRAINING_DATA_INVENTORY.md`). v14 (PERSUADE, Writers SE, MAGE CMV/ELI5) also moved false alarms elsewhere: CNN 8→25/500, LLMTrace human 5→28/720, broad 19→25/3,579.
- **LLMTrace: drop for this run.** It is the best short-span and multi-generator source, and rain1 capped it at 3% for good reason (`span_external_sources_v5.md`). It is also a v3 evaluation panel, and 46 rows overlap v3 directly. It could return in a later run only if v3's LLMTrace panels are retired or reported as exposed.
- **ACL/PMC v14: drop.** These are synthetic joins of unrelated excerpts (`span-training-data-v4.md` warns that join detection becomes a shortcut). 27% of boundaries fall mid-sentence at arbitrary cut points. AI text comes from SmolLM2-1.7B or Qwen2.5-0.5B. The source is redundant with the base's 6,000 paper windows per epoch from a much stronger generator.
- **ASAP2, PERSUADE, Writers SE, Dolly, craphound: drop.** These are redundant with the base human pool, overlap v3 persuade, or carry NC/label-purity risk:
  - PERSUADE and craphound are NC.
  - The ASAP AI side is half Qwen2.5-3B (NC).
  - Dolly is 2023 instruction responses whose "no generative AI" rule is unverifiable. It belongs in the same class as No Robots, which the user distrusts, and it was itself a frequent false-alarm source (9–15/150).
- **Beige Book: drop.** The user considers it confounding. Hetero writers rewrote Fed district reports, a near-formulaic genre, so they are easy to tell apart on content alone. 73/173 pairs also share boilerplate sentences with the hetero eval or v3 splits.
- **science_v9 (NASA/NOAA): keep the human side only.** The paired science articles produced rain1's largest single false-alarm drop (external 77→39/150, `science_paired_v9_comparison.md`). The base pool has no dated, bylined science journalism. The AI side comes from 1.7B/3B models, and 150 rows are NC.
- **v14 GRADTEX spans: keep.** It is rain1's own preserved-context completions from gemma-4-31b and mistral-small-3.2, two more modern generator families. 50% of its boundaries are intra-sentence and 20% of spans are under 200 chars. Boundaries are exact by construction ("inferred from retained context"), though not published gold.
- **Heterogeneous-ai-spans: keep as the core addition.** These are coherent in-place replacements, not unrelated joins, written by 6 frontier writers with matched controls. The risks:
  1. Every span is block-level, so it may reinforce "AI = whole paragraphs".
  2. The writers expand a Haiku brief, so they share a prompt style.
  3. Opus 5.5 is fiction-only, so writer and domain are confounded.
  4. Old-fiction contrast could become a "modern prose in Victorian text" shortcut. The controls and Standard Ebooks guard against this, but only partly.

## 5. Proposed mix (stage 2; stage 1 unchanged)

The dose is about **2,850 windows per epoch, +11.9% on the base's 24,000 rows** (simulated with the builder's own selection logic; epochs 0/1/2 = 2,853 / 2,823 / 2,855). That is larger than rain1's +6% (v14), because heterogeneous-ai-spans is the point of the experiment. It still keeps the base at ≥89% of every epoch.

Every addition document appears in exactly one epoch, assigned by a hash of its pair key. No addition text repeats, whereas base rows are drawn from much larger pools. A cap above about a third of a key's available windows can never be reached.

| Source key(s) | Decision | Windows/epoch | Filter or sampling | Rationale |
|---|---|---:|---|---|
| hetero:mixed:jmlr_pre2015 | keep | 600 (≈all available) | window_kinds mixed+ai | Paper-domain anchor with new generators. Pre-2015, not ICML/NeurIPS |
| hetero:mixed:gutenberg_selected | keep, capped | 650 (of ~1,110) | mixed+ai | Breadth and Opus 5.5/Opus 3 coverage. Capped so fiction does not dominate |
| hetero:mixed:standardebooks | keep | 200 (≈all) | mixed+ai | Clean modern transcriptions; limits the Gutenberg-formatting shortcut |
| hetero:mixed:hansard | keep | 2 | mixed+ai | Trivial |
| hetero:human_control:{jmlr, gutenberg, standardebooks, hansard} | keep | 450 / 450 / 120 / 2 | all windows; paired first | Same-passage negatives, including the original of each replaced block. All selected controls are paired (≈430 pairs per epoch, 0 unpaired). About half of the ≈840 selected mixed docs have their control in the same epoch |
| hetero Beige Book (both kinds), WikiText (both) | drop | 0 | — | User concern, leakage, v3 wikitext panel |
| v14:elisabeth-pl-pl/GRADTEX | keep | 200 (≈all) | — | Only remaining short and intra-sentence span supply, from 2 modern open models |
| v14:science_v9:noaa_fisheries / nasa_earth_observatory | keep, human only | 140 / 55 | window_kinds human | Hard negatives that proved themselves in v9 |
| v14 LLMTrace | drop | 0 | — | v3 exposure (see §3) |
| v14 MAGE ×9, ACL, PMC, ASAP2, PERSUADE, Writers SE, Dolly, Common Pile ×5, craphound | drop | 0 | — | §4 |

Per-epoch composition of the additions:
- hetero_mixed ≈1,440: ~1,250 mixed + ~180 pure-AI windows
- hetero_controls ≈1,020 (human)
- v14_mixed (GRADTEX) ≈200
- v14_human (science) ≈190

The AI share of labelled characters in the additions is 0.36, matching the base's 0.34.

Hetero mixed windows by writer, per epoch: Opus 5.5 ~385, Haiku ~280, Sol ~270, Sonnet ~245, Luna ~240, Opus 3 ~25. **About 83% of hetero AI windows (and all GRADTEX ones) come from writers the base has never seen.**

Expected effects:
1. Better localisation on frontier writers other than Luna, and on non-CS domains (hetero val/test, and v3 `generate_heterogeneous`, which is in-distribution for the treatment arm).
2. Better mixed-document recall on paragraph-scale replacements in long documents.
3. Little change in small-edit recall (still ~0.21–0.27 expected).

What to watch, evaluated with the same rule for both arms:
- Human false alarms that rain1 saw move with every mix-in: document-any highlight rate on the v3 calibration and human panels (cnn_dailymail, pg19, imdb, wikitext, persuade, scientific_papers), frozen-suite `human_paper_*` and ELLIPSE.
- Human sentences adjacent to edits in paper_v3 and workflow. Hetero teaches block-level edits and could raise false positives on neighbours.
- Writer-sliced recall on hetero test: Opus 3 (n≈25/epoch in training) and Opus 5.5 (fiction-only).
- Small-edit recall, to confirm the expected null.

## 6. Files

- `mix-spec.json` follows the coordinator's key convention (`v14:<source>`, `hetero:<kind>:<source_dataset>`). It has top-level `stage1` and `pair_controls_with_mixed: true`. **Set `pair_id = source_id` for hetero rows** so that each mixed document and its control share an epoch. For v14, `group_id` is fine.
- `leakage-drop-ids.json` maps `{source_key: [parquet id]}`: 397 ids covering the builder sets, the frozen suite, hetero val/test and benchmark-v3, closed over pairs. It also lists ids for sources the spec already drops, for completeness.

Uncertainties:
- The near-duplicate thresholds (≥8 shingles of 12 words, or ≥5% coverage) are judgement calls. Weaker hits were kept as prompt or boilerplate quotes (ASAP 44, PERSUADE 13, a few JMLR and Gutenberg).
- The base boundary and clip statistics are approximate because base rows are pre-cropped.
- The rain1 results cited above come from Qwen3-1.7B and a different base mix; their direction should transfer, but not their magnitude.
