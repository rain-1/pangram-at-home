# Realistic span evaluation source: CoAuthor v1

## Recommendation

Use the Stanford CoAuthor release as a small independent mixed-authorship evaluation set. The official project describes 63 writers, four GPT-3 instances, 1,445 writing sessions, and timestamped keystroke-level interaction logs. Writers entered their own text, requested suggestions, and could accept, dismiss, or edit suggestions. The writing tasks are creative stories and argumentative essays, not scientific papers, so this tests realistic mixed authorship while leaving paper-domain evaluation to the paper sources and corpora.

Primary sources:

- [CoAuthor project page and download links](https://coauthor.stanford.edu/)
- [CoAuthor CHI 2022 paper (author-hosted PDF)](https://cs.stanford.edu/~minalee/pdf/chi2022-coauthor.pdf), DOI: [10.1145/3491102.3502030](https://doi.org/10.1145/3491102.3502030)
- [Dataset archive linked by the project](https://drive.google.com/file/d/1C9FCCsyY-5I7mcBHi-__R7lxHkGX_-9Q/view?usp=sharing)
- [Metadata and survey sheet linked by the project](https://docs.google.com/spreadsheets/d/1O3EXJm52TQHfFSbzVGZmNIzzdu5ow6IjnOBrGTUY02o/edit?usp=sharing): metadata tabs use sheet IDs `1870708729` (creative) and `320516663` (argumentative).
- [Authors' replay interface repository](https://github.com/minalee-research/coauthor-interface): its interface code is MIT licensed. [`dropdown.js`](https://github.com/minalee-research/coauthor-interface/blob/main/frontend/js/dropdown.js) records suggestion selection before inserting selected text; [`editor.js`](https://github.com/minalee-research/coauthor-interface/blob/main/frontend/js/editor.js) records Quill text-change source and delta; [`logging.js`](https://github.com/minalee-research/coauthor-interface/blob/main/frontend/js/logging.js) replays `text-insert` and `text-delete` events with `quill.updateContents(replayLog.textDelta.ops)`. The code license applies to the interface, not necessarily the dataset.

## Provenance and label construction

The released event records identify text changes with `eventName`, `eventSource`, `eventTimestamp`, and Quill `textDelta` operations. The project interface records suggestion selection, then calls `appendText(suggestion)`; Quill reports that insertion as an API sourced text change. User keystrokes appear as user sourced text changes. In the 206 selected session logs, 1,352 suggestion selections pair in order one-to-one with 1,352 API sourced text insertions. The script maps those API insertions to AI label `1`, and one-character user insertions to human label `0`.

Conservative exclusions:

- The prefilled prompt/starter string in `system-initialize.currentDoc` is unlabelled (`-100`): it is not a participant keystroke.
- User multi-character insertions are unlabelled because the event does not distinguish pasted text from typed/composed input. This avoids claiming provenance the log does not establish.
- When a user inserts inside a surviving AI suggestion or deletes any part of one, all remaining characters from that suggestion are masked. User text inserted in the same edit is also masked; nearby single-character input continues to inherit the ambiguous edit context. This is a conservative approximation of edited AI spans.
- The parser uses UTF-16 offsets for source Quill operations and emits exclusive-end offsets in Unicode code points, matching Python tokenizer offset maps.
- Spans partition the full text; prompt, paste-ambiguous, and edited regions use label `-100`.

The source has `currentDoc` populated on initialization events only; it does not provide post-edit snapshots for an independent snapshot comparison. All 1,447 archive session files inspected had exactly one `system-initialize` event. The builder replays every text delta twice, once with provenance tracking and once with a text-only implementation; it asserts both final texts match and that each logged cursor is within the replayed document's UTF-16 length. Focused unit checks exercise cursor offsets, replacements, edits, and mask propagation. This is a mechanical audit, not an author-reviewed label audit.

## Acquired evaluation slice

Raw archive and metadata CSVs, along with derived text, are stored outside the repository under `/mnt/f/pangram-at-home/data/`. The builder is [build_span_realistic_eval_v1.py](../scripts/build_span_realistic_eval_v1.py); its output is `span_realistic_eval_v1/test.jsonl`, with a manifest containing archive, per-selected-session, metadata, and output hashes; fixed selected session IDs; hashed writer groups; and counts. The deterministic writer split uses seed 8; it selected 206 sessions across 31 writers, and no CoAuthor sessions or writers were used in our span training. Of the selected sessions, 119 contain at least 100 scored characters and at least 50 scored characters of each class. The others failed a minimum-class or minimum-scored-length rule. This is source-disjoint and writer-disjoint within CoAuthor; identities are anonymous, so person-level overlap with writers in other corpora or the model's pretraining data cannot be established.

Included scored character counts are 112,053 human and 32,674 AI, with 134,009 ignored. The ignored total includes all initial prompts/starters and masked pasted or edited material. With the local Qwen3-1.7B tokenizer used for the current span model, all 119 rows retain both human and AI token labels after offset mapping (24,306 human tokens and 7,026 AI tokens scored). A normalized exact 24-word-window check found zero matches between the scored text in these rows and the current `span_training_v4` train and validation files (12,150 evaluation windows checked; 1,401,369 train/validation windows indexed). The initial prompt text remains in the model input with ignored labels, so this is not a prompt-independent evaluation. Keep it private and external for the planned research evaluation.

## Rights and limitations

The project makes the dataset downloadable for research and the paper encourages researchers to use, analyze, and extend it. I found no explicit license for the dataset on the project page, metadata sheet, paper, or archive. The MIT license in the separate interface repository covers code, not automatically participant text. Treat the acquired text and derived JSONL as private research-evaluation material; do not publish or redistribute them absent clarification from the dataset authors. The paper itself carries ACM copyright terms; those terms should not be assumed to grant a data license.

This set is small and covers only English creative and argumentative writing with GPT-3 suggestions collected in 2021. Labels capture recorded insertion provenance, not the cognitive origin of an idea. Single-character logged user input is stronger evidence than final text alone, but cannot rule out copying performed outside the interface; all logged multi-character user insertions are therefore masked. The masks reduce known ambiguity but cannot resolve all influence between human and AI text.

## Rebuild and focused checks

```bash
python scripts/build_span_realistic_eval_v1.py
PYTHONPATH=scripts python -m unittest scripts/test_build_span_realistic_eval_v1.py -v
```

The focused checks exercise UTF-16 offsets around emoji, exact delta replay, insertion/deletion replacement, propagation of edited-span masks, and masking multi-character user insertions.
