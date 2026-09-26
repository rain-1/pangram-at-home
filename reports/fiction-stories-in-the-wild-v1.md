# StoriesInTheWild fiction candidate

Downloaded the authors' [primary release](https://gist.github.com/talaugust/edc87a274ae6dcedf43774a8d9e91bb4)
at revision `15a32423073310bfd521a851d10367321248254f`, committed on
2020-05-13. The [2020 study](https://aclanthology.org/2020.nuse-1.6/) describes
volunteers writing stories from image prompts, either all at once or in chunks.
Its writing-process evidence is stronger human-origin support than a dated web
byline alone. It is still source-level provenance, not a sentence-level forensic
guarantee.

The pinned CSV has **1,563 rows**, although the paper describes 1,630 stories.
We preserve this unresolved discrepancy rather than claiming the whole reported
corpus was obtained. Filtering at 80 words excludes 479 rows; normalized exact
deduplication excludes another 49. The resulting **1,035 stories contain 142,901
words**. Story text is otherwise unchanged except outer whitespace. Processed
records omit demographic and personality fields.

| Image prompt | Retained stories |
| --- | ---: |
| Dog | 428 |
| Jail | 252 |
| Park | 248 |
| Marathon | 97 |
| Snow | 10 |

There are 481 stories written all at once and 554 written in chunks. The release
does not expose a stable author ID, so story IDs cannot establish author
independence. With only five prompts, a random story split would be a weak
generalization test. Prefer reserving this small corpus for an independent
human false-positive audit; if split later, hold out whole prompts. Human-only
data cannot measure AI recall or balanced accuracy by itself.

The authors publicly release the data for research and request citation; no
separate data license was found in the pinned files. Keep the collected text
outside Git. This source has not been added to detector training or scored.

Files live under
`/mnt/f/pangram-at-home/data/fiction_candidates_v1/stories_in_the_wild/`:
`records.jsonl`, `manifest.json`, the original `stories.csv`, and pinned source
metadata. The builder is `scripts/fetch_fiction_stories_in_the_wild.py`.
The combined fiction audit checks overlap against its explicitly listed
detector datasets before any split or training assignment.
