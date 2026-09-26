# Original online fiction candidate: Beneath Ceaseless Skies

Collected 2026-09-26 as a **private candidate**, separate from active detector training and frozen evaluations. Full text is in `/mnt/f/pangram-at-home/data/fiction_candidates_v1/original_online/records.jsonl`, with `manifest.json` beside it. The reproducible collector is `scripts/fetch_fiction_original_online.py`. This adds full, professionally published online short fiction rather than more benchmark snippets or historical public-domain books.

| Measure | Result |
|---|---:|
| Complete cleaned stories | 200 |
| Distinct bylined authors | 162 |
| Maximum stories by one author | 3 |
| Publication dates | 2008–2021; 162 published 2018–21 |
| Length | 1,513–28,166 words; median 5,689 |
| Archived original-page snapshots dated by 2022-12-31 | 9 |
| Current official pages with an older publication date, revision unknown | 191 |
| Exact or normalized 24-word overlap with prior diverse parent train/val/test | 0 |

## Provenance and quality

The [publisher's issue pages](https://beneath-ceaseless-skies.com/issues/2021/) provide title, byline and issue date, and link to each [full story page](https://beneath-ceaseless-skies.com/stories/letters-from-a-travelling-man/). The collector scanned the 2018–21 issue pages, found 300 distinct story URLs, and retained 200 full works with at least 450 cleaned words and no more than three from one author. Some older stories appear in later issues' archive sections; their actual publication date, parsed from the story page, is recorded. The 200 selected works have unique IDs and cleaned-text hashes.

Nine rows have a [Wayback Machine](https://web.archive.org/) snapshot of the publisher's original page dated no later than 2022-12-31. The other 191 come from current official pages and are marked `source_version_status=current_revision_unverified`; a pre-2023 issue date **does not prove the current wording was present then**. The archive service stopped accepting connections after a few successful requests, so the nine archived rows were preserved and the rest were collected with this lower provenance status. `strict_human_candidate` is true only for the nine archived versions, and even that does not prove every sentence was unaided human prose.

The [publisher's submission guidelines](https://beneath-ceaseless-skies.com/submissions/) now reject AI-drafted or AI-edited story text, but that policy was updated in 2023 and is **not retrospective proof** for the older stories. The same guidelines describe purchase of first publication and related rights; the stories remain copyrighted, so public access is not a general permission to redistribute or train on them. The full text stays private pending a rights review. No copyrighted story text is in Git. It may still be useful immediately as a private research-only evaluation candidate, subject to local policy review.

Each row carries `text`, `source`, `source_id`, `work_id`, `group_id`, `author_ids`, `author_name`, issue and publication date, original URL, archive URL/date where available, `source_sha256`, `clean_sha256`, `source_version_status`, `human_origin_status`, `rights_status`, `word_count`, and overlap flags. `group_id` equals the whole work ID: split by work **and author** before generating model windows. A single story can yield many correlated windows, so windows are not independent evaluation examples.

## Extraction and audit

The collector takes paragraphs only from the page's story-content element. It excludes page navigation, byline, author bio, discussion, advertising, and visible page furniture. An inspection of the starts and ends of sampled works found story prose rather than the publisher footer; no retained story begins with an extracted `By` or copyright line. All 200 work IDs and cleaned hashes are unique; the author cap is satisfied. The saved file hash agrees with the manifest: `bdf24eb92cf0758a1620663f15e0e45785aa50c105d021c875d01a5cab3d5473`.

The overlap check used exact cleaned SHA-256 and normalized 24-word shingles against `diverse_pyramid_v1/{train,val,test}_full.parquet`; it found zero selected records sharing a shingle. This is evidence for independence from those three files, not proof against all prior corpora or pretraining. The current creative mix already contains EditLens and MAGE WritingPrompts; this magazine is a new source family with longer stories, but remains a single editorial venue and mostly literary fantasy. Its 200 rows must not be presented as 200 genres or 200 independent publishing sources.

Of 300 indexed URLs, 55 current pages failed the story parser, five parsed with publication after 2022, and ten were excluded by the author cap before the 200-story target was reached. These counts describe this bounded pass, not a complete crawl of the publisher. Its 2018–21 issue archive and additional years offer a larger future route, provided source versions and rights are handled carefully.

For a **different** online storytelling mode, the 2020 [STORIUM paper](https://aclanthology.org/2020.emnlp-main.525/) reports about 6,000 long collaborative stories and 125 million tokens. The [official distribution site](https://storium.cs.umass.edu/) requires login and acceptance of a research data-transfer agreement. No STORIUM data was acquired or folded into this candidate. The separately collected StoriesInTheWild pool covers controlled online writing tasks; it should be reported as its own source family with its own lengths and rights.
