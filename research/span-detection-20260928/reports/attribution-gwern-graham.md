# Private writer-attribution acquisition: Gwern and Paul Graham

Collected on 2026-09-26 for a **future exploratory author-attribution probe**. The full essay text is outside Git at `/mnt/f/pangram-at-home/data/attribution_authors_v1/gwern_graham/records.jsonl`; its manifest is beside it. The reproducible collector is `scripts/fetch_attribution_gwern_graham.py`. These essays are **not detector training data**, and no text has been assigned to train, validation, or test yet.

| Author | Whole essays | Attribution eligible | Stronger pre-2023 version candidates | Published 2020–22 | Published before 2020 | Eligible length, median (range), words |
|---|---:|---:|---:|---:|---:|---:|
| Gwern | 80 | 75 | 75 | 15 | 65 | 2,939 (459–50,200) |
| Paul Graham | 80 | 80 | 13 | 30 | 50 | 1,100 (312–12,525) |

Each row is a complete cleaned essay with a unique `document_id` and matching `group_id`. The parent attribution builder should split by `group_id` before creating any chunks. The `split` field is `unsplit_private_attribution` throughout. The `attribution_eligible` flag means the page is a usable author-byline example; `strict_human_candidate` records stronger **pre-2023 version evidence**, not proof that every sentence was unaided human writing.

## Source and provenance

- **Gwern:** collected from the [official essay index](https://gwern.net/index), with `created`, `modified`, `status`, and `author` fields from each official `.md` page. Selection required a finished page, site or explicit Gwern authorship, and both dates no later than 2022-12-31. The current source metadata identified 103 dated index candidates created or modified after 2022, which were excluded. Current HTML and Markdown source hashes are recorded for each selected essay. A site's retrospective date metadata is useful provenance, but it is not an independently archived 2022 snapshot.
- **Paul Graham:** collected from the [official essay index](https://www.paulgraham.com/articles.html); the [site biography](https://www.paulgraham.com/bio.html) supports attribution. Essay pages give month and year but no reliable current-page modification date. The 80 selected essays display dates through 2022; 13 also have successfully retrieved [Internet Archive](https://web.archive.org/) snapshots of the official pages dated no later than 2022-12-31. The remaining 67 are usable for author attribution but their **current revisions are unverified**. `publication_date` uses day `01` solely to encode Graham's month/year as ISO; `publication_date_precision` says so. An archived page's timestamp and URL are saved per row, with its source hash. The current HTML hash is separately saved even when archived text was used.

The [Gwern license page](https://gwern.net/about#license) states CC0 for the site's own text. Graham's [FAQ](https://www.paulgraham.com/gfaq.html) asks readers to link rather than mirror his essays. We keep all full text privately on the external drive and include only metadata and aggregate counts in Git. Any embedded third-party quotations or permissions would require separate review before redistribution.

## Text preparation and exclusions

Extraction removes navigation, byline/header material, block quotes, code, tables, figures, side notes, footnotes, link/reference sections, and obvious notes or thanks. After a final boundary pass, 53 Gwern records lost external-link or reference-list text. Eligible records contain at least 300 cleaned words. No eligible text begins with an author name byline. This is a conservative structural clean, not a line-by-line human-origin audit.

Five Gwern records remain in the file for audit but have `attribution_eligible=false` and `strict_human_candidate=false`: `gwern:cyoa`, `gwern:gpt-3-nonfiction`, `gwern:gpt-2-preference-learning`, and `gwern:rnn-metadata` discuss or demonstrate model-generated text; `gwern:lorem` contains placeholder or word-list material. `quarantine_reason` identifies each. Other essays can still contain uncaught quotation or model output, so strict candidates need human review before use as pure-human detector calibration.

An exact cleaned-text SHA-256 and normalized 24-word shingle comparison against **the existing diverse pyramid `train_full.parquet`, `val_full.parquet`, and `test_full.parquet` text** found zero overlapping selected records. This checks text overlap with those files; it does not establish independence from every earlier experiment, every quoted source, or unseen future data. `source_sha256`, `source_markdown_sha256` where applicable, `clean_sha256`, and the aggregate `records_sha256` make the saved version auditable. The final records file SHA-256 is `4aabb86e4ab7661da7cb847efd4acd914b58928a25bec49e22ca2e74d87e7a31`.

## Use in the future attribution experiment

The broad four-writer attribution cohort can use eligible, bylined pre-2023 essays while preserving the `human_origin_status` distinctions. A stricter subset could begin with the 75 Gwern metadata candidates and 13 archived Graham candidates, then require further review of potential mixed-origin passages. There are too few strict Graham examples for a balanced four-author evaluation, so the broad cohort and strict subset should be reported separately. Keep whole works together across splits, mask author names in surrounding prompts/metadata, and do not turn this probe corpus into detector training examples.
