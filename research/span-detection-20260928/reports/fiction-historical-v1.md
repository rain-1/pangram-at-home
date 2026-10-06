# Historical fiction human-source pilot

## Result

Acquired **12 complete works by 12 distinct authors**—not chapter-level pseudo-examples—into `/mnt/f/pangram-at-home/data/fiction_candidates_v1/historical/records.jsonl`. The set totals **869,170 words**, with work lengths from 39,258 to 148,329 words. Each record keeps the whole work in one `work_id` / `group_id`, plus author ID(s), first-publication year, exact Standard Ebooks Git commit and date, original-source links, rights statement, and raw/clean SHA-256 hashes.

| Original year | Work — author | Broad style/genre | Words | Snapshot date |
|---:|---|---|---:|---|
| 1818 | *Frankenstein* — Mary Shelley | Gothic/speculative | 74,853 | 2022-11-12 |
| 1889 | *A Connecticut Yankee in King Arthur’s Court* — Mark Twain | Satire/time-slip | 117,001 | 2022-11-12 |
| 1895 | *The Red Badge of Courage* — Stephen Crane | War fiction | 45,994 | 2022-11-12 |
| 1900 | *The Wonderful Wizard of Oz* — L. Frank Baum | Children’s fantasy | 39,258 | 2022-11-12 |
| 1906 | *The Jungle* — Upton Sinclair | Social realism | 148,329 | 2022-11-12 |
| 1908 | *The Man Who Was Thursday* — G. K. Chesterton | Mystery/speculative | 57,412 | 2022-11-12 |
| 1912 | *The Autobiography of an Ex-Colored Man* — James Weldon Johnson | African American literary fiction | 51,565 | 2022-11-12 |
| 1915 | *Herland* — Charlotte Perkins Gilman | Utopian/speculative | 52,193 | 2022-11-12 |
| 1920 | *The Age of Innocence* — Edith Wharton | Literary/social realism | 101,358 | 2022-11-12 |
| 1925 | *The Great Gatsby* — F. Scott Fitzgerald | Modernist literary fiction | 48,185 | 2021-01-01 |
| 1925 | *Mrs Dalloway* — Virginia Woolf | Modernist literary fiction | 63,363 | 2022-11-12 |
| 1927 | *Death Comes for the Archbishop* — Willa Cather | Historical literary fiction | 69,659 | 2022-11-13 |

These 12 authors and works are disjoint from the ten Standard Ebooks works used in the existing classic-fiction calibration set. The acquisition script checks that author/work exclusion against that set before fetching.

## Provenance and rights

[Standard Ebooks describes itself as a volunteer-driven project producing carefully formatted public-domain ebooks](https://standardebooks.org/about). It says the underlying text and artwork are believed to be in the U.S. public domain and that it dedicates its ebook production to the public domain; the [public-domain explanation](https://standardebooks.org/about/standard-ebooks-and-the-public-domain) states that the edition files are dedicated under CC0. Each selected EPUB source package also carries a `dc:rights` statement, retained in row metadata.

The source texts are historical books first published from 1818 to 1927. Each ebook is pinned to the exact Standard Ebooks GitHub commit in the row; all 12 commits are dated no later than November 13, 2022. The Standard Ebooks OPF metadata records each author, edition date, subjects, rights statement, and one or more source-edition URLs (such as Gutenberg, HathiTrust, Internet Archive, or Faded Page). This gives a more defensible human-origin basis than a recent platform page with a byline but no version history. It still does not certify every transcription or edit by forensic authorship review.

The source project regards the underlying books as U.S. public domain and its own ebook files as CC0. Public-domain status is jurisdiction-specific; verify local status before redistribution or commercial release. The corpus stays external to Git.

## Data checks

The extractor follows each EPUB spine to preserve reading order, excludes edition apparatus such as imprint/colophon/license pages, and keeps all prose from a work in one document. Text is read from the XHTML body; no generated continuation or chapter-level sample is created. Full metadata and source tarballs are in the external output directory.

Checks on the acquired output found:

- 12 rows, 12 distinct author IDs, and 12 distinct work groups.
- Every source commit predates 2023; every row has author, source-version, rights, raw-hash, and clean-hash metadata.
- 39,258–148,329 words per work; hashes match the raw archive bytes and cleaned text.
- No overlap in source work or author with the existing Standard Ebooks calibration manifest.
- Opening and closing samples are story text; no edition license/colophon text appears in the selected chapter-body extraction.

Reproduce into an empty destination with `python scripts/fetch_fiction_historical.py`. The script refuses to overwrite an existing `records.jsonl`. Its bounded selection is intentional; do not treat the 12 works as 12 chapters or split samples from one work across train and evaluation.

## Scale-up route and limitations

The strongest documented scale-up source found is the Library of Congress [Selected Digitized Books Data Package](https://data.labs.loc.gov/digitized-books/). Its official package describes **90,414 book records and 84,058 full-text OCR files**, with thousands of fiction titles; its metadata and text collection were assembled from records available through August 2022, and the package says the books are public domain and free to use and reuse. The [official README](https://data.labs.loc.gov/digitized-books/README.pdf) describes tens of thousands of mostly U.S.-published English works and explicitly warns that the collection reflects selection and preservation priorities rather than a random sample. The field `date` can represent a creation, publication, or referenced date, so confirm dates against item-level metadata and title-page scans before calling a work date-bounded. OCR quality and segmentation need auditing. A practical next batch would select a few hundred fiction books, verify author/date against each scan, normalize OCR, cap per author, and reserve entire works for source-level splits.

This Standard Ebooks pilot is also historical: it gives clear pre-2023 editions and broad genre variety, but not contemporary fiction. Historical prose has archaic spelling, typesetting, and genre conventions, so scores on this pool do not establish performance on living writers. For more recent writing, the practical path is a purpose-built opt-in corpus from writers with explicit authorship and version-date attestations and a usable license. A periodical-fiction collection could add short stories, but magazine pages often have OCR, fragmented layouts, uncertain bylines, and item-specific rights; it needs per-story source review. No magazine-derived texts were included in this pilot.
