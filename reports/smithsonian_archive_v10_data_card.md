# Archived magazine prose candidate pool

This is a future hard-negative source for publication false positives. It is
**not part of v9 training**. Article text and WARC captures stay locally at
`/mnt/f/pangram-at-home/data/smithsonian_archive_v10/`; the repository contains
only collection code and aggregate documentation.

The source is [Smithsonian Magazine's dated Smart News archive](https://www.smithsonianmag.com/category/smart-news/?page=355).
We read [Common Crawl WARC captures](https://commoncrawl.org/get-started) made
in 2022 and accepted only articles with a named author, a pre-2023 publication
date, a capture after that date, and at least 450 words of extracted body prose.
The collection contains 100 articles: 79 training candidates and 21 locked
human-test articles. The test set holds out three authors entirely from the
training candidates. Median article length is 587 words.

The normalized 24-word fingerprint audit found zero overlaps within the pool
or against protected v9 training, calibration, external article, collaboration,
and archived government-science partitions. Exact source URLs from the
Human Detectors article evaluation were excluded before collection. The source
and split manifests on F record hashes and capture metadata.

These articles are copyrighted. Keep the prose and WARC captures in local
research storage and do not redistribute them with the open-source code.
