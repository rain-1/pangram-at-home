# Science publication human-text pool v9

This collection addresses the detector's high false-positive rate on science and
magazine-style articles. It contains full-article prose, not abstracts. The
article text and raw HTML stay on `/mnt/f/pangram-at-home/data/science_articles_v9/`;
this repository contains only collection code and aggregate documentation.

## Sources and intended use

| Editorial source | Accepted articles | Use | Date evidence | Reuse evidence |
|---|---:|---|---|---|
| NASA Earth Observatory | 41 | Training candidate | Page `article:published_time`; visible NASA story credit | [NASA Earth FAQ](https://science.nasa.gov/earth/faq/) |
| NOAA Fisheries feature stories | 105 | Training candidate | Visible feature-story publication date | [NOAA Fisheries website policy](https://www.fisheries.noaa.gov/website-policies-and-disclaimers) |
| NOAA Climate.gov science explainers | 41 | Calibration human pool | Page `article:published_time` and byline | [Climate.gov about](https://www.climate.gov/about) and [NOAA reuse policy](https://sos.noaa.gov/copyright/) |
| EPA Science Matters | 240 current pages; 151 archived extracts | Locked human test pool | Visible `Published` line in the 2017–2022 archives; historical WARC timestamp for 151 | [EPA disclaimers](https://www.epa.gov/web-policies-and-procedures/epa-disclaimers) |

The source-exclusive split is 146 training candidates, 41 calibration-human documents, and 240
locked-test-human documents. The split manifest, data and raw HTML are on F.
The full overlap audit found zero sampled 24-word matches against the listed
prior train/validation/test partitions and zero matches across the four new
sources (`overlap_audit.json` on F).

All accepted pages carry an original publication date before 2023. Text comes
from article paragraphs after figures, figure captions, navigation, references,
and boilerplate are removed. We require about 450–500 words, preserve the raw
page and hashes, and reject sampled 24-word phrase overlap with existing v6/v8
training and protected evaluation files. Scripts are
`scripts/collect_science_articles_v9.py`,
`scripts/collect_epa_science_matters_v9.py`,
`scripts/clean_science_articles_v9.py`, and
`scripts/build_science_articles_v9_splits.py`. The overlap check is
`scripts/audit_science_articles_v9.py`.

The date is evidence of original publication, not proof that every word in the
current page snapshot predates 2023. NASA pages in particular may show later CMS
modification dates. These articles are therefore **candidate human data** until
a historical-snapshot or editorial audit confirms their current text. A
rate-limited Common Crawl lookup recovered 164 dated pre-2023 EPA captures;
151 yielded substantial historical article prose. Of those, 147 retain at
least 90% of the current page's sampled 13-word sequences. Use the 151
historical extracts as the primary high-confidence human publication test,
and treat the remaining current-page EPA articles as provisional stress data.
Keep EPA locked during training and threshold selection. NOAA Climate.gov and NOAA
Fisheries share an agency, so the calibration source is editorially distinct but
not institutionally independent of one training source.
`scripts/find_science_articles_v9_archives.py` can locate pre-2023 Common
Crawl captures; `scripts/fetch_science_articles_v9_archives.py` extracts the
dated WARC text. The archive-verified extracts live in their own file on F;
the 240 current-page EPA records remain separately available.

Human articles are dated before 2023 while locally generated AI mirrors are
from 2026. Authorship and era are therefore confounded in this pilot; evaluate
on separately sourced recent verified-human material before making broad claims.

Each record retains the URL, title, author or editorial-staff attribution,
publication date, retrieval time, text and raw-page SHA-256 hashes, word count,
and source. The split builder also creates title/topic-only prompts for matched
AI articles. The local Qwen2.5-3B-Instruct pilot generated 60 accepted AI
articles (20 calibration, 40 held-out test); four incomplete last sentences
were trimmed. The resulting full-article paired pilot has 20 human/20 AI
calibration documents and 40 human/40 AI held-out documents. These AI texts
come from one model only and are not a broad multi-model benchmark.

There is a length shortcut in the full articles: human versus AI median lengths
are 773.5 versus 599 words in calibration and 729 versus 563 in the held-out
pairs. A length-only score obtains AUROC 0.7825 and 0.7634 respectively.
**Do not use full-article paired accuracy as the main detector result.** The
equal-input view contains opening and ending 512-source-token windows for each
paired article (80 calibration rows, 160 held-out rows, each half human). Score
these with the window detector; cluster uncertainty by parent article because
the two windows from each document are correlated. The full human-only pools
remain separate for false-positive stress testing.

For the strongest paired view, 27 AI pilots have a corresponding archived EPA
human article. `paired_archived_locked_test_windows_pilot.jsonl` contains 108
equal-token windows (54 human, 54 AI). The archived 151-document human-only
file should be the primary false-positive check. A sampled 24-word fingerprint audit
found no overlap between these historical extracts and protected training or
evaluation partitions, and none between them and the 60 generated AI pilots.
The archived full-article pairs have an even stronger length shortcut
(length-only AUROC 0.854), reinforcing the need to use equal-token windows.

Use the raw/source text locally for research, preserving attribution. Individual
pages can contain material with separate rights even on public-sector sites;
figures are omitted and obvious third-party text is rejected, but review any
page before redistributing its content.
