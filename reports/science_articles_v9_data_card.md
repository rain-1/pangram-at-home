# Science publication human-text pool v9

This collection addresses the detector's high false-positive rate on science and
magazine-style articles. It contains full-article prose, not abstracts. The
article text and raw HTML stay on `/mnt/f/pangram-at-home/data/science_articles_v9/`;
this repository contains only collection code and aggregate documentation.

## Sources and intended use

| Editorial source | Use | Date evidence | Reuse evidence |
|---|---|---|---|
| NASA Earth Observatory | Training candidate | Page `article:published_time`; visible NASA story credit | [NASA Earth FAQ](https://science.nasa.gov/earth/faq/) |
| NOAA Fisheries feature stories | Training candidate | Visible feature-story publication date | [NOAA Fisheries website policy](https://www.fisheries.noaa.gov/website-policies-and-disclaimers) |
| NOAA Climate.gov science explainers | Calibration human pool | Page `article:published_time` and byline | [Climate.gov about](https://www.climate.gov/about) and [NOAA reuse policy](https://sos.noaa.gov/copyright/) |
| EPA Science Matters | Locked human test pool | Visible `Published` line in the 2017–2022 archives | [EPA disclaimers](https://www.epa.gov/web-policies-and-procedures/epa-disclaimers) |

Current accepted counts: 41 NASA Earth Observatory, 105 NOAA Fisheries,
41 NOAA Climate.gov, and 240 EPA Science Matters articles. The source-exclusive
split is 146 training candidates, 41 calibration-human documents, and 240
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
a historical-snapshot or editorial audit confirms their current text. Keep EPA
locked during training and threshold selection. NOAA Climate.gov and NOAA
Fisheries share an agency, so the calibration source is editorially distinct but
not institutionally independent of one training source.

Each record retains the URL, title, author or editorial-staff attribution,
publication date, retrieval time, text and raw-page SHA-256 hashes, word count,
and source. The split builder also creates title/topic-only prompts for matched
AI articles. Those prompts are not generated AI data; the human-only files must
not be treated as a balanced detector evaluation.

Use the raw/source text locally for research, preserving attribution. Individual
pages can contain material with separate rights even on public-sector sites;
figures are omitted and obvious third-party text is rejected, but review any
page before redistributing its content.
