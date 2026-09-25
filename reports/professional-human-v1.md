# Professional human prose pilot v1

Fetched 2026-09-25 from the Federal Reserve Board's Beige Book year archives.
The pilot contains 32 reports released from 2019-01-16 through 2022-11-30,
with 1,622 extracted passages (all labeled human). Passages have 90–360 words;
each report is a group to preserve source-level splitting. The raw HTML,
Parquet, manifest, per-report hashes, and retrieval timestamps are under
`/mnt/f/pangram-at-home/data/federal_reserve_beige_book_v1/` and are not in Git.

## Provenance and rights

The Board's [Beige Book archive](https://www.federalreserve.gov/monetarypolicy/beigebook2022.htm)
lists eight dated releases per year. Each release says it was prepared at one
of the Reserve Banks from information collected by a stated cutoff date and
that it summarizes comments from contacts outside the Federal Reserve System.
The reports predate the end of 2022 by release date, supporting human authorship
of the edited prose. The contact material is partly paraphrased and some
individual statements originate with outside business contacts, so the source
is less definitive for authorship of every underlying sentence than a signed
article.

The Board's [copyright disclaimer](https://www.federalreserve.gov/disclaimer.htm)
says that, unless otherwise indicated, Board website information is public
domain and can be copied and distributed without permission, with attribution
to the Board. It also cautions that use may infringe privately owned rights.
The script excludes tables and graphics and retains the Board attribution and
rights URL per row. This is a reasonable public-domain candidate for research
and training, but the Board's caveat means it should remain labeled as a
candidate until the passages are audited for embedded third-party text. No
private emails, named contact details, or customer records were collected.

## Use

This source adds professional/economic prose absent from the diverse pilot and
works well as a separate human false-positive audit slice. Keep all passages
from a report together in one split. At 32 report groups, reserve several whole
reports for validation and test; do not split passages from a release across
those partitions. The collected reports are all from one publisher family and
one genre, so report results as a source-specific professional/finance slice,
not as representative of all business writing. The passages should not be
added to the current training run without updating its frozen manifest and
rebuilding the splits.

Reproduce with:

```bash
python scripts/fetch_professional_human.py
```

The script's default output is the raw data directory above. The manifest
records report URLs, dates, source hashes, passage counts, rights evidence, and
collection timestamp.
