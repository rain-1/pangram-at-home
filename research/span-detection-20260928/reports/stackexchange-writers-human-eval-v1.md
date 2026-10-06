# Writers Stack Exchange human writing audit

Downloaded the official `writers.stackexchange.com.7z` dump from the
[Stack Exchange archive](https://archive.org/download/stackexchange/writers.stackexchange.com.7z).
The archive files are dated April 2024; the extractor kept only posts created
and last edited by **2022-12-31**, with an identified owner, 50–700 words of
cleaned prose, and no email address in the body. Obvious bot/community user
names were excluded using the matching `Users.xml`. Questions and answers are
grouped by parent post for future splitting. The bounded reservoir is 5,000
rows from 38,991 eligible posts, with 1,975 distinct owners. It contains 1,193
questions and 3,807 answers; median passage length is 164 words. Source XML
and Parquet remain on `/mnt/f` outside Git.

This is a **second social/Q&A platform** alongside Reddit, useful for an
independent human false-positive audit. The 1,000-row model audit is selected
by a fixed text-ID hash from the 5,000-row corpus. No Writers Stack Exchange
text enters training or threshold selection. A pre-2023 post date is strong
but not absolute human-authorship evidence; copied, assisted or automated
writing may remain. Current archive extraction of old posts is also weaker
than an actual 2022 snapshot because revision history is not independently
reconstructed.

Contribution terms are CC BY-SA, with license version depending on contribution
history. Attribution and share-alike obligations and possible revision-level
license changes need review before training or redistribution. Keep this a
research evaluation slice. The extraction records source URL, owner ID, post
ID, creation/edit dates, license label, hashes, and cleaning method. See
`scripts/fetch_social_human.py` and the local manifest at
`/mnt/f/pangram-at-home/data/stackexchange_writers_v1/manifest.json`.

The raw `Posts.xml` SHA-256 is
`1f04f8bf463f1223f7c5d548c7737bd861e9292befc25c41435e12881e44b745`;
the Parquet SHA-256 is
`5b701b902a37b92b79589e586f994a5e7232d589cbc5f16020ef0622785557d7`.

Source and rights: [Stack Exchange dump releases](https://meta.stackexchange.com/questions/224873/all-stack-exchange-data-dump-releases),
[Writers Stack Exchange licensing](https://writers.stackexchange.com/help/licensing).
