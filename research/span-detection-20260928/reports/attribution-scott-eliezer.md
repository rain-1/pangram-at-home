# Scott Alexander and Eliezer Yudkowsky author-attribution corpus

## Acquisition result

The bounded acquisition produced 144 whole-post records, stored outside the repository at `/mnt/f/pangram-at-home/data/attribution_authors_v1/scott_eliezer/`:

| Author ID | Records | Publication years | Basis |
|---|---:|---|---|
| `scott-alexander` | 99 | 2022 | Astral Codex Ten public archive/API; essay-length public posts |
| `eliezer-yudkowsky` | 14 | 2020–2022 | LessWrong public profile/API; essay-length posts |
| `eliezer-yudkowsky` | 31 | 2015–2019 | Older, separately date-stratified fallback |

Records are whole essays, not chunks. For ACX, the fetcher requires `audience=everyone` and exactly one published byline, Scott Alexander. It excludes short items, open threads, link roundups, comments/highlight compilations, prediction-contest posts, recognizable research/update compilations, and titles indicating an AI-writing experiment. It strips page chrome, code/preformatted blocks, and blockquotes where present; this does not guarantee all quotations or third-party text are removed. A 150-word minimum is applied after cleaning. Cleaned text is generated with `lxml`, already listed in `requirements.txt`.

## Sources and evidence

- Scott Alexander: [Astral Codex Ten public archive](https://www.astralcodexten.com/archive?sort=new), [archive API](https://www.astralcodexten.com/api/v1/archive?sort=new), and public per-post API at `https://www.astralcodexten.com/api/v1/posts/<slug>`. Records include the post's `publishedBylines`, publication date, canonical URL, audience, type, and platform `updated_at` value. The collected 99 posts were published in 2022; current-version modification dates are 2023–2026. The legacy [Slate Star Codex site](https://slatestarcodex.com/) returned HTTP 403 from this acquisition environment, so the SSC originals were not fetched.
- Eliezer Yudkowsky: [LessWrong author page](https://www.lesswrong.com/users/eliezer-yudkowsky) and its public [GraphQL API](https://www.lesswrong.com/graphql), which exposes author-post lists, post body HTML, and `postedAt`. The official [GraphQL tutorial](https://www.lesswrong.com/posts/LJiGhpq8w4Badr5KJ/graphql-tutorial-for-lesswrong-and-effective-altruism-forum) documents the public endpoint. The API schema queried here did not expose an edit/last-modified timestamp for posts.

The author IDs denote the source-platform byline, not independently verified legal or biological identity. A pre-2023 publication date and byline make a post a plausible human-authored candidate; neither proves how its current text was composed or edited. For Scott, the current archive explicitly reports later updates. For Eliezer, the endpoint lacks version timestamps. Therefore all **144** rows have `strict_human_candidate=false`; these are attribution examples, not known-human labels and not detector-training data. No pre-2023 Wayback snapshot was verified for a matching stored version. One sampled Scott URL's nearest Wayback capture was from 2023, after the cutoff.

## Rights and handling

The post pages did not expose a dataset-level reuse license for this corpus. Substack's [Terms of Use](https://substack.com/tos) state that original creator content remains the creator's and restrict copying/distribution without owner consent. We did not verify a LessWrong-wide license or audit every post for an individual license; treat these texts as copyrighted unless a separate post-level license is confirmed. The corpus is kept private and external to Git; do not redistribute the text absent permission or an applicable legal basis. The raw HTML files and cleaned text stay under the external data directory.

## Files, integrity, and limitations

- `records.jsonl`: cleaned essay text plus `author_id`, source-platform `document_id`, title, canonical URL, publication date, source modification date when available, attribution/human-origin flags, rights status, and hashes.
- `raw/`: source HTML bodies fetched from the public endpoints. Each row hashes the fetched source body and cleaned text separately. `group_id` uses exact cleaned-text SHA-256 to help identify exact duplicates; it does not detect semantic or cross-platform duplicates.
- `manifest.json`: counts and machine-readable provenance limitations.
- `scripts/fetch_attribution_scott_eliezer.py`: repeatable bounded fetcher. Its defaults fetch at most 100 Scott posts, 60 Eliezer posts, and up to 31 older Eliezer fallbacks.

### Parser correction

An audit found that Python's `html.parser` retained invalid nested `<p>` tags in older LessWrong HTML. Because the cleaner extracted both paragraph ancestors and descendants recursively, it repeated text dramatically: `Hero Licensing` was 3,007,203 words, `Security Mindset and Ordinary Paranoia` 518,648, and `Security Mindset and the Logistic Success Curve` 411,239. The cleaner now parses with `lxml` (which repairs malformed paragraph markup) and extracts text owned by each paragraph/heading/list-item while skipping descendant block tags. All 144 cleaned texts were regenerated offline from the stored raw HTML; no source was fetched again.

After correction, those three essays are 16,267, 8,060, and 6,174 words. Across the corpus, the shortest is 150 words and the longest is 19,936. Raw-source byte hashes and cleaned-text hashes all verify; strict-human candidates remain zero. A regression test covering malformed nested paragraphs and nested list items passes (`pytest -q tests/test_attribution_clean_html.py`). A grant-updates compilation was excluded because it summarizes multiple grantees' writing. Re-run only into a fresh versioned directory before treating a later acquisition as a frozen release. No exact-duplicate or quote-level human review was performed.

## Other contemporary author candidates

For a later, smaller comparison set, consider Joel Spolsky ([Joel on Software](https://www.joelonsoftware.com/)), Jeff Atwood ([Coding Horror](https://blog.codinghorror.com/)), and Patrick McKenzie ([Kalzumeus](https://www.kalzumeus.com/)). Their first-party archives offer dated byline-level material. They are only candidates: establish publication/version dates, inspect licenses, and quarantine any text without a version-specific human-origin basis before use as human data.
