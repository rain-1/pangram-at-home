# Creative fiction: new source collection

## What we acquired

**1,849 candidate records, containing 4,380,761 words**, are stored outside Git
at `/mnt/f/pangram-at-home/data/fiction_candidates_v1/`. These are human-source
research candidates with different levels of provenance, not 1,849 independently
verified human authors. No new fiction was inserted into the active Repeat2 run.

| Source | Downloaded pilot | Why it helps | Main qualification |
| --- | ---: | --- | --- |
| PAN 2020 fanfiction | 500 excerpts | Contemporary fanfiction; 491 benchmark author IDs and 411 fandoms in this sample | Cropped texts, not whole stories; no original work IDs |
| PAN 2019 fanfiction | 100 excerpts | Another dated authorship benchmark | Author IDs are local to each benchmark problem |
| AO3-2020 mirror | 2 chapter records from 2 works | Direct AO3 candidate with matched current metadata | Third-party snapshot provenance and reuse terms need more work; not a main training source |
| Beneath Ceaseless Skies | 200 whole stories, 162 bylined authors | Longer, professionally edited original fantasy, rather than prompted benchmark snippets | Only 9 stored versions are archived before 2023; 191 are current copies of older stories |
| StoriesInTheWild | 1,035 short stories | Controlled volunteer writing study, pinned to the authors' 2020 release | Only five image prompts; no stable author IDs |
| Standard Ebooks expansion | 12 whole books, 12 authors | Long prose across several genres, with pre-2023 edition commits | Historical language; only 12 independent works |

The strongest immediate scale-up route is **PAN 2020**. We downloaded its
official small training archive: 52,601 pairs yield 93,660 distinct exact text
excerpts before excluding one text with conflicting author labels. The pilot
samples 500. This gives a substantial accessible pool without relying on a
fresh, weakly dated web scrape. Preserve the benchmark's author IDs and fandoms,
deduplicate text, and audit author/work-related overlap before creating splits.
Do not count both appearances of a paired text as independent stories.

Source reports and primary links:

- [Fanfiction collection and PAN sources](fiction-fanfiction-v1.md).
- [Original online fiction and publisher/archive evidence](fiction-original-online-v1.md).
- [StoriesInTheWild controlled-writing release](fiction-stories-in-the-wild-v1.md).
- [Historical fiction, editions and Library of Congress expansion](fiction-historical-v1.md).

## How I would use these

1. **Expand contemporary fiction training from PAN 2020**, once the grouping
   and source-overlap checks have been applied to the larger pool. Keep an
   entire fandom/author holdout to test generalization. Original work IDs are
   missing, so author grouping and passage similarity remain necessary.
2. **Reserve StoriesInTheWild for an independent human false-positive audit.**
   Its small number of prompts makes random story splitting less informative.
   Keep this writing-study provenance separate from ordinary online bylines.
3. **Use the original stories and historical books as separate long-document
   strata.** Cap windows per author/work, retain coherent contiguous text, and
   report current-copy versus archived-version results separately for BCS.
   Thousands of book windows still represent only twelve books.
4. **Build corresponding AI fiction coverage before a balanced training mix.**
   Match genre, length, narrative form and prompt type across several generators;
   reserve prompts and source works before generation. Human-only data is useful
   for false-positive testing but cannot establish recall or ROC performance.
   The separately reserved generator-attribution corpus remains separate.

This expands the current creative sources—MAGE WritingPrompts/ROCStories and
EditLens WritingPrompts—with new source families. It also adds natural long
fiction, useful for sliding-window inference and later mixed-document work.
Randomly joining unrelated stories should not substitute for coherent prose.

## Checks and limits

`scripts/audit_fiction_candidates_v1.py` verified stored clean-text hashes,
recorded counts and lengths, and found **zero normalized exact duplicate groups
among the candidates**. It also found **zero exact or sampled 24-word matches**
against the diverse full train/validation/test files, frozen RAID evaluation,
and the existing Standard Ebooks calibration corpus. The sampling selects
approximately one in sixteen content hashes; this is not an exhaustive
near-duplicate or semantic audit. The manifest lists the precise reference
files and hashes. The historical acquisition additionally excludes the ten
previously used Standard Ebooks works and their authors by metadata.

The audit is saved as `fiction_candidates_v1/audit.json`. Acquisition backups,
including the first Wayback-only attempt, are excluded from these counts.
The reports distinguish dates, exact stored versions, bylines, benchmark labels,
and reuse terms. This is discriminative research, but that purpose alone does
not establish redistribution rights or an unrestricted license for every text.
Keep the source texts external and carry those distinctions into any later
public dataset/model release.

STORIUM is another promising large contemporary source, but its official
download requires login and a research data agreement; it was not acquired.
The historical report identifies a much larger Library of Congress collection,
which needs OCR and item-level selection work. Neither is included in the
downloaded counts above.
