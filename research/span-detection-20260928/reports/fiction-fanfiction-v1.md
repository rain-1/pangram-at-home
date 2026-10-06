# Fanfiction human-text candidates (26 September 2026)

## Decision

The official [PAN 2020 authorship-verification archive](https://zenodo.org/records/5106099) is the strongest immediately available fanfiction source for a **private discriminative research pilot**. Its small training archive was released in 2020 and contains source-author pseudonyms and fandom labels. Those facts support, but do not prove, human origin for each excerpt. The small archive has 52,601 text pairs and 93,660 distinct exact eligible text excerpts after pair deduplication; one distinct text was associated with conflicting author IDs and must be excluded. These are roughly 21,000-character *excerpts*, not whole stories or chapters. The original work IDs and per-work publication dates are absent. A larger pilot can use this archive without another download, but all splits must be audited for near-duplicate/work leakage first.

The [PAN 2019 cross-domain attribution training archive](https://zenodo.org/records/20141612) is useful secondary evidence of different fanfiction style and fandoms. Its candidate-author IDs are **local to a problem**, so identical strings across problems must not be equated. The text is from a 2019 research benchmark, but per-work dates, original work IDs, and individual permissions are absent.

An AO3-labelled [third-party 2020 mirror](https://huggingface.co/datasets/ray0rf1re/AO3-2020) is not a clean large-scale route yet. A pinned 718,572-byte parquet shard contained 76 chapter-like rows from 29 story IDs, but this is not an official AO3 release and the uploader's license tag is not evidence of each author's permission. Public AO3 metadata supplied a pre-2023 publication date and author byline for two works. One metadata request returned unavailable, and the pilot stopped rather than retrying or bypassing access controls; 15 further selected work pages were not attempted. The resulting two chapter rows are provenance examples, not a meaningful sample. The shard's `idx` is an index, not a verified AO3 chapter ID.

## Frozen bounded pilot

The local artifact is `/mnt/f/pangram-at-home/data/fiction_candidates_v1/fanfiction/records.jsonl`, built by [fetch_fiction_fanfiction.py](../scripts/fetch_fiction_fanfiction.py). It has 602 rows and SHA-256 `1feb1b08227c40c7de6ac1e85853535544d86b748a91306058c3a1e413449eb4`. The external [manifest](/mnt/f/pangram-at-home/data/fiction_candidates_v1/fanfiction/manifest.json) records archive hashes, revisions, source counts, and selection rules. Every `clean_sha256` is the SHA-256 of the exact UTF-8 `text` stored in its row; `normalized_text_sha256` is separate. No train/validation/test split has been assigned.

| Source | Pilot rows | Origin groups | Other metadata | Median words |
| --- | ---: | ---: | --- | ---: |
| PAN20 official small | 500 | 491 archive-wide author IDs | 411 fandoms; 52,601 source pairs | 3,896 |
| PAN19 official English training | 100 | 42 problem-local author IDs | 66 fandoms; five English problems | 793 |
| AO3-labelled mirror, pinned shard | 2 | 2 AO3 byline IDs and work IDs | 2013 dated work pages; chapter index only | — |

The PAN20 selection is deterministic, caps an author at four texts and a fandom at thirty, and keeps up to three source pair IDs for provenance. Its 500 rows are intentionally broad in authors rather than a ready-made split. The PAN20 archive SHA-256 is `3922c36a91857355c40baaa0ac4e8beb3b1c5c3a7c2034d685f6824eac85233a`; its published MD5 agrees with the [official Zenodo file listing](https://zenodo.org/records/5106099). The AO3 shard is pinned to revision `5fa81ed59f6d75bd2aa8a593d56813316b1301c3`.

## How to use it

1. Complete overlap and near-duplicate checks against all detector train, validation, and test corpora. PAN20 pair reuse has already been removed by exact text hash, but near-duplicate excerpts can still derive from the same unidentified story. Do not split excerpts from the same story across partitions when a link is detectable.
2. For PAN20, use archive-wide author IDs as a minimum grouping key and cluster near-duplicate text before splitting; report any residual uncertainty because original work IDs are unavailable. For PAN19, keep each entire problem together unless a more reliable global author/work mapping can be established.
3. Use these as *human-only* fiction material: they can measure fiction false-positive rates and add style diversity. They do not measure AI recall. Build independently sourced, matched AI fiction examples before a balanced fiction evaluation, and keep prompt/work groups disjoint.
4. Record the difference between private research use of a published benchmark and permission to redistribute original stories or publish a model trained on them. The [AO3/OTW explanation of AI scraping](https://www.transformativeworks.org/ai-and-data-scraping-on-the-archive/) and [FanFiction.net terms](https://www.fanfiction.net/tos/) do not provide blanket author consent for model training. The PAN records do not specify a per-story license. Resolve this before use in a released model; do not treat a repository-level label as per-author permission.

The consent-oriented [FicSim](https://huggingface.co/datasets/ficsim/ficsim) corpus was considered, but its dataset terms expressly restrict training/fine-tuning; no gated copy was accessed. It could be examined later for a compatible *frozen-model evaluation* use, subject to its actual terms. Creative data already in this project includes MAGE and EditLens WritingPrompts; PAN fanfiction is a distinct source family, which is more valuable than just increasing the WritingPrompts row count.

## Reproduction and checks

Run `python scripts/fetch_fiction_fanfiction.py` with the external drive mounted and the Python dependencies installed. The script refuses to overwrite existing `records.jsonl`, verifies the official PAN20 archive MD5 on download, pins the AO3 mirror revision, limits live AO3 requests, and stops on unavailable metadata. `python -m py_compile scripts/fetch_fiction_fanfiction.py` passed. A full scan of the 602 JSONL rows confirmed each exact stored-text SHA-256 and the source counts above.
