# Human source data shortlist

## Current direction — September 26, 2026

The project targets **diverse human and AI prose**, with papers as one important
domain. The paper-focused percentages below record the initial proposal; they
are not the current requirement. Creative fiction is a priority for expansion:
original online stories, fanfiction, web serials, and historical fiction offer
different writing styles and document lengths. Current creative training draws
from MAGE WritingPrompts/ROCStories and EditLens WritingPrompts, so new source
families matter more than simply adding more rows from the same benchmarks.

Keep work and author IDs where available and split before chunking. Retain
version-specific evidence for human-origin labels; dates and site membership
alone are insufficient. Evaluate human false positives by fiction source and
genre. Human-only fiction supports that audit, but matched AI fiction is also
needed to measure recall and train without source/label shortcuts. Record reuse
terms separately from authorship evidence. This is a discriminative project;
that purpose does not itself establish permissions for every source.

Research checked 2026-09-24. This is a source plan, not a downloaded or rights-cleared corpus. The goal is a paper-focused detector with human examples whose source, date, and reuse terms can be traced per document. A pre-2023 date is strong evidence of human authorship, especially for edited proceedings and journals, but is not absolute proof: AI writing tools existed before 2023, and downloaded copies may have been revised later.

## Mix to try

[Pangram 4's report](https://pangram-public.s3.us-east-1.amazonaws.com/pdf/pangram_4_technical_report.pdf) gives these approximate human-source shares: creative writing 22.2%, scientific/medical 18.0%, reference/educational 15.9%, consumer reviews 10.6%, social/Q&A/chat 10.4%, general web/mixed 8.2%, news 6.6%, essays/academic writing 4.9%, professional/finance 3.1%. Its source corpora are not named. Pangram says its training data is commercially licensed or owned by the company.

Our provisional human training target puts papers first. These are sampling targets, not estimates of available data:

| Category | Target |
| --- | ---: |
| Scientific and medical papers | 35% |
| Essays and academic writing | 10% |
| Reference and educational | 15% |
| Creative writing | 15% |
| Social, Q&A, and chat | 10% |
| Consumer reviews | 5% |
| News | 5% |
| General web and mixed | 3% |
| Professional and finance | 2% |

Make a separate paper-domain validation/test set. Report paper and non-paper false-positive rates separately; a broad overall score could hide failures on papers. Do not fill a category with unclear-rights data just to hit its target.

## Best sources to ingest first

| Source | What to use | Provenance and rights | Practical path |
| --- | --- | --- | --- |
| [PMC Open Access article datasets](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) | Published journal article body/abstract, publication date through 2022; start with biomedical papers | NLM supplies article/version metadata and a **per-article license**. Start with CC0 or CC BY. Exclude preprints, manuscripts, noncommercial terms, and articles whose license cannot be verified. Keep PMCID, DOI, version, journal, date, license, attribution, and source URL. | Use NLM's current [`pmc-oa-opendata` S3 bucket and inventory](https://pmc-oa-opendata.s3.amazonaws.com/README.txt), then its per-version JSON metadata and JATS XML. **Legacy PMC FTP/cloud article files were removed in August 2026**; old `oa_comm` directory recipes are stale. |
| [PLOS articles](https://api.plos.org/text-and-data-mining.html) | Pre-2023 journal article body/abstract across PLOS titles | [PLOS terms](https://plos.org/terms-of-use/) say articles are generally CC BY or similarly reusable unless indicated otherwise. Check each article's license and provenance. Likely overlaps PMC, so dedupe by DOI. | PLOS offers article metadata and JATS XML for text mining; use publication dates and item rights. |
| [ACL Anthology](https://aclanthology.org/) | 2016–2022 published conference/journal papers | The [Anthology copyright FAQ](https://aclanthology.org/faq/copyright/) says materials from 2016 onward are CC BY 4.0. Use the publisher version, retain Anthology ID, venue/year, author and license details. Narrow topical coverage: computational linguistics. | Use [official metadata/repository](https://github.com/acl-org/acl-anthology) to select papers, then acquire bounded PDF batches. PDF extraction needs paragraph/header/reference cleanup. |
| [Standard Ebooks](https://standardebooks.org/about) | Books and essays published long before generative AI | Volunteer-edited editions of works believed to be US public domain; [selection policy](https://standardebooks.org/contribute/collections-policy). Strong human-origin source for creative prose. Check individual work/edition and target-jurisdiction rights. | Download selected ebooks, strip front matter/navigation, keep work and edition IDs, and cap any one author. |
| [Project Gutenberg](https://www.gutenberg.org/policy/license) | Public-domain books, essays, historical reference text | Many works are US public domain, but **not every ebook is**; inspect each item. Strip Project Gutenberg branding/license material where its policy requires. Jurisdiction matters. | Use its sanctioned bulk/mirror access, not automated traffic against the main website. |

PMC + PLOS provide breadth within biomedical science; ACL adds a different scholarly style but is a narrow field. Sample by journal/venue/year rather than drawing random chunks from the largest collection. The current PMC S3 metadata may be updated without a new version number, so a publication date alone does not prove that every byte in the retrieved copy predates 2023. Store hashes and check article history where possible.

## Conditional sources and why they are conditional

| Source | Value | Condition before use |
| --- | --- | --- |
| [arXiv bulk data](https://info.arxiv.org/help/bulk_data.html) | Large, diverse science and mathematics preprints | Filter each paper's license and submission **version** and date. arXiv's default distribution license is not a blanket commercial reuse grant. Treat preprints as a separate stratum from published papers; LaTeX/PDF extraction is noisy. |
| [Stack Exchange historical dumps](https://meta.stackexchange.com/questions/224873/all-stack-exchange-data-dump-releases) | Large 2022-or-earlier Q&A corpus | Post-era CC BY-SA terms, attribution and share-alike obligations need an implementation decision; filter bots, edits after cutoff, and personal information. A 2022 snapshot bounds existence but does not verify each poster. |
| [PERSUADE v1](https://github.com/scrosseye/PERSUADE_corpus) | 2021–22 student argumentative essays with excellent human provenance | CC BY-NC-SA 4.0; useful for noncommercial research/evaluation, **not** in a commercial-compatible training pool without permission. Keep v1 separate from later releases. |
| [OpenStax textbooks](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks) | Human-edited educational prose | Current library guidance says CC BY-NC-SA. Older **individual** books can say CC BY, so resolve license per edition and any AI-use conditions before inclusion. |
| [Chronicling America](https://www.loc.gov/collections/chronicling-america/about-this-collection/rights-and-access/) | Historical US news; [official OCR bulk](https://chroniclingamerica.loc.gov/ocr/) | Old works may be public domain; OCR and article segmentation are noisy, and issue-level rights need checking. A possible small news slice. |
| [Enron email](https://www.cs.cmu.edu/~enron/) | Authentic pre-LLM professional email | Sensitive personal and business content; resolve rights and privacy filtering before use. Narrow author population. |
| [PubMed baseline](https://pubmed.ncbi.nlm.nih.gov/download/) | Huge dated scientific abstract pool | Citation access does **not** grant blanket abstract reuse. Use only records with independently verified item rights; short abstracts also behave differently from full papers. |
| [Amazon review data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html), [Yelp Open Dataset](https://www.yelp.com/dataset), [AG News](https://zenodo.org/records/7555424), Reddit/Pushshift | Potential reviews, news, and social text | Access or downstream reuse terms are unclear or restricted; published date does not verify that each review/post is authentic human text. Do not place these in the first training pool. |
| [S2ORC](https://github.com/allenai/s2orc) | Broad structured scholarly text | Useful for discovery, but underlying paper rights vary and the original release is no longer supported. Reacquire from licensed publishers rather than treating its dataset license as a license for every paper. |

## Existing Hugging Face links

| Dataset | Decision |
| --- | --- |
| [Samarth0710/reviewbench](https://huggingface.co/datasets/Samarth0710/reviewbench) | Discovery/evaluation candidate only. It has ~51,500 papers from 2020–2026, OCR full text plus reviews/rebuttals. Its metadata says CC BY, but its [own license section](https://huggingface.co/datasets/Samarth0710/reviewbench#license) says underlying texts remain authors' IP and describes **noncommercial research** redistribution. The whole dataset is neither pre-2023 nor rights-cleared for this training plan. If used in research, filter years and fields explicitly. |
| [woog/arena-prose-100-49-models](https://huggingface.co/datasets/woog/arena-prose-100-49-models) | **AI evaluation data, not human data.** Its 5,000 responses are model generated; the human-origin prompts are not human answer examples. Source and model-output terms are mixed. Group any evaluation split by prompt. |

Hugging Face credentials in `.env` are for authenticated access when needed; the first-priority sources above are public. A Hugging Face card's top-level license tag is not enough to establish rights to every underlying text.

## Acceptance rule for a human positive

1. Preserve `source`, `source_id`, `canonical_url`, `original_publication_date`, `retrieved_at`, `source_version`, `license`, `attribution`, `raw_sha256`, `clean_sha256`, `genre`, and `extraction_method` for each document. Record the source URL of rights evidence.
2. Require an independently dated source published no later than 2022-12-31, and prefer older material when available. For strict evaluation positives, prioritize pre-2020 edited publications, public-domain classics, or similarly well-documented human work. Flag later revisions and uncertain authorship rather than silently labeling them human.
3. Keep only body prose intended for readers. Remove references, tables, boilerplate, quotations of other works, supplementary files, OCR garbage, and markup; do not let extracted paper text mix with reviews or comments.
4. Resolve rights at the **item/edition/version** level. Record attribution for CC BY. Use CC0, public-domain (in relevant jurisdictions), and clearly CC BY items for the first training pass; quarantine NC, SA, ND, unclear, and platform-restricted material for separate review.
5. Deduplicate by DOI/PMCID/Anthology ID where available and by normalized text/near-duplicate matching across sources. Split by whole work before chunking, and keep journal/venue/author or source-family holdouts to expose leakage and domain shift.
6. Store raw and cleaned text outside Git. Commit manifests, parsers, rights rules, and small non-sensitive fixtures only. Preserve a removal path keyed by source ID and hash.

This is deliberately a **candidate list**, not a claim that any source certifies every sentence as human-written. The first pilot should build a modest licensed PMC + PLOS + ACL sample, audit a random set of raw/clean pairs and license records, and only then scale ingestion.
