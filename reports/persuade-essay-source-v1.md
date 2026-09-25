# PERSUADE student essay source v1

## Decision

Fetched the official PERSUADE 2.0 training CSV and exported **15,594 unique essays** to `/mnt/f/pangram-at-home/data/persuade_essays_v1/human_eval.parquet`. This is a source-family evaluation corpus for student argumentative writing and human false-positive measurement. It is **not included in training**. The source carries CC BY-NC-SA 4.0 terms, which do not fit the project's commercial-compatible training target. Keep the text outside Git and restrict access: contributors were students in grades 6–12.

The source repository says PERSUADE 1.0 formed the core of the Feedback Prize Kaggle competition in winter 2021–2022, and describes human annotated essays. The PERSUADE 2.0 README identifies the expanded corpus as over 25,000 US grade 6–12 student essays written to 15 prompts, with independent and source-based tasks. It explicitly notes demographic information, including disability status, and applies CC BY-NC-SA 4.0. This ingest uses the authors' official Google Drive training file linked from that repository; the file itself has 15,594 unique essay IDs. [PERSUADE 1.0 repository](https://github.com/scrosseye/PERSUADE_corpus), [PERSUADE 2.0 repository](https://github.com/scrosseye/persuade_corpus_2.0), [2022 corpus paper](https://doi.org/10.1016/j.asw.2022.100667).

## Ingest and privacy

Run `python scripts/fetch_essays_human.py`. It retrieves the CSV referenced by the official repository, deduplicates on essay ID, normalizes whitespace, retains full essay bodies, and writes Parquet plus a machine-readable manifest. The downloaded source CSV is temporary and deleted on successful export; use `--input-csv` only with the official file. The Parquet contains essay text, opaque corpus ID, source/domain, license, collection-period note, and hashes. It drops demographic fields, prompt/assignment metadata, scores, annotations, and source texts. It does not claim that free text is scrubbed of every personally identifying detail. Restrict access, do not publish rows or excerpts, honor removal requests, and do not use it for training.

Manifest: `/mnt/f/pangram-at-home/data/persuade_essays_v1/manifest.json`

| Check | Result |
| --- | ---: |
| Unique essay rows | 15,594 |
| Word count, minimum / median / maximum | 144 / 384 / 1,656 |
| Parquet SHA-256 | `34f9de3268cf01512f35516e86afbdb586b0446650f16733953eadbbd943f453` |
| Essay attributes retained | None |
| Source CSV retained | No |

## Audit sample

Spot-checked rows at offsets 0, 1, 42, 1,000, 7,788, and 15,593. All six had nonempty essay text, 212–1,055 words, and essay IDs. The six inspected samples were argumentative student prose; one used the literal `STUDENT_NAME` placeholder. A simple email-address pattern found no email in those six. This is a small formatting spot check, not a comprehensive PII screen or independent authorship verification. The competition provenance and pre-ChatGPT collection window provide strong human-origin evidence, but the corpus is narrow: school-aged argumentative essays on assigned prompts, not broad adult academic essays.

Use this as a dedicated FPR slice and report it separately from adult academic papers and other domains. Do not let any of these essay IDs or texts enter training or tuning. The per-essay records do not retain prompt identifiers, so split decisions must happen before joining this corpus with any other data, or keep it as one untouched external source-family evaluation.
