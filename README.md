# Pangram at home: first data and baseline run

This repository builds a reproducible **research pilot** for human versus AI text detection. Text, model weights, and predictions live outside Git at `/mnt/f/pangram-at-home` by default. [The source shortlist](notes/datasets.md) explains provenance and rights decisions; [the first run report](reports/first-data-and-baselines.md) gives counts and results.

## Current corpora

| Corpus | Purpose | Size and rights |
| --- | --- | --- |
| [EditLens ICLR](https://huggingface.co/datasets/pangram/editlens_iclr) | General-text binary research baseline and two published reference checkpoints | 37,568 balanced train rows at the largest tier, plus frozen validation/test sets. **CC BY-NC-SA 4.0, noncommercial use.** AI-edited rows are retained in the raw download but excluded from the binary pyramid. |
| [PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) | Published medical/science human papers and locally generated topic-matched AI | The pilot fetched 963 pre-2023 full-text articles. Each selected JATS record states CC BY 4.0. Abstract pairs are filtered for length and journal concentration. |
| [ACL Anthology](https://aclanthology.org/faq/copyright/) | Human scholarly prose beyond biomedicine and a false-positive audit | 20,214 abstracts from official 2016–2022 metadata, under the Anthology's CC BY 4.0 policy. AI counterparts are generated locally for a selected subset. |

The full upstream PMC collection contains millions of articles and is larger than this pilot. No source here certifies human authorship sentence by sentence; a dated publisher or official proceedings record is the provenance rule. The Pangram data/model terms apply to any mixed research split, even though the PMC and ACL components are CC BY. Raw texts and gated assets are not committed.

## Reproduce

This run used Python 3.12, a 16 GB RTX 4080, and `/mnt/f` for storage. Install `requirements.txt` in an environment with a CUDA-capable PyTorch build for generation and reference model inference. Set `HF_TOKEN` or put `HF_API_TOKEN` in the ignored `.env`; gated Pangram and Llama access must already be granted by their owners.

```bash
export PANGRAM_DATA_ROOT=/mnt/f/pangram-at-home
python scripts/fetch_editlens.py
python scripts/build_editlens_pyramid.py
python scripts/fetch_models.py
python scripts/fetch_pmc_pilot.py --per-year 100
git clone --depth 1 --filter=blob:none --sparse https://github.com/acl-org/acl-anthology.git "$PANGRAM_DATA_ROOT/reference/acl-anthology"
git -C "$PANGRAM_DATA_ROOT/reference/acl-anthology" sparse-checkout set data
python scripts/extract_acl_abstracts.py
python scripts/generate_paper_ai.py --source pmc --model qwen
python scripts/generate_paper_ai.py --source acl --model smollm --limit 800
python scripts/build_pmc_pyramid.py
python scripts/build_paper_pyramid.py
python scripts/build_mixed_pyramid.py
python scripts/build_pmc_body_audit.py
python scripts/generate_paper_ai.py --source pmc --model smollm --paper-test-only
python scripts/generate_paper_ai.py --source acl --model qwen --paper-test-only
python scripts/build_cross_model_test.py
for name in editlens pmc paper mixed cross; do python scripts/verify_splits.py "$name"; done
```

Generation appends JSONL and resumes by source ID. The [frozen manifests](manifests) record source counts, rejected samples, exact parquet hashes, generator revisions, and split sizes. Whole papers stay in one split; the paper test also holds out journals or venues. Each binary split is 50% human and 50% AI; the mixed splits target 35% papers within each label. Small tiers are prefixes of larger tiers.

Run n-gram and embedding baselines, then the two Pangram reference checkpoints:

```bash
python scripts/run_baselines.py --dataset mixed --train-tier medium --model char
python scripts/run_baselines.py --dataset mixed --train-tier medium --model word
python scripts/run_baselines.py --dataset mixed --train-tier medium --model embedding
python scripts/run_editlens_reference.py --dataset paper --model roberta --tier full
python scripts/run_editlens_reference.py --dataset paper --model llama --tier full
```

The baseline scripts select a threshold on **validation humans** for at most 2% empirical false positives, then report test false-positive rate, true-positive rate, precision, ROC AUC, and counts. The target is a calibration rule, not a guarantee on new domains. Use `--audit-acl`, `--audit-pmc-body`, or `--cross-test` with the relevant data set to run the extra checks.

## First result

On the balanced paper test, the EditLens RoBERTa reference detected 203/221 AI abstracts with 2/221 human false positives. On the 35%-paper mixed test it detected 617/631 AI texts with 8/631 human false positives. The mixed character n-gram baseline detected 601/631 AI texts with 13/631 human false positives, including **8/87 PMC abstracts**. See [the report](reports/first-data-and-baselines.md) for all models, source breakdowns, the swapped-generator test, and the large human-only audit.
