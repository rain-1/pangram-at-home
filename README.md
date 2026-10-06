# Pangram Workbench

A private, model-selectable classification service with a Pangram-style dashboard and report interface.

The local interface connects to the backend for text, URL and upload submission, history, reports, model settings and API keys. The default detector is **MELD v5 · Local evidence**, with sentence and token heatmaps. **EditLens · Qwen3 4B v3** remains selectable for passage-level editing scores. **Laya · Experimental contextual phrases** is available as a local zero-shot alternative, with surrounding paragraph context and Apple GPU acceleration; see [Laya setup and design](docs/LAYA.md). The original Llama adapter remains disabled pending access.

## Run the interface

Start the backend below, then in another terminal:

```sh
cd app
npm ci
npm run dev -- --host 127.0.0.1 --port 3000
```

Open http://localhost:3000 for Paper Atlas, the searchable paper library and PDF reading interface. The existing private workbench is at http://localhost:3000/workbench. “Try a sample text” previews an explicitly illustrative report without running a model. The Models & settings screen configures detectors.

The development server forwards `/backend` to the loopback backend and reads the owner key on the server; the key is never embedded in the browser bundle. Keep both servers bound to loopback. This automatic owner authorization is for a trusted local workspace only. Production deployment needs authenticated server-side routing; the static/Worker build alone does not include this local proxy.

## Run the backend

```sh
cd backend
uv sync --frozen --extra models
uv run python -m pangram_backend serve
```

Open http://127.0.0.1:8000/docs. The **Authorize** button accepts the owner key stored in `backend/.data/admin.key`; the key is generated on first start and is never printed by the service. Do not put keys in URLs or paste them into chat. The root page is a short service overview.

The installed environment can also run directly:

```sh
cd backend
.venv/bin/python -m pangram_backend serve
```

Default binding is loopback only. Requests to `/v1` require a scoped Bearer API key. This is a **single-workspace backend**: authorized keys share its scan history. It is not a multi-tenant SaaS or a replacement for Pangram's proprietary production service.

## Backend features

- Persistent model registry: select a default or select a model for each scan.
- Pinned MELD v5 baseline (1.58 GB) with full-document token evidence, sentence highlights and Unicode character offsets.
- Selectable, pinned EditLens Qwen3 4B v3 model (8.06 GB), running locally. Original Llama and RoBERTa EditLens adapters are also supported.
- Generic HTTP adapter for hosted text or image classifiers, with encrypted upstream credentials.
- Durable SQLite scan queue with atomic claims, worker leases, restart recovery, cancellation, explicit retries, and idempotent text submission.
- Immutable model/configuration snapshots for every scan. Changing a default doesn't change past reports.
- Text, public article URL, and TXT, Markdown, CSV, RTF, DOCX, and text-based PDF inputs.
- Batch text submissions and up to 100 document uploads, including per-file error reporting.
- Image uploads (PNG/JPEG/WebP) to a configured image classifier.
- Searchable/paginated history, notes, feedback, recoverable trash, usage counts, and audit events.
- JSON, CSV, and PDF report export; expiring, revocable report links.
- Read/scan API keys, revocation, rate limits, request limits, and SSRF-resistant URL fetching with pinned DNS resolution.
- Exact phrase matching against a private reference corpus, clearly labeled as corpus-only matching.
- Interactive OpenAPI documentation and automated integration tests.

## What is not active

- **Original Open Pangram Llama inference:** awaiting gated access. The requested Qwen EditLens model is active instead.
- **Image detection:** needs an external image-classification endpoint.
- **Internet-wide plagiarism search:** needs a separately licensed search/index provider; local corpus matching is not a substitute.
- **OCR, Gmail, Google Docs, browser extensions, and LMS integrations:** not connected or implemented. The HTTP API provides a base for future integrations.
- **Billing, enterprise SSO, organization roles and multi-tenancy:** not implemented.
- **Complete product parity:** the interface follows the inspected dashboard and report layout, but paid-only services and the integrations listed above are unavailable. Promotional controls explain their availability rather than simulating purchases.

See [backend setup and API guide](docs/BACKEND.md), [provider contract](docs/PROVIDERS.md), and [third-party attribution](docs/THIRD_PARTY.md).

## Verification

```sh
cd backend
uv run pytest -q
uv run ruff check pangram_backend tests
```

The tests use synthetic fixtures and a local HTTP test classifier. They do **not** establish detection quality or prove gated model inference works. No test-only classifier is shipped as a production provider.

## Local model and baseline data

Rain1's separate span-detection project (Qwen3-1.7B token models, v1–v14) is imported under [research/span-detection-20260928](research/span-detection-20260928/README.md). It covers calibration lessons, hard negatives and extra evaluation sources.


See [model serving](docs/PROVIDERS.md#local-qwen-editlens-v3-active) and [baseline inventory](research/README.md). The four collections include 100 ICLR 2023 papers, 100 ICLR 2026 papers, 100 historical books and 6,397 modern AI responses. These are research inputs, not a calibrated accuracy claim.

## Local PDF reading prototype

Open **PDF reader** in the sidebar, or http://localhost:3000/pdf-reader after starting both local servers. The reader lists downloaded PDFs under `research/data` (restart the backend to discover newly added files). Nothing is uploaded or classified by opening a PDF.

- Original PDF pages rendered locally with PDF.js, zoom, page overview and navigation.
- Saved ReviewBench reports linked by exact paper ID; papers with reports are marked with a dot and listed first. Choose a saved model result in the right panel.
- Click page highlights or passage entries to inspect the original classified text and score. Toggle highlights or switch to the full OCR source text.
- Unique normalized text matches map existing OCR segments onto the native PDF text layer. Short, changed, and ambiguous passages are **not** overlaid and remain in the “Not located” filter. The original report's PDF revision is not known, so this prototype does not certify version correspondence.
- Pages scroll continuously; only nearby pages are rendered, with page-sized placeholders preserving the document layout. Local byte-range responses allow partial PDF loading. Papers with saved reports currently read text from every page to build the alignment index; this may fetch most of that PDF. Production ingestion should precompute verified segment coordinates and bind them to a PDF hash.
- The local worker, fonts, character maps, and WASM assets are copied from the pinned PDF.js dependency by `predev`/`prebuild`; there is no external PDF rendering service or CDN.

Verification: `node --experimental-strip-types --test app/tests/pdf-alignment.test.mjs`, plus `backend/.venv/bin/python -m pytest backend/tests/test_pdf_reader.py` (run Python tests from `backend` instead if the package is not on the import path). The API retains the workspace's read authorization and restricts files to its PDF catalogue; the existing development proxy supplies local authorization.

## Paper Atlas website preview

The homepage is a responsive library backed by the existing authenticated local PDF catalogue. Search titles/IDs, filter by collection or result availability, sort, and paginate. Opening a paper uses a deep link (`/?paper=<catalogue-id>`) and displays the original PDF with available saved classifications. Browser Back and the collection button return to the library. About explains the source collection and model limitations.

The public research preview is deployed at https://pangram-paper-atlas.woog09.workers.dev using a separate, read-only Cloudflare build. The library automatically discovers PDFs under the bucket's public `papers/<sha256>.pdf` prefix, including uploads without extraction or classification results. Use **PDF only** to browse those papers, or select **ICLR 2027** from the collection menu. Inventory is refreshed on demand every minute per Worker instance and the browser refreshes each minute while visible. Unknown metadata falls back to the filename under Other uploads.

## Cloudflare deployment

All three resources are named `pangram-paper-atlas` in the same Cloudflare account:

- Worker: frontend static assets and read-only catalogue/PDF API; configuration in `app/wrangler.atlas.json`.
- R2 Standard bucket: deduplicated PDFs under `papers/<sha256>.pdf` and public report JSON under `results/<paper-id>.json`.
- D1 database: catalogue metadata, object keys, and result availability.

The standalone entry is `app/atlas-cloud/main.tsx`. It reuses the public library and reader without deploying the local workbench, Python service, owner key, or model configuration. PDFs are delivered through the Worker binding with byte-range support; the bucket does not need a public R2 URL or CORS configuration. Hosting uses the included workers.dev address. The existing unrelated Worker is untouched.

To update the published dataset from the saved local results (the backend need not be running):

```sh
backend/.venv/bin/python scripts/prepare_cloud_atlas.py
backend/.venv/bin/python scripts/upload_cloud_atlas.py
cd app
npx wrangler d1 execute pangram-paper-atlas --remote --file .sites-runtime/atlas/catalogue.sql
npm run deploy:atlas
```

The exporter reads the local database without starting inference, considers all classification selection manifests, and verifies PDF/input hashes. It retains previously published papers and adds new papers only once results are complete. The upload script uses the existing Wrangler login and checkpoints uploads locally. Run `cd app && npx wrangler whoami` first if the login token needs refreshing. Upload before importing changed catalogue records. The SQL upserts records; removing papers requires an explicit reviewed deletion. Frontend-only updates need only `cd app && npm run deploy:atlas`, after the initial export has prepared PDF.js assets.

To upload all downloaded PDFs independently of classification/publication:

```sh
cd app && npx wrangler whoami && cd ..
backend/.venv/bin/python scripts/upload_paper_pdfs.py
```

The archive index now contains 14,561 distinct PDFs, including the latest 6,834-paper batch. All PDFs in the latest batch were verified against R2 sizes and checksums. The uploader preserves older archive entries when local files have been removed and does not delete local copies.

This deduplicates PDFs under `papers/<sha256>.pdf`, skips objects whose remote size and MD5 already match, verifies every object after upload, and stores a conference/year index at `indexes/downloaded-papers.json`. The website now discovers these PDFs automatically, using this index for conference/year and filenames. Rerunning resumes from the verified R2 objects; local summaries are saved under `app/.sites-runtime/atlas/`.

This is a snapshot, not an automatic ICLR ingestion pipeline. Before the planned 60,000-paper release, implement server-side pagination/search in the UI, verified version-bound annotation coordinates, and load testing against the account's Workers/D1 quotas. Current cache headers allow browser caching; they do not implement a shared Worker API cache. PDF range requests each invoke the Worker and can issue R2 metadata/read operations. R2 free egress does not eliminate operation or compute usage charges. No paid Workers plan was activated by this deployment.

When retiring the site, remove only the `pangram-paper-atlas` Worker, its D1 database, and its R2 objects/bucket in Cloudflare. Export anything to retain first. Removing the Worker alone leaves stored data in R2. Review account subscriptions separately; no automatic shutdown has been scheduled.


### Automatic PDF discovery (October 3, 2026)

The Cloudflare reader merges the verified results manifest with a paginated R2 listing of the public `papers/` prefix. Published classifications take precedence and are never invented for new uploads. An unclassified PDF needs no report or extracted-text object: the API supplies an empty report list and the existing PDF.js reader opens the original file. Other bucket prefixes remain inaccessible. ETags cover the merged public inventory, so new uploads invalidate the browser catalogue even if results have not changed.

ICLR submission titles come from `indexes/iclr2027-pdfs.json`. Refresh that metadata independently with `backend/.venv/bin/python scripts/index_iclr2027_pdfs.py`; the regular PDF uploader and ICLR publisher also refresh it automatically. This command uploads only title/conference metadata, and the site includes an entry only once the corresponding PDF exists. It does not run inference, alter results, or remove files. Bucket custom metadata `title`, `filename`, and `collection` can describe other direct uploads.

### Research discovery browser (October 3, 2026)

The standalone Cloudflare homepage now defaults to an ICLR 2027 discovery browser. It offers research-area and keyword filters; search across titles, keywords, TL;DRs, abstract previews and submission numbers; title/update/submission sorting; a browser-local reading list; full abstracts and BibTeX; OpenReview links; related papers ranked by shared area/keywords; and next/previous navigation within filtered results. Filters and pagination survive shared URLs and browser history. The PDF reader opens in reading-only mode; the prior classification explorer is available at `/?view=classifications`.

The indexer reads locally archived public OpenReview notes, choosing latest mdate per ID and excluding restricted fields. `indexes/iclr2027-pdfs.json` remains lean for bucket discovery. Rich preview metadata is streamed through `/backend/v1/discovery` from `indexes/iclr2027-discovery.json` and merged in the browser, avoiding the Worker memory limit. Full abstracts/citations are in 256 metadata shards, served only for papers present in the public PDF catalogue through `/{id}/metadata`. Existing uploader/indexer hooks refresh these artifacts; no model calls are required.

Search currently covers metadata and abstract previews, not full PDF text. Anonymous submissions stay anonymous; no review ratings or acceptance decisions are inferred. Reading lists are local to each browser, not account-synced. The extracted text/position recovery archives are preserved separately and were not modified by this UI change.
