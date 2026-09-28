# Essay/forum hard-negative pilot (v14): preflight

V14 retains all 20,000 v13 documents and adds 1,200 documents. The added human
texts are 400 PERSUADE 2.0 essays from the 2021–2022 competition and 200
pre-2023 Writers Stack Exchange posts. The added AI texts are 400 ChangeMyView
and 200 ELI5 open-ended responses from the MAGE training split, balanced across
GPT-3.5, text-davinci-003, and text-davinci-002. This is a **small data pilot**,
not a guarantee of lower false positives.

| Check | Result |
|---|---:|
| Training documents | 21,200 |
| Pure human / mixed / pure AI | 7,025 / 7,663 / 6,512 |
| Training windows (512 tokens, stride 256) | 26,810 |
| AI-labeled supervised-token share | 42.31% |
| Largest single-source supervised-token share (DAMASHA) | 23.00% |
| PERSUADE / Writers supervised-token share | 1.92% / 0.54% |
| LLMTrace supervised-token share | 0.87% |

The builder excludes essay source IDs and Writers thread/owner connected
components already present in protected calibration or evaluation files. It
rejects exact normalized text and sampled 24-word phrase overlap with v13
training/validation and eight locked evaluations. An additional check found
zero new-row overlaps with the older realistic, v4 validation, v3 validation,
and diverse-pyramid validation/test files. The new human corpus is limited to
the project's noncommercial research use; PERSUADE carries CC BY-NC-SA 4.0,
and Writers posts retain their per-post CC BY-SA attribution metadata.

The remote package is `artifacts/span_hardneg_v14_package.tar.gz` (159,446,549
bytes; SHA-256 `0dd30745fb4c01874be1c65a215203cb992075a2707b47c8b4af42c26426f29b`).
Its 29 files were checked against the embedded manifest; it contains no API
keys or `.env` file. The controller trains Qwen3-1.7B Repeat2 for 3,352
optimizer steps with the previously selected LoRA settings, logs to W&B,
calibrates once on the existing independent human set at 2% document-any
false highlights, and automatically evaluates the same locked sets as v13.
Its export contains the adapter, per-set scores, reports, logs, and status.

## Launch status

**Prepared, not launched.** The current session has
`CODEX_SANDBOX_NETWORK_DISABLED`; the Vast CLI fails DNS resolution for
`console.vast.ai`, and no Vast connector is available. The external drive and
Git metadata are read-only in this session, so outputs are in the repository's
ignored `data/` and `artifacts/` folders. The run requires a network-enabled
session to search current offers, rent a GPU, upload the package and runtime
credentials, monitor the controller, retrieve/verify its export, and close the
instance. The last recorded Vast 4090 rental was about $0.42/hour; this is
historical, not a current offer or cost estimate.

Files: `scripts/build_span_hardneg_v14.py`, `scripts/audit_span_hardneg_v14.py`,
`scripts/package_span_hardneg_v14.py`, `scripts/bootstrap_span_hardneg_v14.sh`,
`scripts/launch_span_hardneg_v14.sh`, and `scripts/run_span_hardneg_v14.py`.
