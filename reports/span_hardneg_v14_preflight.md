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

The remote package is `artifacts/span_hardneg_v14_package.tar.gz` (159,446,515
bytes; SHA-256 `61dd1faff5fc819ff84d6bcf096e78505e188868f7f8b8cf72701c5a396a2d8d`).
Its 29 files were checked against the embedded manifest; it contains no API
keys or `.env` file. The controller trains Qwen3-1.7B Repeat2 for 3,352
optimizer steps with the previously selected LoRA settings, logs to W&B,
calibrates once on the existing independent human set at 2% document-any
false highlights, and automatically evaluates the same locked sets as v13.
Its export contains the adapter, per-set scores, reports, logs, and status.

## Launch status

**Starting on Vast.ai.** Instance 53167732 uses one RTX 3090 at
$0.1733/hour with a host driver supporting CUDA 12.8. The first instance
(53167175) was closed after a CUDA driver mismatch, before any data was
uploaded. A GPU preflight, package upload, training, and evaluation follow.
The prepared controller records each phase and exports the weights and scores
for local verification before the replacement instance is closed.

Files: `scripts/build_span_hardneg_v14.py`, `scripts/audit_span_hardneg_v14.py`,
`scripts/package_span_hardneg_v14.py`, `scripts/bootstrap_span_hardneg_v14.sh`,
`scripts/launch_span_hardneg_v14.sh`, and `scripts/run_span_hardneg_v14.py`.
