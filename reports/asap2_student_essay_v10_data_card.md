# ASAP 2.0 student essay pilot

The [original ASAP 2.0 repository](https://github.com/scrosseye/ASAP_2.0)
describes essays written by students in standardized state writing tests and
publishes the data under CC BY 4.0. The public training archive contains
17,307 essays. The authors state that the broader corpus includes essays also
found in PERSUADE 2.0, so overlap protection is essential for our locked
PERSUADE evaluation.

The local preparation discards student demographic fields. It rejects essays
shorter than 300 words and any essay sharing a normalized 24-word passage with
the full local PERSUADE corpus or the v9 training and validation corpus. The
result is 642 training-side candidates from two source-based assignments and
200 held-out human essays from two different assignments. The held-out
assignments are excluded from v10 training and hard-negative selection.

Only local research storage on F contains the student prose, source readings,
prompts, generations, and manifests:
`/mnt/f/pangram-at-home/data/asap2_student_essays_v10/`. They are not
redistributed with this repository. The final detector is discriminative, but
the training data still requires source attribution and careful handling.

The v10 pilot selects 256 training-side human essays using v9 false-positive
scores, with fixed quotas across the two prompts. AI companions are generated
from each essay's assignment and the original source reading, without showing
the student's essay. The goal is to teach the detector source-based student
prose without contaminating the protected PERSUADE test.

The exhaustive AI audit accepted 223 of 256 generated pairs (114 Qwen2.5-3B
and 109 SmolLM2-1.7B). It rejected long overlaps with protected human essays
or source readings, short outputs, incomplete endings, and meta responses.
Each accepted human/AI pair contributes one excerpt with exactly the same
Qwen3 source-token length. The resulting 20,000-document training mix assigns
2.53% of supervised-token positions to ASAP 2.0 essays, 3.61% to the earlier
paired science data, and 0.89% to LLMTrace; the largest single source is
DAMASHA at 26.2%. The class balance is 43.1% AI supervised tokens.
