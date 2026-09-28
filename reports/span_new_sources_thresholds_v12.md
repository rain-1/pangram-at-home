# v12 threshold transfer

Thresholds below are chosen only from the existing, disjoint pure-human
generic and student-essay calibration sets. The source-aware rule takes
their maximum at the same target rate. Locked tests are measured only
after the threshold is fixed; their labels do not set the cutoff.

| Model | Calibration target | Rule | PERSUADE alarms | Writers alarms | External human alarms | External AI-token recall | LLMTrace mixed AI-token recall / human-token FPR |
|---|---:|---|---:|---:|---:|---:|---:|
| v10 | 0.5% | generic | 0/3,000 | 3/579 | 9/150 | 90.1% | 23.0% / 0.28% |
| v10 | 0.5% | source aware max | 0/3,000 | 3/579 | 9/150 | 90.1% | 23.0% / 0.28% |
| v10 | 1.0% | generic | 0/3,000 | 4/579 | 11/150 | 93.6% | 31.2% / 0.46% |
| v10 | 1.0% | source aware max | 0/3,000 | 4/579 | 11/150 | 93.6% | 31.2% / 0.46% |
| v10 | 2.0% | generic | 2/3,000 | 7/579 | 16/150 | 95.9% | 38.6% / 0.76% |
| v10 | 2.0% | source aware max | 2/3,000 | 7/579 | 16/150 | 95.9% | 38.6% / 0.76% |
| v10 | 5.0% | generic | 15/3,000 | 19/579 | 22/150 | 98.6% | 51.7% / 2.1% |
| v10 | 5.0% | source aware max | 15/3,000 | 19/579 | 22/150 | 98.6% | 51.7% / 2.1% |
| v12 | 0.5% | generic | 2/3,000 | 5/579 | 9/150 | 96.4% | 34.2% / 0.63% |
| v12 | 0.5% | source aware max | 2/3,000 | 5/579 | 9/150 | 96.4% | 34.2% / 0.63% |
| v12 | 1.0% | generic | 8/3,000 | 9/579 | 10/150 | 97.4% | 40.5% / 0.96% |
| v12 | 1.0% | source aware max | 8/3,000 | 9/579 | 10/150 | 97.4% | 40.5% / 0.96% |
| v12 | 2.0% | generic | 31/3,000 | 14/579 | 11/150 | 98.9% | 51.8% / 2.1% |
| v12 | 2.0% | source aware max | 31/3,000 | 14/579 | 11/150 | 98.9% | 51.8% / 2.1% |
| v12 | 5.0% | generic | 240/3,000 | 45/579 | 22/150 | 99.9% | 68.5% / 7.9% |
| v12 | 5.0% | source aware max | 148/3,000 | 30/579 | 17/150 | 99.7% | 63.9% / 5.4% |

## Key operating points

| Model | Calibration | Broad human alarms | External human alarms | External AI-token recall | LLMTrace pure-AI documents caught | LLMTrace mixed AI-token recall / human-token FPR | New GRADTEX mixed AI-token recall / human-token FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| v10 | 2.0% | 9/3,579 | 16/150 | 95.9% | 479/516 | 38.6% / 0.76% | 44.5% / 0.18% |
| v12 | 0.5% | 7/3,579 | 9/150 | 96.4% | 457/516 | 34.2% / 0.63% | 46.2% / 0.02% |
| v12 | 1.0% | 17/3,579 | 10/150 | 97.4% | 468/516 | 40.5% / 0.96% | 53.6% / 0.08% |
| v12 | 2.0% | 45/3,579 | 11/150 | 98.9% | 483/516 | 51.8% / 2.1% | 66.5% / 0.38% |

The new-source holdout contains no pure AI documents; its mixed
recall/FPR and human alarms are included in the JSON report. A tighter
threshold changes the recall–FPR balance; it does not repair a lack of
discrimination on a source. Treat these as development operating points.
The source-aware maximum changes neither model at targets up to 2%;
the generic calibration cutoff is already stricter.

[Download operating-point chart](span_new_sources_thresholds_v12.pdf)
