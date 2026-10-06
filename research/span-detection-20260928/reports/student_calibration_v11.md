# Student essay calibration transfer

A separate 1,000-essay PERSUADE calibration set is disjoint by essay ID and exhaustive 24-word passage matching from the locked 3,000-essay test. Each source-aware threshold is the maximum of the generic-human and student-essay calibration cutoffs at the stated target. No locked-test labels set these cutoffs. This is an in-domain calibration experiment, not a guarantee for other student populations.

| Model | Calibration target | Rule | PERSUADE human alarms | CNN alarms | External human alarms | External AI-token recall | LLMTrace AI-token recall / FPR |
|---|---:|---|---:|---:|---:|---:|---:|
| v8 | 0.5% | generic only | 0/3,000 | 1/500 | 31/150 | 80.3% | 17.4% / 0.0% |
| v8 | 0.5% | source aware max | 0/3,000 | 1/500 | 31/150 | 80.3% | 17.4% / 0.0% |
| v8 | 1.0% | generic only | 0/3,000 | 1/500 | 49/150 | 88.4% | 32.0% / 0.0% |
| v8 | 1.0% | source aware max | 0/3,000 | 1/500 | 49/150 | 88.4% | 32.0% / 0.0% |
| v8 | 2.0% | generic only | 0/3,000 | 1/500 | 63/150 | 92.7% | 43.3% / 0.0% |
| v8 | 2.0% | source aware max | 0/3,000 | 1/500 | 63/150 | 92.7% | 43.3% / 0.0% |
| v8 | 5.0% | generic only | 2/3,000 | 2/500 | 77/150 | 97.4% | 62.0% / 0.3% |
| v8 | 5.0% | source aware max | 2/3,000 | 2/500 | 77/150 | 97.4% | 62.0% / 0.3% |
| v9 | 0.5% | generic only | 1/3,000 | 2/500 | 9/150 | 87.2% | 39.9% / 0.0% |
| v9 | 0.5% | source aware max | 1/3,000 | 2/500 | 9/150 | 87.2% | 39.9% / 0.0% |
| v9 | 1.0% | generic only | 8/3,000 | 4/500 | 15/150 | 93.4% | 55.5% / 0.1% |
| v9 | 1.0% | source aware max | 8/3,000 | 4/500 | 15/150 | 93.4% | 55.5% / 0.1% |
| v9 | 2.0% | generic only | 33/3,000 | 9/500 | 20/150 | 96.5% | 66.2% / 0.4% |
| v9 | 2.0% | source aware max | 33/3,000 | 9/500 | 20/150 | 96.5% | 66.2% / 0.4% |
| v9 | 5.0% | generic only | 239/3,000 | 29/500 | 39/150 | 99.5% | 81.2% / 2.9% |
| v9 | 5.0% | source aware max | 172/3,000 | 25/500 | 35/150 | 99.3% | 79.0% / 2.0% |
| v10 | 0.5% | generic only | 0/3,000 | 2/500 | 9/150 | 90.1% | 50.5% / 0.1% |
| v10 | 0.5% | source aware max | 0/3,000 | 2/500 | 9/150 | 90.1% | 50.5% / 0.1% |
| v10 | 1.0% | generic only | 0/3,000 | 2/500 | 11/150 | 93.6% | 58.6% / 0.2% |
| v10 | 1.0% | source aware max | 0/3,000 | 2/500 | 11/150 | 93.6% | 58.6% / 0.2% |
| v10 | 2.0% | generic only | 2/3,000 | 4/500 | 16/150 | 95.9% | 64.8% / 0.3% |
| v10 | 2.0% | source aware max | 2/3,000 | 4/500 | 16/150 | 95.9% | 64.8% / 0.3% |
| v10 | 5.0% | generic only | 15/3,000 | 7/500 | 22/150 | 98.6% | 74.1% / 1.0% |
| v10 | 5.0% | source aware max | 15/3,000 | 7/500 | 22/150 | 98.6% | 74.1% / 1.0% |

PERSUADE text and calibration files stay on local research storage; they are not added to model training or the Git repository. Older v8/v9 saved score exports were float32, so retrospective threshold tie counts can differ by one document from direct evaluation.

Download calibration chart (PDF on the `main` branch: `reports/student_calibration_v11.pdf`)
