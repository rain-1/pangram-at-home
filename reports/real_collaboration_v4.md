# Real collaboration: where the v4 model succeeds and fails

All figures below are from cached scores. The frozen operating threshold was selected on a separate pure-human calibration set; neither benchmark changed model weights or threshold.

| Measure | CoAuthor | AITDNA |
| --- | ---: | ---: |
| Documents | 119 | 362 |
| Mixed documents | 119 | 258 |
| Writer groups | 25 | 99 |
| Source tokens with usable labels | 51.3% | 97.8% |
| AI share of labeled tokens in mixed documents | 22.4% | 72.3% |
| AI token recall in mixed documents | 3.3% | 88.4% |
| Human token false positive rate in mixed documents | 0.4% | 18.0% |
| Token AUROC in mixed documents | 0.676 | 0.918 |
| Mixed documents with any AI token detected | 7/119 | 236/258 |
| Mixed documents with no AI token detected | 112/119 | 22/258 |

The frozen threshold gives CoAuthor 3.3% AI-token recall (233/7,026) and 0.35% human-token FPR (86/24,306). AITDNA's corresponding overall numbers are 88.4% (86,520/97,824) and 9.3% (6,974/75,094). AITDNA contains 103 pure-human documents, of which one has a false highlight. CoAuthor contains no pure-human documents in this evaluation.

Within CoAuthor, argumentative writing has 0/1,777 AI tokens detected across 36 documents; creative writing has 233/5,249 (4.4%) across 83. These are sessions from 25 writers, so document counts should not be treated as independent writer counts.

## Ranking at matched false positive rates

These are **retrospective** points on each benchmark's mixed documents, with a separate threshold for each benchmark. They show score separation, not deployable calibration or a single common operating point.

| Within-benchmark human-token FPR | CoAuthor AI recall | AITDNA AI recall |
| ---: | ---: | ---: |
| 1.0% | 9.8% | 16.0% |
| 2.0% | 13.8% | 28.5% |
| 5.0% | 21.7% | 56.1% |
| 10.0% | 31.8% | 76.3% |

## AI span length

The length figures describe the source's surviving AI-character spans. Token recall weights long spans much more heavily than short ones.

| Span length | CoAuthor AI spans | CoAuthor share of AI characters | CoAuthor token recall | AITDNA AI spans | AITDNA share of AI characters | AITDNA token recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <20 characters | 25 | 1.0% | 6.0% | 2767 | 2.2% | 31.7% |
| 20–79 characters | 202 | 33.1% | 0.8% | 553 | 4.3% | 49.9% |
| 80–319 characters | 176 | 58.5% | 2.0% | 382 | 12.0% | 70.4% |
| 320+ characters | 4 | 7.4% | 21.9% | 471 | 81.5% | 93.9% |

Some very short character spans contain no complete model token and cannot contribute to the span-bin token recall. The JSON records the number of scorable spans in each bin.

## Interpretation

CoAuthor is an observed GPT-3 era collaboration corpus with relatively short surviving AI inserts. AITDNA uses recent generators and contains long AI passages alongside tiny edits. In AITDNA, 81.5% of AI characters are in spans of at least 320 characters; in CoAuthor, only 7.4% are. That difference, along with generator, task, participant and provenance differences, plausibly contributes to the large recall gap. It is not a controlled causal test.

The mixed-document AUROC gap (0.676 versus 0.918) persists across thresholds. Therefore CoAuthor's low recall cannot be explained solely by choosing a conservative threshold. The model's ranking of short, integrated AI inserts is substantially weaker on this benchmark. Neither benchmark alone estimates broad real-world accuracy.

[ROC and span-length chart](real_collaboration_v4.pdf). Source descriptions: [CoAuthor](https://coauthor.stanford.edu/) and [AITDNA](https://huggingface.co/datasets/UKPLab/AITDNA).
