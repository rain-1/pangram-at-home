# Load-bearing vocabulary transfer baseline

Louis Abraham's [load-bearing project](https://github.com/louisabraham/load-bearing) clusters GitHub pull request descriptions by vocabulary. Its [published detector](https://louisabraham.github.io/load-bearing/detector.html) asks whether text resembles the cluster that rose sharply in recent PRs. The author explicitly limits that result to **writing style**, rather than who wrote a text. We tested whether its frozen score transfers to our AI-versus-human task, without fitting or reversing it on our labels.

We pinned upstream commit [`45d3536`](https://github.com/louisabraham/load-bearing/commit/45d353684c82ec59342cbcd96a85829d22acfd31) (MIT license), stored `model.js` outside Git at `/mnt/f/pangram-at-home/models/load-bearing/model.js`, and verified SHA-256 `22a9f53fdefa7c812e481c6d96f1011624765413567b0c16cf6598246bc76f8d`. The score is the log odds of the arriving cluster against the other nine, exactly corresponding to the browser detector's cluster probability. Higher score means more like that PR-writing cluster. Our Python decoder matches the upstream weight decoder exactly; tokenization matched five upstream sample strings, and the published positive and negative example PRs receive the expected verdicts.

| Holdout | AUROC | AI detected | Human false positives |
| --- | ---: | ---: | ---: |
| Mixed test | 0.456 | 48/631 (7.6%) | 11/631 (1.7%) |
| Paper abstracts | 0.253 | 0/221 | 2/221 (0.9%) |
| General EditLens | 0.543 | 137/2,000 (6.9%) | 31/2,000 (1.6%) |
| Enron email | 0.296 | 0/1,800 | 2/1,800 (0.1%) |
| PMC-only abstracts | 0.235 | 0/87 | 1/87 (1.1%) |
| Swapped-generator papers | 0.283 | 0/182 | 1/182 (0.5%) |

Each threshold was selected on that dataset's validation humans for at most 2% empirical FPR, following the existing baseline convention. The Enron row uses the general EditLens validation threshold; swapped-generator papers use the paper validation threshold. These thresholds do not affect AUROC. Raw score caches are under `/mnt/f/pangram-at-home/runs/load_bearing_v1`, and the [metric JSON files](metrics) record source breakdowns. The [full results chart](charts/full_results.pdf) and [AUROC/ROC charts](charts/roc_curves.pdf) now include this baseline.

The score has high vocabulary coverage (median 87% of mixed-test tokens and 90% of paper-test tokens), so this is not simply an out-of-vocabulary failure. It detects a specific vocabulary associated with recent software PR descriptions. Scholarly abstracts, reviews, stories, and emails are different tasks. On paper abstracts, human text ranks *higher* on this PR-style score than generated text; reversing its direction after inspecting the test would create a different, test-selected baseline, so we do not present that as the published detector's performance. It does rank AI higher within the Reddit writing-prompt subset (0.847 AUROC on 107 human and 108 AI), which is a useful clue about domain dependence, not evidence of reliable authorship detection.

Reproduce with the pinned `model.js`:

```bash
mkdir -p /mnt/f/pangram-at-home/models/load-bearing
curl -L --fail https://raw.githubusercontent.com/louisabraham/load-bearing/45d353684c82ec59342cbcd96a85829d22acfd31/model.js -o /mnt/f/pangram-at-home/models/load-bearing/model.js
for dataset in mixed paper editlens pmc; do
  python scripts/run_load_bearing_baseline.py --dataset "$dataset"
done
python scripts/chart_auroc.py
python scripts/chart_full_results.py
```
