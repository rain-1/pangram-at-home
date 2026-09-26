# Open Pangram pure-document comparison

Open Pangram EditLens RoBERTa-large and Llama-3.2-3B are four-bucket models that score the extent of AI intervention. Their score is the expected bucket divided by three, following Pangram’s inference code. We average scores across overlapping native-token windows (512 for RoBERTa; 1024 for Llama). The checkpoints are licensed CC BY-NC-SA 4.0 for noncommercial use.

**Interpretation limit:** Our 20k run trained on 15,036 LLMTrace training records, and this benchmark uses 2,000 LLMTrace test records. The test is split-disjoint, with zero shared exact text hashes and zero shared LLMTrace group IDs against train and validation, but it is still the same source corpus and construction pipeline. Pangram’s published EditLens training corpus and objective differ. This comparison measures performance on an in-corpus holdout for our model, not general superiority across unseen sources.

Every model’s threshold is chosen on the same separate 1,120-document pure-human calibration set to allow at most 5% document-level false alarms. On our token model, a document is positive if any token is highlighted; on EditLens, its mean window score must cross the threshold. Both are then evaluated without retuning.

| Model | AI detected (516) | Human falsely flagged (720) | Pure-document AUROC | Locked human falsely flagged (3,579) |
| --- | ---: | ---: | ---: | ---: |
| Earlier v4 | 451/516 (87.4%) | 2/720 (0.28%) | 0.982 | 25/3,579 |
| 5k × 1 | 458/516 (88.8%) | 0/720 (0.00%) | 0.992 | 0/3,579 |
| 10k × 1 | 488/516 (94.6%) | 0/720 (0.00%) | 0.994 | 1/3,579 |
| 20k × 1 | 500/516 (96.9%) | 0/720 (0.00%) | 0.995 | 0/3,579 |
| 5k × 4 | 500/516 (96.9%) | 1/720 (0.14%) | 0.996 | 2/3,579 |
| EditLens RoBERTa | 423/516 (82.0%) | 10/720 (1.39%) | 0.951 | 11/3,579 |
| EditLens Llama | 486/516 (94.2%) | 19/720 (2.64%) | 0.983 | 11/3,579 |

The LLMTrace pure set is a different named corpus from the published EditLens training dataset. Source-level overlap has not been fully audited. The older synthetic evaluation may be closer to EditLens training data, so LLMTrace is the primary comparison here. The pure-document task is easier than localizing AI spans within mixed documents.

EditLens has no native token labels. This report does not compare EditLens against our token recall; that would require explicitly broadcasting its window scores to tokens and evaluating the resulting coarse spans.

## Additional shared synthetic pure-document result for EditLens

| Model | AI detected (148) | Human falsely flagged (152) | AUROC |
| --- | ---: | ---: | ---: |
| EditLens RoBERTa | 52/148 | 2/152 | 0.740 |
| EditLens Llama | 75/148 | 0/152 | 0.852 |

Checkpoint sources: [RoBERTa-large](https://huggingface.co/pangram/editlens_roberta-large), [Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B), [upstream inference](https://github.com/pangramlabs/EditLens/blob/05a588f15d792330ccaf91be8ee4fdb54ce26835/scripts/inference.py).
