# Severity grading comparison: heuristic vs learned

> **Every label compared below is heuristic-derived (configs/severity.yaml), not ground truth — CarDD has no severity annotation. This evaluation measures agreement with the heuristic, not real-world grading accuracy. See this script's module docstring, and autoassess/train/train_severity.py, for the full circularity discussion and what fixing it requires (real adjuster labels).**

Evaluated on 32 images. Learned model checkpoint: `runs/severity_smoke/weights/best.pt`.
Learned-vs-heuristic agreement (accuracy): 0.6562

## Confusion matrices

Rows = heuristic label (treated as "truth" for this comparison only — see circularity note above). Columns = predicted label.

### Heuristic vs itself (identity by construction — included as a visual anchor, not a result)

| true \ pred | minor | moderate | severe |
|---|---|---|---|
| minor | 7 | 0 | 0 |
| moderate | 0 | 17 | 0 |
| severe | 0 | 0 | 8 |

### Learned model vs heuristic

| true \ pred | minor | moderate | severe |
|---|---|---|---|
| minor | 3 | 2 | 2 |
| moderate | 1 | 13 | 3 |
| severe | 0 | 3 | 5 |

## Per-class agreement (learned vs heuristic)

| Class | Precision | Recall |
|---|---|---|
| minor | 0.750 | 0.429 |
| moderate | 0.722 | 0.765 |
| severe | 0.500 | 0.625 |

## What this does and doesn't show

- High agreement means the two-branch ResNet-18 has enough capacity to approximate the heuristic's closed-form area-weighted formula from pixels. That is a useful architecture sanity check.
- It is **not** evidence the heuristic (or the learned model) grades real damage severity correctly — there is no ground truth in this evaluation to check that against.
- Fix: evaluate both against a sample of adjuster-labeled or claim-payout-banded photos. See train_severity.py and this script's module docstrings.
