"""Compare the heuristic severity grader against the learned classifier — autoassess-severity-eval.

Loads a trained severity model's checkpoint (from `train_severity.py`),
re-runs both the heuristic (`autoassess.models.severity`) and the learned
model over the same validation split, and reports side-by-side confusion
matrices plus a written summary.

Read this before trusting any number this script prints
---------------------------------------------------------
**Every comparison here is against heuristic labels, because heuristic
labels are the only severity labels that exist in this project.** CarDD has
no severity annotation — no adjuster grading, no claim-payout band, nothing
independent of the heuristic formula in `configs/severity.yaml`.

That makes this evaluation structurally circular:

- The "heuristic" confusion matrix (heuristic label vs. heuristic label) is
  the identity matrix by construction — included only as a visual anchor
  next to the learned model's matrix, not as a real result.
- The "learned model" confusion matrix (heuristic label vs. learned-model
  prediction) measures how well a CNN reproduced the arithmetic formula it
  was trained to imitate. High agreement here means the two-branch
  architecture has enough capacity to approximate a closed-form function of
  mask areas and class weights from pixels — a reasonable thing to want to
  confirm — but it says **nothing** about whether minor/moderate/severe as
  graded by the heuristic correspond to real damage severity, and nothing
  about whether the learned model would generalize to a photo whose true
  severity the heuristic itself grades wrong (e.g. heavy hail damage spread
  as many small dents, which the area-fraction formula underweights because
  the padded union-bbox denominator grows with the damage spread).
- Any disagreement between heuristic and learned-model labels is *equally
  likely* to be a learned-model mistake or a case where imperfect detections
  (crops built from a detector's boxes, not ground truth, once this is wired
  to real inference) pushed the CNN toward a different answer than the exact
  polygon-area arithmetic the heuristic used. Nothing in this evaluation can
  tell those two apart.

What would fix this: a sample of real claim photos with severity assigned by
human adjusters (or, better, actual claim payout bands, which are what the
business ultimately cares about), large enough to check class balance across
minor/moderate/severe. That set should be held out and used to evaluate
*both* the heuristic and the learned model against real labels — at which
point this becomes a real evaluation instead of a self-consistency check.
Ideally the same real labels also replace heuristic labels as the learned
model's training signal, closing the loop properly instead of training a
network to imitate a formula we already have in closed form.

Usage
-----
    autoassess-severity-eval \\
        --weights runs/severity_v1/weights/best.pt \\
        --out-dir reports/figures/severity_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoassess.eval.coco_eval import load_class_names
from autoassess.models.severity import SEVERITY_LABELS, SeverityConfig
from autoassess.train.train_severity import (
    SeverityDataset,
    TwoBranchSeverityNet,
    evaluate,
    resolve_device,
)
from autoassess.utils.config import load_config, resolve_paths

CIRCULARITY_NOTE = (
    "Every label compared below is heuristic-derived (configs/severity.yaml), not "
    "ground truth — CarDD has no severity annotation. This evaluation measures "
    "agreement with the heuristic, not real-world grading accuracy. See this "
    "script's module docstring, and autoassess/train/train_severity.py, for the "
    "full circularity discussion and what fixing it requires (real adjuster labels)."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare heuristic vs learned severity grading (confusion matrices). "
                    "See module docstring: this compares against heuristic labels only."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"))
    p.add_argument("--severity-config", type=Path, default=Path("configs/severity.yaml"))
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to a severity model checkpoint (runs/<name>/weights/best.pt).")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="Which converted split to evaluate on (default: val).")
    p.add_argument("--imgsz", type=int, default=224,
                   help="Input size per branch — must match the checkpoint's training "
                        "imgsz (default: 224).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run evaluation on (default: cpu).")
    p.add_argument("--out-dir", type=Path, default=Path("reports/figures/severity_compare"),
                   help="Output directory for the confusion-matrix figure and report.")
    return p.parse_args()


def confusion_matrix(labels: list[int], predictions: list[int], n_classes: int) -> np.ndarray:
    """Rows = true (heuristic) label, columns = predicted label."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for label, pred in zip(labels, predictions, strict=True):
        cm[label, pred] += 1
    return cm


def build_markdown_report(
    heuristic_cm: np.ndarray,
    learned_cm: np.ndarray,
    learned_accuracy: float,
    n_images: int,
    weights_path: Path,
) -> str:
    lines = [
        "# Severity grading comparison: heuristic vs learned",
        "",
        "> **" + CIRCULARITY_NOTE + "**",
        "",
        f"Evaluated on {n_images} images. Learned model checkpoint: `{weights_path}`.",
        f"Learned-vs-heuristic agreement (accuracy): {learned_accuracy:.4f}",
        "",
        "## Confusion matrices",
        "",
        "Rows = heuristic label (treated as \"truth\" for this comparison only — "
        "see circularity note above). Columns = predicted label.",
        "",
        "### Heuristic vs itself (identity by construction — included as a visual "
        "anchor, not a result)",
        "",
        _cm_to_markdown_table(heuristic_cm, SEVERITY_LABELS),
        "",
        "### Learned model vs heuristic",
        "",
        _cm_to_markdown_table(learned_cm, SEVERITY_LABELS),
        "",
        "## Per-class agreement (learned vs heuristic)",
        "",
        "| Class | Precision | Recall |",
        "|---|---|---|",
    ]
    for i, label in enumerate(SEVERITY_LABELS):
        col_sum = learned_cm[:, i].sum()
        row_sum = learned_cm[i, :].sum()
        precision = learned_cm[i, i] / col_sum if col_sum else 0.0
        recall = learned_cm[i, i] / row_sum if row_sum else 0.0
        lines.append(f"| {label} | {precision:.3f} | {recall:.3f} |")

    lines += [
        "",
        "## What this does and doesn't show",
        "",
        "- High agreement means the two-branch ResNet-18 has enough capacity to "
        "approximate the heuristic's closed-form area-weighted formula from pixels. "
        "That is a useful architecture sanity check.",
        "- It is **not** evidence the heuristic (or the learned model) grades real "
        "damage severity correctly — there is no ground truth in this evaluation to "
        "check that against.",
        "- Fix: evaluate both against a sample of adjuster-labeled or claim-payout-"
        "banded photos. See train_severity.py and this script's module docstrings.",
    ]
    return "\n".join(lines) + "\n"


def _cm_to_markdown_table(cm: np.ndarray, labels: list[str]) -> str:
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [header, sep]
    for i, label in enumerate(labels):
        row = [str(cm[i, j]) for j in range(len(labels))]
        rows.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(rows)


def build_confusion_matrix_figure(
    heuristic_cm: np.ndarray, learned_cm: np.ndarray, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _plot_cm(axes[0], heuristic_cm, "Heuristic vs itself\n(identity — visual anchor only)")
    _plot_cm(axes[1], learned_cm, "Learned model vs heuristic\n(agreement, not accuracy)")

    fig.suptitle(
        "Severity grading: heuristic vs learned — both measured against heuristic "
        "labels (see report for circularity caveat)",
        fontsize=10.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cm(ax: Any, cm: np.ndarray, title: str) -> None:  # noqa: ANN401 — matplotlib Axes
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(SEVERITY_LABELS)))
    ax.set_yticks(range(len(SEVERITY_LABELS)))
    ax.set_xticklabels(SEVERITY_LABELS)
    ax.set_yticklabels(SEVERITY_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (heuristic)")
    ax.set_title(title, fontsize=9.5)

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_val * 0.6 else "#333333"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    class_names = load_class_names(args.dataset_config)
    severity_config = SeverityConfig.load(args.severity_config)

    processed_dir = Path(cfg.data.processed_dir) / "cardd"
    dataset = SeverityDataset(
        processed_dir / "images" / args.split, processed_dir / "labels" / args.split,
        class_names, severity_config, imgsz=args.imgsz,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No labels found for split '{args.split}' under {processed_dir}. "
            "Run autoassess-convert first."
        )

    device = resolve_device(args.device)
    model = TwoBranchSeverityNet().to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    result = evaluate(model, loader, device)

    label_to_index = {name: i for i, name in enumerate(SEVERITY_LABELS)}
    heuristic_labels = [
        label_to_index[dataset.severity_label_for_index(i)] for i in range(len(dataset))
    ]

    # sanity check: evaluate()'s "labels" (fed to the model as targets) must be the
    # same heuristic labels computed independently here, or the comparison is unsound
    assert result["labels"] == heuristic_labels, (
        "Heuristic labels computed by SeverityDataset during evaluation don't match "
        "an independent recomputation — this would silently invalidate the comparison."
    )

    heuristic_cm = confusion_matrix(heuristic_labels, heuristic_labels, len(SEVERITY_LABELS))
    learned_cm = confusion_matrix(heuristic_labels, result["predictions"], len(SEVERITY_LABELS))

    report = build_markdown_report(
        heuristic_cm, learned_cm, result["accuracy"], len(dataset), args.weights
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "severity_comparison.md"
    report_path.write_text(report, encoding="utf-8")

    figure_path = args.out_dir / "severity_confusion_matrices.png"
    build_confusion_matrix_figure(heuristic_cm, learned_cm, figure_path)

    raw_path = args.out_dir / "severity_comparison_raw.json"
    raw_path.write_text(json.dumps({
        "heuristic_confusion_matrix": heuristic_cm.tolist(),
        "learned_confusion_matrix": learned_cm.tolist(),
        "learned_accuracy_vs_heuristic": result["accuracy"],
        "labels": SEVERITY_LABELS,
        "n_images": len(dataset),
        "circularity_warning": CIRCULARITY_NOTE,
    }, indent=2), encoding="utf-8")

    print(report)
    print(f"Written: {report_path}")
    print(f"Written: {figure_path}")
    print(f"Written: {raw_path}")


if __name__ == "__main__":
    main()
