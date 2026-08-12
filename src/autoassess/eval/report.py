"""Report-generation helpers — autoassess-report.

Produces the artifacts a project report needs from a trained YOLOv8-seg
model's predictions on its own test split, all built on the same
ground-truth reconstruction (`autoassess.eval.coco_eval.build_coco_ground_truth`)
and detection format used everywhere else in this project:

  - a qualitative results grid (N test images, predictions overlaid)
  - a failure-case panel (worst images by per-image mean mask IoU,
    predictions vs. ground truth side by side, plus a written analysis)
  - per-class precision-recall curves (from pycocotools' own PR arrays,
    not a hand-rolled curve, so they agree with the mAP numbers elsewhere)

Every figure here is real model output on real images — there is no
synthetic-data path in this module. If a caller points it at a model/data
pair that doesn't exist, it fails loudly rather than substituting anything.

Usage
-----
    autoassess-report \\
        --weights runs/yolov8_seg_v1/weights/best.pt \\
        --dataset-config configs/cardd.yaml \\
        --processed-dir data/processed/cardd \\
        --split test \\
        --out-dir reports/figures/cardd
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from autoassess.eval.coco_eval import build_coco_ground_truth, load_class_names
from autoassess.eval.yolo_eval import polygon_xy_to_rle
from autoassess.utils.logging import get_logger

log = get_logger(__name__)

PRED_COLOR = (66, 133, 244)  # blue — matches dataviz slot 1 used in compare.py
GT_COLOR = (52, 168, 83)  # green — distinct from prediction blue and severity red/orange
QUALITATIVE_DPI = 300
FAILURE_DPI = 300


@dataclass
class ImagePrediction:
    """One test image's model output, at original image resolution."""

    file_name: str
    image_path: Path
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    mask_rles: list[dict[str, Any]] = field(default_factory=list)


def run_predictions(
    weights_path: Path, images_dir: Path, conf: float = 0.25, device: str = "cpu"
) -> dict[str, ImagePrediction]:
    """Run a trained YOLOv8-seg checkpoint once over every image in
    `images_dir`, keyed by file name. Loads the model a single time
    (unlike `associate.run_yolo_seg_inference`, which reloads per call —
    fine for one-off CLI use, wasteful across a whole test split)."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model: Any = YOLO(str(weights_path))
    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    predictions: dict[str, ImagePrediction] = {}
    for image_path in image_paths:
        result = model.predict(str(image_path), conf=conf, device=device, verbose=False)[0]
        height, width = result.orig_shape
        pred = ImagePrediction(
            file_name=image_path.name, image_path=image_path, width=width, height=height
        )
        if result.masks is not None:
            class_names_lookup = result.names
            for poly_xy, cls, box_conf in zip(
                result.masks.xy, result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
            ):
                if poly_xy.shape[0] < 3:
                    continue
                x0, y0 = poly_xy[:, 0].min(), poly_xy[:, 1].min()
                x1, y1 = poly_xy[:, 0].max(), poly_xy[:, 1].max()
                pred.boxes.append((float(x0), float(y0), float(x1), float(y1)))
                pred.class_names.append(class_names_lookup[int(cls)])
                pred.confidences.append(float(box_conf))
                pred.mask_rles.append(polygon_xy_to_rle(poly_xy, width, height))
        predictions[image_path.name] = pred

    return predictions


def _decode_masks(rles: list[dict[str, Any]]) -> list[np.ndarray]:
    from pycocotools import mask as mask_utils

    return [mask_utils.decode(rle).astype(bool) for rle in rles]


def draw_predictions(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    class_names: list[str],
    mask_rles: list[dict[str, Any]],
    color: tuple[int, int, int],
    confidences: list[float] | None = None,
) -> Image.Image:
    """Overlay filled masks + labeled boxes for one set of instances (all
    the same color — caller picks prediction vs. ground-truth color)."""
    from PIL import ImageDraw

    masks = _decode_masks(mask_rles)
    fill = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    for mask in masks:
        fill[mask] = (*color, 100)
    overlay = Image.fromarray(fill, "RGBA")
    draw = ImageDraw.Draw(overlay)

    for i, (box, name) in enumerate(zip(boxes, class_names, strict=True)):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=2)
        label = name if confidences is None else f"{name} {confidences[i]:.2f}"
        draw.text((x0 + 2, max(y0 - 12, 0)), label, fill=(*color, 255))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def build_qualitative_grid(
    predictions: dict[str, ImagePrediction],
    image_names: list[str],
    out_path: Path,
    n_cols: int = 4,
    title: str = "Qualitative results",
) -> None:
    """A matplotlib grid of `image_names` (in the given order — caller
    picks which are "easy/medium/hard") with predictions overlaid,
    saved at publication DPI."""
    n = len(image_names)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, name in zip(axes_flat, image_names, strict=False):
        pred = predictions[name]
        with Image.open(pred.image_path) as im:
            overlaid = draw_predictions(
                im, pred.boxes, pred.class_names, pred.mask_rles, PRED_COLOR, pred.confidences
            )
        ax.imshow(overlaid)
        ax.set_title(f"{name}\n{len(pred.boxes)} instance(s)", fontsize=8)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=QUALITATIVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote qualitative grid (%d images) -> %s", n, out_path)


def per_image_mask_iou(
    coco_gt_dict: dict[str, Any], predictions: dict[str, ImagePrediction], iou_thr: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Per-image mean matched-mask IoU (same greedy same-class matcher as
    `autoassess.eval.metrics.mask_iou_mean`, scoped per image instead of
    aggregated across the dataset), plus unmatched GT/DT counts.

    An image with predictions but zero GT-IoU matches gets score 0.0 (not
    skipped) — that is exactly the kind of image a failure-case section
    should surface, not silently drop.
    """
    from pycocotools import mask as mask_utils

    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for ann in coco_gt_dict["annotations"]:
        img = next(i for i in coco_gt_dict["images"] if i["id"] == ann["image_id"])
        gt_by_image.setdefault(img["file_name"], []).append(ann)
    category_id_by_name = {c["name"]: c["id"] for c in coco_gt_dict["categories"]}

    results: dict[str, dict[str, Any]] = {}
    for file_name, pred in predictions.items():
        gt_anns = gt_by_image.get(file_name, [])
        n_gt, n_dt = len(gt_anns), len(pred.mask_rles)

        if n_gt == 0:
            # No ground truth to score against — this is a data-labeling gap, not a
            # model failure, whether or not the model predicted anything. Scoring it
            # 0.0 would rank it as a "worst failure" even when the prediction is
            # plausibly correct on an unlabeled image; mean_iou=None marks it as
            # unscoreable so find_worst_images excludes it from the failure ranking.
            results[file_name] = {"mean_iou": None, "n_gt": 0, "n_dt": n_dt, "n_matched": 0}
            continue
        if n_dt == 0:
            # Real ground truth exists and the model predicted nothing: a genuine miss.
            results[file_name] = {"mean_iou": 0.0, "n_gt": n_gt, "n_dt": 0, "n_matched": 0}
            continue

        gt_rles = [a["segmentation"] for a in gt_anns]
        gt_cats = [a["category_id"] for a in gt_anns]
        dt_cats = [category_id_by_name.get(name, -1) for name in pred.class_names]
        iscrowd = [0] * n_gt
        iou_matrix = mask_utils.iou(pred.mask_rles, gt_rles, iscrowd)

        matched_gt: set[int] = set()
        ious: list[float] = []
        # sort predictions by confidence, highest first, mirroring COCOeval's own matching order
        order = sorted(range(n_dt), key=lambda i: -pred.confidences[i])
        for dt_idx in order:
            best_gt, best_iou = -1, iou_thr
            for gt_idx in range(n_gt):
                if gt_idx in matched_gt or dt_cats[dt_idx] != gt_cats[gt_idx]:
                    continue
                if iou_matrix[dt_idx, gt_idx] > best_iou:
                    best_iou, best_gt = iou_matrix[dt_idx, gt_idx], gt_idx
            if best_gt >= 0:
                matched_gt.add(best_gt)
                ious.append(float(best_iou))

        mean_iou = statistics.mean(ious) if ious else 0.0
        results[file_name] = {
            "mean_iou": mean_iou, "n_gt": n_gt, "n_dt": n_dt, "n_matched": len(ious),
        }

    return results


def find_worst_images(
    iou_scores: dict[str, dict[str, Any]], n: int = 10
) -> list[tuple[str, dict[str, Any]]]:
    """Worst `n` images by mean mask IoU, restricted to images that have
    real ground truth to score against (`mean_iou is not None` — see
    `per_image_mask_iou`). A zero-GT image is a labeling gap, not a model
    failure, and would otherwise dominate the ranking with score 0.0
    regardless of whether the model's predictions were any good — use
    `find_unscoreable_images` to surface those separately."""
    scored = [(name, s) for name, s in iou_scores.items() if s["mean_iou"] is not None]
    scored.sort(key=lambda pair: pair[1]["mean_iou"])
    return scored[:n]


def find_unscoreable_images(
    iou_scores: dict[str, dict[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """Images with zero ground-truth instances after dataset conversion —
    a data-labeling gap worth reporting on its own, kept separate from
    `find_worst_images` so it isn't misread as model failure."""
    return [(name, s) for name, s in iou_scores.items() if s["mean_iou"] is None]


def build_failure_case_figure(
    predictions: dict[str, ImagePrediction],
    coco_gt_dict: dict[str, Any],
    worst: list[tuple[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    """Predictions (blue) vs. ground truth (green) side by side, one row
    per failure case, worst first."""
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for ann in coco_gt_dict["annotations"]:
        img = next(i for i in coco_gt_dict["images"] if i["id"] == ann["image_id"])
        gt_by_image.setdefault(img["file_name"], []).append(ann)
    id_to_name = {c["id"]: c["name"] for c in coco_gt_dict["categories"]}

    n = len(worst)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n))
    axes = np.atleast_2d(axes)

    for row, (file_name, score) in enumerate(worst):
        pred = predictions[file_name]
        gt_anns = gt_by_image.get(file_name, [])
        gt_boxes = [_rle_to_bbox_xyxy(a["segmentation"]) for a in gt_anns]
        gt_names = [id_to_name[a["category_id"]] for a in gt_anns]
        gt_rles = [a["segmentation"] for a in gt_anns]

        with Image.open(pred.image_path) as im:
            pred_img = draw_predictions(
                im, pred.boxes, pred.class_names, pred.mask_rles, PRED_COLOR, pred.confidences
            )
            gt_img = draw_predictions(im, gt_boxes, gt_names, gt_rles, GT_COLOR)

        iou_label = "0.000 (no match)" if score["mean_iou"] == 0 else f"{score['mean_iou']:.3f}"
        axes[row, 0].imshow(pred_img)
        axes[row, 0].set_title(
            f"{file_name} — prediction ({score['n_dt']} pred)", fontsize=9
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_img)
        axes[row, 1].set_title(
            f"mean IoU = {iou_label} — ground truth ({score['n_gt']} GT)", fontsize=9
        )
        axes[row, 1].axis("off")

    fig.suptitle("Worst test images by mean mask IoU (prediction vs. ground truth)",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FAILURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote failure-case figure (%d images) -> %s", n, out_path)


def _rle_to_bbox_xyxy(rle: dict[str, Any]) -> tuple[float, float, float, float]:
    from pycocotools import mask as mask_utils

    x, y, w, h = mask_utils.toBbox(rle).tolist()
    return (x, y, x + w, y + h)


def plot_pr_curves(
    metrics_by_model: dict[str, dict[str, Any]],
    class_names: list[str],
    out_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """Per-class PR curves for one or more models. `metrics_by_model` maps
    a display name -> the raw per-class PR-curve dict from
    `pr_curve_from_coco_eval` (below); each entry has `recall_thrs` and,
    per class, a `precision` array of the same length (pycocotools' own
    101-point interpolated curve at IoU=0.5, not hand-rolled)."""
    default_colors = ["#2a78d6", "#eb6834", "#34a853", "#a142f4"]
    colors = colors or {
        name: default_colors[i % len(default_colors)]
        for i, name in enumerate(metrics_by_model)
    }

    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, class_name in enumerate(class_names):
        ax = axes_flat[i]
        for model_name, pr_data in metrics_by_model.items():
            recall = pr_data["recall_thrs"]
            precision = pr_data["per_class"][class_name]["precision_curve"]
            ap = pr_data["per_class"][class_name]["ap50"]
            ax.plot(recall, precision, label=f"{model_name} (AP={ap:.2f})",
                    color=colors[model_name], linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(class_name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)

    for ax in axes_flat[n_classes:]:
        ax.axis("off")

    fig.suptitle("Precision-recall curves per class (IoU=0.5)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=QUALITATIVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote PR curves -> %s", out_path)


def pr_curve_from_coco_eval(
    coco_gt: Any, coco_dt: Any, class_ids: list[int], class_names: list[str]  # noqa: ANN401
) -> dict[str, Any]:
    """Extract the full per-class precision-recall curve (not just AP50)
    from a pycocotools COCOeval run — the same `ev.eval["precision"]` array
    `autoassess.eval.metrics.coco_eval_summary` reduces to a scalar, kept
    here at full recall-threshold resolution for plotting."""
    from pycocotools.cocoeval import COCOeval

    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.evaluate()
    ev.accumulate()

    iou_idx, area_idx, maxdet_idx = 0, 0, -1  # IoU=0.5, area='all', maxDets=100
    recall_thrs = ev.params.recThrs.tolist()

    per_class = {}
    for k, name in zip(class_ids, class_names, strict=True):
        precision = ev.eval["precision"][iou_idx, :, class_ids.index(k), area_idx, maxdet_idx]
        precision = np.where(precision < 0, 0.0, precision)  # -1 = undefined at that recall level
        per_class[name] = {
            "precision_curve": precision.tolist(),
            "ap50": float(precision[precision >= 0].mean()) if precision.size else 0.0,
        }

    return {"recall_thrs": recall_thrs, "per_class": per_class}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate qualitative grid, failure cases, and PR curves for a "
                    "trained YOLOv8-seg model on its own test split."
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--dataset-config", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="e.g. data/processed/cardd")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-qualitative", type=int, default=12)
    p.add_argument("--n-failures", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    p.add_argument("--model-name", type=str, default="model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.dataset_config)
    images_dir = args.processed_dir / "images" / args.split
    labels_dir = args.processed_dir / "labels" / args.split

    log.info("Running inference on %s...", images_dir)
    predictions = run_predictions(args.weights, images_dir, args.conf, args.device)
    coco_gt = build_coco_ground_truth(images_dir, labels_dir, class_names)

    iou_scores = per_image_mask_iou(coco_gt, predictions)
    all_names = sorted(predictions.keys())

    qualitative_names = all_names[: args.n_qualitative]
    build_qualitative_grid(
        predictions, qualitative_names,
        args.out_dir / f"{args.model_name}_qualitative_grid.png",
        title=f"{args.model_name} — qualitative results ({args.split} split)",
    )

    worst = find_worst_images(iou_scores, args.n_failures)
    build_failure_case_figure(
        predictions, coco_gt, worst, args.out_dir / f"{args.model_name}_failure_cases.png"
    )

    from pycocotools.coco import COCO
    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    detections = _predictions_to_coco_detections(predictions, coco_gt)
    coco_dt = coco_gt_obj.loadRes(detections) if detections else None
    if coco_dt is not None:
        class_ids = [c["id"] for c in coco_gt["categories"]]
        pr_data = pr_curve_from_coco_eval(coco_gt_obj, coco_dt, class_ids, class_names)
        pr_path = args.out_dir / f"{args.model_name}_pr_curves.png"
        plot_pr_curves({args.model_name: pr_data}, class_names, pr_path)

    unscoreable = find_unscoreable_images(iou_scores)
    if unscoreable:
        log.info(
            "%d/%d test images have zero ground-truth instances after conversion "
            "(data-labeling gap, excluded from failure ranking below).",
            len(unscoreable), len(all_names),
        )

    log.info("Worst %d images by mean mask IoU (excludes zero-GT images):", len(worst))
    for name, score in worst:
        log.info(
            "  %s: mean_iou=%s n_gt=%d n_dt=%d",
            name, score["mean_iou"], score["n_gt"], score["n_dt"],
        )

    failures_json_path = args.out_dir / f"{args.model_name}_failure_cases.json"
    _write_failure_cases_json(worst, unscoreable, predictions, failures_json_path)
    log.info("Wrote failure-case data -> %s", failures_json_path)


def _write_failure_cases_json(
    worst: list[tuple[str, dict[str, Any]]],
    unscoreable: list[tuple[str, dict[str, Any]]],
    predictions: dict[str, ImagePrediction],
    out_path: Path,
) -> None:
    """Machine-readable sidecar for the failure-case figure: per-image
    score plus the predicted classes, so a report-writing step can quote
    real per-image data instead of eyeballing the figure. `worst` (real
    model failures, ranked) and `unscoreable` (zero-GT data gaps, not
    ranked — see `find_unscoreable_images`) are kept as separate lists so
    a reader can't mistake one for the other."""
    import json

    def _records(pairs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        out = []
        for name, score in pairs:
            pred = predictions[name]
            out.append({
                "file_name": name,
                "mean_iou": score["mean_iou"],
                "n_gt": score["n_gt"],
                "n_dt": score["n_dt"],
                "n_matched": score["n_matched"],
                "predicted_classes": pred.class_names,
            })
        return out

    payload = {"worst_scoreable": _records(worst), "unscoreable_zero_gt": _records(unscoreable)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _predictions_to_coco_detections(
    predictions: dict[str, ImagePrediction], coco_gt_dict: dict[str, Any]
) -> list[dict[str, Any]]:
    """Convert `ImagePrediction`s into COCO-format detection dicts for
    `COCOeval`, matched to the ground-truth `image_id`s by file name."""
    from pycocotools import mask as mask_utils

    file_to_id = {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}
    cat_id_by_name = {c["name"]: c["id"] for c in coco_gt_dict["categories"]}

    detections = []
    for file_name, pred in predictions.items():
        image_id = file_to_id.get(file_name)
        if image_id is None:
            continue
        for box, name, conf, rle in zip(
            pred.boxes, pred.class_names, pred.confidences, pred.mask_rles, strict=True
        ):
            if name not in cat_id_by_name:
                continue
            bbox = mask_utils.toBbox(rle).tolist()
            detections.append({
                "image_id": image_id,
                "category_id": cat_id_by_name[name],
                "segmentation": rle,
                "bbox": bbox,
                "score": conf,
            })
    return detections


if __name__ == "__main__":
    main()
