"""Build COCO-format ground truth from converted YOLO-seg labels, and run
pycocotools evaluation against a model's COCO-format detections.

Both ``train_yolo.py`` and ``train_maskrcnn.py`` read the same converted
``data/processed/cardd/{images,labels}/<split>`` directory (produced by
``autoassess.data.convert``); this module is the single place that turns
those YOLO-format polygon labels back into COCO ground truth, so both
models are scored against byte-identical annotations regardless of which
training framework consumed them.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autoassess.eval.scoring import OPERATING_CONF

IOU_THR = 0.5


def load_class_names(dataset_config_path: Path) -> list[str]:
    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.load(dataset_config_path)
    names: dict[int, str] = OmegaConf.to_container(ds_cfg.names, resolve=True)  # type: ignore[assignment]
    return [names[i] for i in sorted(names, key=int)]


def _polygon_to_rle(
    flat_norm_poly: list[float], width: int, height: int
) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    pts = np.array(flat_norm_poly, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    rles = mask_utils.frPyObjects([pts.flatten().tolist()], height, width)
    rle = mask_utils.merge(rles)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle  # type: ignore[no-any-return]


def build_coco_ground_truth(
    images_dir: Path, labels_dir: Path, class_names: list[str]
) -> dict[str, Any]:
    """Reconstruct a COCO-format ground-truth dict from a converted YOLO-seg
    split. Category ids are 1-indexed (COCO convention), in the same order
    as `class_names` (already validated against the source COCO categories
    by `autoassess.data.convert`, so this ordering is authoritative).
    """
    categories = [
        {"id": i + 1, "name": name, "supercategory": "damage"}
        for i, name in enumerate(class_names)
    ]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    label_files = sorted(labels_dir.glob("*.txt"))
    for image_id, label_path in enumerate(label_files, start=1):
        stem = label_path.stem
        image_path = _find_image(images_dir, stem)
        if image_path is None:
            continue
        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_index = int(parts[0])
            poly = [float(v) for v in parts[1:]]
            rle = _polygon_to_rle(poly, width, height)

            from pycocotools import mask as mask_utils
            bbox = mask_utils.toBbox(rle).tolist()
            area = float(mask_utils.area(rle))

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_index + 1,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def image_id_lookup(coco_gt_dict: dict[str, Any]) -> dict[str, int]:
    """Map file_name -> COCO image_id, so a model's per-image detections
    (produced in file-iteration order) can be attached to the right id."""
    return {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}


def mask_to_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle  # type: ignore[no-any-return]


def _empty_summary(class_names: list[str]) -> dict[str, Any]:
    return {
        "map50": 0.0,
        "map50_95": 0.0,
        "per_class": {n: {"ap50": 0.0, "max_recall": 0.0} for n in class_names},
    }


def _empty_operating_point(
    coco_gt_dict: dict[str, Any], class_ids: list[int], class_names: list[str]
) -> dict[str, Any]:
    """Operating-point result for a model with no detections: every non-crowd
    GT is a false negative, so recall is a real 0.0 rather than undefined."""
    n_gt = {cid: 0 for cid in class_ids}
    for ann in coco_gt_dict["annotations"]:
        if not ann.get("iscrowd", 0) and ann["category_id"] in n_gt:
            n_gt[ann["category_id"]] += 1
    per_class = {
        name: {
            "tp": 0, "fp": 0, "fn": n_gt[cid],
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
        for cid, name in zip(class_ids, class_names, strict=True)
    }
    return {"per_class": per_class, "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}


def run_coco_eval(
    coco_gt_dict: dict[str, Any],
    detections: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Score COCO-format detections against an in-memory ground-truth dict.

    Returns::

        {"box" | "mask": {map50, map50_95, per_class: {name: {ap50, max_recall}}},
         "operating_point": {"conf": OPERATING_CONF, "iou": 0.5,
                             "box" | "mask": {per_class: {name: {tp, fp, fn,
                                              precision, recall, f1}},
                                              macro: {precision, recall, f1}}},
         "f1_optimal": {"box" | "mask": {name: {conf, f1, precision, recall}}},
         "mask_iou_true_positives": float,   # mean IoU of matched pairs only
         "mask_iou_per_gt": float}           # mean over all GTs, misses = 0

    AP is computed on every detection passed in (the scoring threshold);
    operating-point numbers use ``OPERATING_CONF`` at IoU 0.5. With no
    detections the same shape is returned with zeros and FN = number of GTs.
    """
    from pycocotools.coco import COCO

    from autoassess.eval.metrics import (
        coco_eval_summary,
        f1_optimal_conf,
        mask_iou_per_gt,
        mask_iou_true_positives,
        operating_point_metrics,
    )

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    class_ids = [c["id"] for c in coco_gt_dict["categories"]]

    if not detections:
        empty_op = _empty_operating_point(coco_gt_dict, class_ids, class_names)
        zero_f1 = {
            n: {"conf": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0} for n in class_names
        }
        return {
            "box": _empty_summary(class_names),
            "mask": _empty_summary(class_names),
            "operating_point": {
                "conf": OPERATING_CONF,
                "iou": IOU_THR,
                "box": empty_op,
                "mask": copy.deepcopy(empty_op),
            },
            "f1_optimal": {"box": zero_f1, "mask": copy.deepcopy(zero_f1)},
            "mask_iou_true_positives": 0.0,
            "mask_iou_per_gt": 0.0,
        }

    coco_dt = coco_gt.loadRes(detections)

    def op(iou_type: str) -> dict[str, Any]:
        return operating_point_metrics(
            coco_gt, coco_dt, iou_type, class_ids, class_names, OPERATING_CONF, IOU_THR
        )

    def f1opt(iou_type: str) -> dict[str, dict[str, float]]:
        return f1_optimal_conf(coco_gt, coco_dt, iou_type, class_ids, class_names, IOU_THR)

    return {
        "box": coco_eval_summary(coco_gt, coco_dt, "bbox", class_ids, class_names),
        "mask": coco_eval_summary(coco_gt, coco_dt, "segm", class_ids, class_names),
        "operating_point": {
            "conf": OPERATING_CONF,
            "iou": IOU_THR,
            "box": op("bbox"),
            "mask": op("segm"),
        },
        "f1_optimal": {"box": f1opt("bbox"), "mask": f1opt("segm")},
        "mask_iou_true_positives": mask_iou_true_positives(
            coco_gt, coco_dt, OPERATING_CONF, IOU_THR
        ),
        "mask_iou_per_gt": mask_iou_per_gt(coco_gt, coco_dt, OPERATING_CONF, IOU_THR),
    }
