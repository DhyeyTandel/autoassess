"""Shared synthetic COCO fixtures with hand-computed answers.

All images are 100x100 and all masks are axis-aligned rectangles, so every
IoU below is an exact ratio of integer areas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

IMG_W = 100
IMG_H = 100


def rect_rle(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> dict[str, Any]:
    """COCO RLE (counts as str, like coco_eval.mask_to_rle) for a filled rectangle."""
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    m[y : y + h, x : x + w] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle  # type: ignore[no-any-return]


def make_coco(gt_dict: dict[str, Any]) -> COCO:
    """Build an in-memory COCO ground-truth object (as run_coco_eval does)."""
    coco = COCO()
    coco.dataset = gt_dict
    coco.createIndex()
    return coco


def make_dt(coco_gt: COCO, detections: list[dict[str, Any]]) -> COCO:
    """Build a detection COCO object via loadRes."""
    return coco_gt.loadRes(detections)


@dataclass(frozen=True)
class SyntheticCase:
    gt_dict: dict[str, Any]
    detections: list[dict[str, Any]]
    class_names: list[str]
    expected: dict[str, Any]


def _ann(ann_id: int, image_id: int, cat: int, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    x, y, w, h = rect
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat,
        "segmentation": rect_rle(x, y, w, h, IMG_W, IMG_H),
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }


def _det(image_id: int, cat: int, rect: tuple[int, int, int, int], score: float) -> dict[str, Any]:
    x, y, w, h = rect
    return {
        "image_id": image_id,
        "category_id": cat,
        "segmentation": rect_rle(x, y, w, h, IMG_W, IMG_H),
        "bbox": [x, y, w, h],
        "score": score,
    }


@pytest.fixture
def synthetic_case() -> SyntheticCase:
    """Scenarios (dent = cat 1, scratch = cat 2), GT ids in brackets:

    img1: [1] dent TP (IoU 0.8); [2] scratch missed; [3] scratch TP (IoU 0.9)
    img2: [4] dent GT with a wrong-class (scratch) detection on it; plus a dent FP
    img3: [5] dent GT, two competing dets: score 0.9/IoU 0.6 beats score 0.6/IoU 0.9
    img4: [6] scratch GT with a low-confidence (0.1) TP, IoU 0.7
    img5: [7] dent GT, zero detections
    img6: no GT, no detections
    """
    gt_rects: list[tuple[int, int, int, tuple[int, int, int, int]]] = [
        (1, 1, 1, (10, 10, 10, 10)),  # area 100
        (2, 1, 2, (50, 50, 10, 10)),  # area 100
        (3, 1, 2, (70, 10, 10, 10)),  # area 100
        (4, 2, 1, (10, 10, 10, 10)),  # area 100
        (5, 3, 1, (10, 10, 20, 10)),  # area 200
        (6, 4, 2, (20, 20, 10, 10)),  # area 100
        (7, 5, 1, (30, 30, 10, 10)),  # area 100
    ]
    gt_dict: dict[str, Any] = {
        "images": [
            {"id": i, "width": IMG_W, "height": IMG_H, "file_name": f"img{i}.jpg"}
            for i in range(1, 7)
        ],
        "categories": [
            {"id": 1, "name": "dent"},
            {"id": 2, "name": "scratch"},
        ],
        "annotations": [_ann(a, img, cat, r) for a, img, cat, r in gt_rects],
    }

    detections = [
        _det(1, 1, (10, 10, 10, 8), 0.9),   # [0] vs GT1: 80/100 = 0.8 -> TP
        _det(1, 2, (70, 10, 10, 9), 0.9),   # [1] vs GT3: 90/100 = 0.9 -> TP
        _det(2, 1, (70, 70, 10, 10), 0.9),  # [2] no GT overlap -> FP
        _det(2, 2, (10, 10, 10, 8), 0.9),   # [3] on GT4, wrong class: 80/100 = 0.8
        _det(3, 1, (10, 10, 12, 10), 0.9),  # [4] vs GT5: 120/200 = 0.6, higher score -> wins
        _det(3, 1, (10, 10, 18, 10), 0.6),  # [5] vs GT5: 180/200 = 0.9, lower score -> FP
        _det(4, 2, (20, 20, 10, 7), 0.1),   # [6] vs GT6: 70/100 = 0.7, score below 0.25
    ]

    expected: dict[str, Any] = {
        # (det_index, gt_id, mask IoU) for every spatially overlapping pair,
        # ignoring class. All other pairs have IoU 0.
        "pair_ious": [
            (0, 1, 0.8),  # 80 / 100
            (1, 3, 0.9),  # 90 / 100
            (3, 4, 0.8),  # 80 / 100 (wrong class, must not match)
            (4, 5, 0.6),  # 120 / 200
            (5, 5, 0.9),  # 180 / 200
            (6, 6, 0.7),  # 70 / 100
        ],
        # GT ids per class, in id order (order of the per_gt_iou lists)
        "gt_ids": {"dent": [1, 4, 5, 7], "scratch": [2, 3, 6]},
        "conf_0.25": {
            "dent": {
                # dets kept: [0] TP, [2] FP, [4] TP, [5] FP (lost GT5 to [4])
                "tp": 2,
                "fp": 2,  # [2] no overlap + [5] outranked by [4]
                "fn": 2,  # GT4 (only wrong-class det) + GT7 (no det)
                "precision": 0.5,  # 2 / (2 + 2)
                "recall": 0.5,  # 2 / (2 + 2)
                "f1": 0.5,  # 2 * 0.5 * 0.5 / (0.5 + 0.5)
                "matched_ious": [0.8, 0.6],  # det [0] then det [4]
                "per_gt_iou": [0.8, 0.0, 0.6, 0.0],  # GT1, GT4 miss, GT5, GT7 miss
            },
            "scratch": {
                # dets kept: [1] TP, [3] FP (wrong class); [6] dropped (0.1 < 0.25)
                "tp": 1,
                "fp": 1,  # [3] wrong-class detection
                "fn": 2,  # GT2 (no det) + GT6 (only low-conf det)
                "precision": 0.5,  # 1 / (1 + 1)
                "recall": 1 / 3,  # 1 / (1 + 2)
                "f1": 0.4,  # 2 * 0.5 * (1/3) / (0.5 + 1/3) = (1/3) / (5/6)
                "matched_ious": [0.9],  # det [1]
                "per_gt_iou": [0.0, 0.9, 0.0],  # GT2 miss, GT3, GT6 miss (conf filtered)
            },
        },
        "no_conf": {
            "dent": {
                # same as at 0.25: no dent det is below 0.25
                "tp": 2,
                "fp": 2,
                "fn": 2,
                "precision": 0.5,  # 2 / 4
                "recall": 0.5,  # 2 / 4
                "f1": 0.5,
                "matched_ious": [0.8, 0.6],
                "per_gt_iou": [0.8, 0.0, 0.6, 0.0],
            },
            "scratch": {
                # det [6] now counts: TP for GT6 at IoU 0.7
                "tp": 2,  # [1] and [6]
                "fp": 1,  # [3] wrong class
                "fn": 1,  # GT2
                "precision": 2 / 3,  # 2 / (2 + 1)
                "recall": 2 / 3,  # 2 / (2 + 1)
                "f1": 2 / 3,  # 2 * (2/3) * (2/3) / (4/3)
                "matched_ious": [0.9, 0.7],  # det [1] (score .9) then det [6] (score .1)
                "per_gt_iou": [0.0, 0.9, 0.7],  # GT2 miss, GT3, GT6
            },
        },
    }

    return SyntheticCase(
        gt_dict=gt_dict,
        detections=detections,
        class_names=["dent", "scratch"],
        expected=expected,
    )
