"""Check the synthetic fixture agrees with its own documented answers."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from tests.conftest import SyntheticCase, make_coco, make_dt


def test_pair_ious_match_documented(synthetic_case: SyntheticCase) -> None:
    gts = {a["id"]: a for a in synthetic_case.gt_dict["annotations"]}
    dets = synthetic_case.detections
    documented = {(d, g): iou for d, g, iou in synthetic_case.expected["pair_ious"]}

    for di, det in enumerate(dets):
        for gid, gt in gts.items():
            if det["image_id"] != gt["image_id"]:
                continue
            iou = float(
                mask_utils.iou([det["segmentation"]], [gt["segmentation"]], [0])[0][0]
            )
            want = documented.get((di, gid), 0.0)
            assert abs(iou - want) < 1e-6, (di, gid, iou, want)


def test_documented_pairs_share_an_image(synthetic_case: SyntheticCase) -> None:
    gts = {a["id"]: a for a in synthetic_case.gt_dict["annotations"]}
    for di, gid, _ in synthetic_case.expected["pair_ious"]:
        assert synthetic_case.detections[di]["image_id"] == gts[gid]["image_id"]


def test_loadres_succeeds(synthetic_case: SyntheticCase) -> None:
    coco_gt = make_coco(synthetic_case.gt_dict)
    coco_dt = make_dt(coco_gt, synthetic_case.detections)
    assert len(coco_dt.getAnnIds()) == len(synthetic_case.detections)
    assert len(coco_gt.getAnnIds()) == len(synthetic_case.gt_dict["annotations"])


def test_expected_counts_are_consistent(synthetic_case: SyntheticCase) -> None:
    gt_ids = synthetic_case.expected["gt_ids"]
    for key in ("conf_0.25", "no_conf"):
        for name in synthetic_case.class_names:
            e = synthetic_case.expected[key][name]
            assert e["tp"] + e["fn"] == len(gt_ids[name])
            assert len(e["per_gt_iou"]) == len(gt_ids[name])
            assert len(e["matched_ious"]) == e["tp"]
            assert abs(e["precision"] - e["tp"] / (e["tp"] + e["fp"])) < 1e-9
            assert abs(e["recall"] - e["tp"] / (e["tp"] + e["fn"])) < 1e-9
            p, r = e["precision"], e["recall"]
            assert abs(e["f1"] - 2 * p * r / (p + r)) < 1e-9
            assert np.isclose(sum(e["per_gt_iou"]), sum(e["matched_ious"]))
