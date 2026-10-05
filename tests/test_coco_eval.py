"""run_coco_eval returns the v2 evaluation shape (synthetic fixture only)."""

from __future__ import annotations

import copy

import pytest

from autoassess.eval import coco_eval
from autoassess.eval.scoring import OPERATING_CONF

from .conftest import SyntheticCase

TOP_KEYS = {
    "box",
    "mask",
    "operating_point",
    "f1_optimal",
    "mask_iou_true_positives",
    "mask_iou_per_gt",
}


def _run(case: SyntheticCase, detections: list[dict] | None = None) -> dict:  # type: ignore[type-arg]
    dets = case.detections if detections is None else detections
    return coco_eval.run_coco_eval(
        copy.deepcopy(case.gt_dict), copy.deepcopy(dets), case.class_names
    )


def test_v2_shape(synthetic_case: SyntheticCase) -> None:
    out = _run(synthetic_case)
    assert set(out) == TOP_KEYS
    assert "mask_iou_mean" not in out
    for kind in ("box", "mask"):
        assert set(out[kind]) == {"map50", "map50_95", "per_class"}
        for name in synthetic_case.class_names:
            assert set(out[kind]["per_class"][name]) == {"ap50", "max_recall"}
    op = out["operating_point"]
    assert op["conf"] == OPERATING_CONF == 0.25
    assert op["iou"] == 0.5
    for kind in ("box", "mask"):
        assert set(op[kind]) == {"per_class", "macro"}
        assert set(out["f1_optimal"][kind]) == set(synthetic_case.class_names)
    assert 0.0 <= out["mask_iou_true_positives"] <= 1.0
    assert 0.0 <= out["mask_iou_per_gt"] <= 1.0


def test_operating_point_matches_expected(synthetic_case: SyntheticCase) -> None:
    out = _run(synthetic_case)
    for name in synthetic_case.class_names:
        exp = synthetic_case.expected["conf_0.25"][name]
        got = out["operating_point"]["mask"]["per_class"][name]
        for field in ("tp", "fp", "fn"):
            assert got[field] == exp[field], (name, field)
        for field in ("precision", "recall", "f1"):
            assert got[field] == pytest.approx(exp[field]), (name, field)


def test_empty_detections_same_shape_fn_is_gt_count(synthetic_case: SyntheticCase) -> None:
    out = _run(synthetic_case, detections=[])
    assert set(out) == TOP_KEYS
    for kind in ("box", "mask"):
        assert out[kind]["map50"] == 0.0
        assert out[kind]["map50_95"] == 0.0
        per_class = out["operating_point"][kind]["per_class"]
        for name in synthetic_case.class_names:
            n_gt = len(synthetic_case.expected["gt_ids"][name])
            assert per_class[name]["fn"] == n_gt
            assert per_class[name]["tp"] == 0
            assert per_class[name]["fp"] == 0
            assert per_class[name]["recall"] == 0.0
            assert per_class[name]["precision"] == 0.0
            assert set(out[kind]["per_class"][name]) == {"ap50", "max_recall"}
            assert out["f1_optimal"][kind][name]["f1"] == 0.0
        assert out["operating_point"][kind]["macro"] == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
    assert out["operating_point"]["conf"] == OPERATING_CONF
    assert out["mask_iou_true_positives"] == 0.0
    assert out["mask_iou_per_gt"] == 0.0
