"""Tests for operating-point metrics against the hand-computed synthetic case."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pycocotools.cocoeval import COCOeval

from autoassess.eval import coco_eval
from autoassess.eval.metrics import (
    coco_eval_summary,
    f1_optimal_conf,
    greedy_match,
    mask_iou_per_gt,
    mask_iou_true_positives,
    operating_point_metrics,
)

from .conftest import SyntheticCase, make_coco, make_dt

CLASS_IDS = [1, 2]
NAMES = ["dent", "scratch"]


@pytest.fixture
def gt_dt(synthetic_case: SyntheticCase) -> tuple[Any, Any]:
    coco_gt = make_coco(copy.deepcopy(synthetic_case.gt_dict))
    coco_dt = make_dt(coco_gt, copy.deepcopy(synthetic_case.detections))
    return coco_gt, coco_dt


@pytest.mark.parametrize(("key", "conf"), [("conf_0.25", 0.25), ("no_conf", 0.0)])
@pytest.mark.parametrize("iou_type", ["segm", "bbox"])
def test_operating_point_counts(
    gt_dt: tuple[Any, Any], synthetic_case: SyntheticCase, key: str, conf: float, iou_type: str
) -> None:
    coco_gt, coco_dt = gt_dt
    res = operating_point_metrics(coco_gt, coco_dt, iou_type, CLASS_IDS, NAMES, conf_thr=conf)
    for name in NAMES:
        exp = synthetic_case.expected[key][name]
        got = res["per_class"][name]
        for field in ("tp", "fp", "fn"):
            assert got[field] == exp[field], (name, field)
        for field in ("precision", "recall", "f1"):
            assert got[field] == pytest.approx(exp[field]), (name, field)


def test_macro_average(gt_dt: tuple[Any, Any], synthetic_case: SyntheticCase) -> None:
    coco_gt, coco_dt = gt_dt
    res = operating_point_metrics(coco_gt, coco_dt, "segm", CLASS_IDS, NAMES, conf_thr=0.25)
    exp = synthetic_case.expected["conf_0.25"]
    for field in ("precision", "recall", "f1"):
        want = (exp["dent"][field] + exp["scratch"][field]) / 2
        assert res["macro"][field] == pytest.approx(want)


def test_wrong_class_overlap_does_not_match(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    m = greedy_match(coco_gt, coco_dt, "segm", conf_thr=0.0)
    # GT4 (dent) only overlaps a scratch detection: stays a miss.
    dent = m[1]
    assert dent.gt_ids == [1, 4, 5, 7]
    assert dent.gt_ious[dent.gt_ids.index(4)] == 0.0


def test_higher_score_lower_iou_wins(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    m = greedy_match(coco_gt, coco_dt, "segm", conf_thr=0.0)
    dent = m[1]
    assert dent.gt_ious[dent.gt_ids.index(5)] == pytest.approx(0.6)
    # the 0.6-score / IoU 0.9 detection lost the GT
    losers = [d for d in dent.dets if d[0] == pytest.approx(0.6)]
    assert losers == [(pytest.approx(0.6), False, None)]


def test_greedy_match_structure(gt_dt: tuple[Any, Any], synthetic_case: SyntheticCase) -> None:
    coco_gt, coco_dt = gt_dt
    m = greedy_match(coco_gt, coco_dt, "segm", conf_thr=0.25)
    for name, cid in zip(NAMES, CLASS_IDS, strict=True):
        exp = synthetic_case.expected["conf_0.25"][name]
        assert m[cid].gt_ids == synthetic_case.expected["gt_ids"][name]
        assert m[cid].gt_ious == pytest.approx(exp["per_gt_iou"])
        matched = [iou for _, ok, iou in m[cid].dets if ok]
        assert sorted(matched) == pytest.approx(sorted(exp["matched_ious"]))


def test_max_dets_caps_per_image_and_class(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    m = greedy_match(coco_gt, coco_dt, "segm", conf_thr=0.0, max_dets=1)
    # img3 has two dent dets; only the top-scoring one survives
    assert len(m[1].dets) == 3  # img1 [0], img2 [2], img3 [4]


def test_mask_iou_true_positives(gt_dt: tuple[Any, Any], synthetic_case: SyntheticCase) -> None:
    coco_gt, coco_dt = gt_dt
    exp = synthetic_case.expected["conf_0.25"]
    ious = exp["dent"]["matched_ious"] + exp["scratch"]["matched_ious"]
    got = mask_iou_true_positives(coco_gt, coco_dt, conf_thr=0.25)
    assert got == pytest.approx(sum(ious) / len(ious))
    assert got >= 0.5


def test_mask_iou_per_gt_counts_misses(
    gt_dt: tuple[Any, Any], synthetic_case: SyntheticCase
) -> None:
    coco_gt, coco_dt = gt_dt
    exp = synthetic_case.expected["conf_0.25"]
    ious = exp["dent"]["per_gt_iou"] + exp["scratch"]["per_gt_iou"]
    got = mask_iou_per_gt(coco_gt, coco_dt, conf_thr=0.25)
    assert got == pytest.approx(sum(ious) / len(ious))
    assert got < mask_iou_true_positives(coco_gt, coco_dt, conf_thr=0.25)


def test_f1_optimal_conf(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    res = f1_optimal_conf(coco_gt, coco_dt, "segm", CLASS_IDS, NAMES)
    # dent, 4 GT. Thresholds {0.9, 0.6}.
    #  conf 0.9: dets [0] TP, [2] FP, [4] TP -> TP2 FP1: P 2/3, R 2/4, F1 = 4/7
    #  conf 0.6: adds [5] FP -> TP2 FP2: P 1/2, R 1/2, F1 = 1/2
    # best is 0.9 with F1 4/7.
    assert res["dent"]["conf"] == pytest.approx(0.9)
    assert res["dent"]["f1"] == pytest.approx(4 / 7)
    assert res["dent"]["precision"] == pytest.approx(2 / 3)
    assert res["dent"]["recall"] == pytest.approx(0.5)
    # scratch, 3 GT. Thresholds {0.9, 0.1}.
    #  conf 0.9: [1] TP, [3] FP -> P 1/2, R 1/3, F1 = 2/5
    #  conf 0.1: adds [6] TP -> TP2 FP1: P 2/3, R 2/3, F1 = 2/3
    # best is 0.1 with F1 2/3.
    assert res["scratch"]["conf"] == pytest.approx(0.1)
    assert res["scratch"]["f1"] == pytest.approx(2 / 3)
    assert res["scratch"]["precision"] == pytest.approx(2 / 3)
    assert res["scratch"]["recall"] == pytest.approx(2 / 3)


def test_f1_optimal_agrees_with_rematching(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    res = f1_optimal_conf(coco_gt, coco_dt, "segm", CLASS_IDS, NAMES)
    for name in NAMES:
        # score ties are included whole (score >= conf), same as re-matching
        op = operating_point_metrics(
            coco_gt, coco_dt, "segm", CLASS_IDS, NAMES, conf_thr=res[name]["conf"]
        )["per_class"][name]
        assert op["f1"] == pytest.approx(res[name]["f1"])
        assert op["precision"] == pytest.approx(res[name]["precision"])
        assert op["recall"] == pytest.approx(res[name]["recall"])


def test_coco_eval_summary_schema(gt_dt: tuple[Any, Any]) -> None:
    coco_gt, coco_dt = gt_dt
    out = coco_eval_summary(coco_gt, coco_dt, "segm", CLASS_IDS, NAMES)
    for name in NAMES:
        assert set(out["per_class"][name]) == {"ap50", "max_recall"}

    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.evaluate()
    ev.accumulate()
    for k, name in enumerate(NAMES):
        prec = ev.eval["precision"][0, :, k, 0, -1]
        prec = prec[prec > -1]
        assert out["per_class"][name]["ap50"] == pytest.approx(float(prec.mean()))
        assert out["per_class"][name]["max_recall"] == pytest.approx(
            float(ev.eval["recall"][0, k, 0, -1])
        )


def test_run_coco_eval_end_to_end(synthetic_case: SyntheticCase) -> None:
    out = coco_eval.run_coco_eval(
        copy.deepcopy(synthetic_case.gt_dict),
        copy.deepcopy(synthetic_case.detections),
        synthetic_case.class_names,
    )
    assert set(out) == {"box", "mask", "mask_iou_mean"}
    assert 0.0 <= out["mask_iou_mean"] <= 1.0
    assert "precision" not in out["mask"]["per_class"]["dent"]
