"""build_metrics_json -> compare rendering, for v2 and legacy v1 files (tmp_path only)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from autoassess.eval import coco_eval
from autoassess.eval.compare import (
    LEGACY_NOTICE,
    build_markdown_table,
    load_metrics,
)
from autoassess.eval.metrics import build_metrics_json, write_metrics_json
from autoassess.eval.scoring import SCORING_CONF_THRESHOLD

from .conftest import SyntheticCase

V1_PRECISION = 0.7777  # distinctive: in v1 this is AP50 mislabelled, never P


def make_v2(case: SyntheticCase, run_name: str = "run_v2", split: str = "test") -> dict[str, Any]:
    ev = coco_eval.run_coco_eval(
        copy.deepcopy(case.gt_dict), copy.deepcopy(case.detections), case.class_names
    )
    return build_metrics_json(
        run_name=run_name,
        model_type="yolov8-seg",
        model="yolov8n-seg.pt",
        class_names=case.class_names,
        epochs_requested=5,
        epochs_run_this_invocation=5,
        batch=8,
        imgsz=640,
        device="cpu",
        patience=3,
        best_epoch=4,
        early_stopped=False,
        wall_time_seconds_total=10.0,
        epoch_wall_times_seconds=[2.0, 2.0],
        eval_result=ev,
        eval_split=split,
        scoring_conf=SCORING_CONF_THRESHOLD,
        inference={"latency_ms_mean": 5.0, "latency_ms_p50": 5.0, "latency_ms_p95": 6.0,
                   "n_images": 2},
        model_info={"params_total": 1000, "params_trainable": 900, "vram_peak_mb": None},
    )


def make_v1(run_name: str = "run_v1") -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model_type": "maskrcnn",
        "model": "maskrcnn_resnet50_fpn",
        "class_names": ["dent", "scratch"],
        "epochs_requested": 5,
        "epochs_run_this_invocation": 5,
        "batch": 8,
        "imgsz": 640,
        "device": "cpu",
        "patience": 3,
        "best_epoch": 3,
        "early_stopped": False,
        "wall_time_seconds_total": 10.0,
        "metrics": {
            "box_map50": 0.5, "box_map50_95": 0.3, "mask_map50": 0.45, "mask_map50_95": 0.25,
            "mask_iou_mean": 0.8123,
            "per_class": {
                "dent": {"precision": V1_PRECISION, "recall": 0.9911,
                         "box_ap50": 0.6111, "mask_ap50": V1_PRECISION},
                "scratch": {"precision": 0.3333, "recall": 0.8822,
                            "box_ap50": 0.2222, "mask_ap50": 0.3333},
            },
        },
        "inference": {"latency_ms_mean": 7.0, "latency_ms_p50": 7.0, "latency_ms_p95": 8.0,
                      "n_images": 2},
        "model_info": {"params_total": 2000, "params_trainable": 1900, "vram_peak_mb": 10.0},
    }


def test_v2_payload_schema(synthetic_case: SyntheticCase) -> None:
    m = make_v2(synthetic_case)
    assert m["schema_version"] == 2
    assert m["eval_split"] == "test"
    mt = m["metrics"]
    assert "mask_iou_mean" not in mt
    op = mt["operating_point"]
    assert op["conf"] == 0.25
    assert op["iou"] == 0.5
    assert op["scoring_conf"] == SCORING_CONF_THRESHOLD
    exp = synthetic_case.expected["conf_0.25"]
    assert op["mask"]["precision"] == pytest.approx(
        (exp["dent"]["precision"] + exp["scratch"]["precision"]) / 2
    )
    dent = mt["per_class"]["dent"]
    assert set(dent) == {
        "box_ap50", "mask_ap50", "max_recall", "precision", "recall", "f1",
        "tp", "fp", "fn", "f1_opt_conf", "f1_opt",
    }
    assert (dent["tp"], dent["fp"], dent["fn"]) == (exp["dent"]["tp"], exp["dent"]["fp"],
                                                    exp["dent"]["fn"])


def test_v2_round_trip_renders_new_columns(
    synthetic_case: SyntheticCase, tmp_path: Path
) -> None:
    m2 = make_v2(synthetic_case)
    path = write_metrics_json(tmp_path / "run_v2", m2)
    loaded = load_metrics(path)
    md = build_markdown_table(loaded, loaded)

    for header in ("Precision@0.25", "Recall@0.25", "F1@0.25", "Best-F1 conf", "Box AP50",
                   "Mask AP50", "Max recall"):
        assert header in md
    for label in ("Mask IoU (true positives)", "Mask IoU (per GT, misses = 0)",
                  "Macro P / R / F1 @ conf 0.25 (mask)", "Evaluated on"):
        assert label in md
    assert "| Evaluated on | test | test |" in md
    assert LEGACY_NOTICE not in md

    exp = synthetic_case.expected["conf_0.25"]["dent"]
    pc = m2["metrics"]["per_class"]["dent"]
    row = next(ln for ln in md.splitlines() if ln.startswith("| dent |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == f"{exp['precision']:.4f}"
    assert cells[2] == f"{exp['recall']:.4f}"
    assert cells[3] == f"{exp['f1']:.4f}"
    assert cells[4] == f"{pc['f1_opt_conf']:.3f}"
    assert cells[5] == f"{pc['box_ap50']:.4f}"
    assert cells[6] == f"{pc['mask_ap50']:.4f}"
    assert cells[7] == f"{pc['max_recall']:.4f}"
    assert f"{m2['metrics']['mask_iou_per_gt']:.4f}" in md


def test_v1_still_loads_and_never_shows_precision(
    synthetic_case: SyntheticCase, tmp_path: Path
) -> None:
    v1_path = tmp_path / "run_v1" / "metrics.json"
    v1_path.parent.mkdir()
    v1_path.write_text(json.dumps(make_v1()), encoding="utf-8")
    v1 = load_metrics(v1_path)
    md = build_markdown_table(make_v2(synthetic_case), v1)

    assert "n/a" in md
    assert LEGACY_NOTICE in md
    assert "val (legacy)" in md
    assert "Mask IoU (legacy, matched pairs only)" in md
    assert "0.8123" in md

    row = next(ln for ln in md.splitlines() if ln.startswith("| dent |") and "0.6111" in ln)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:5] == ["n/a"] * 4  # P / R / F1 / best conf
    assert cells[5] == "0.6111"       # box AP50 kept
    assert cells[6] == f"{V1_PRECISION:.4f}"  # mask AP50 kept
    assert cells[7] == "n/a"          # max recall
    assert "0.9911" not in md  # v1 "recall" is never shown
    # the v1 precision value only ever appears as Mask AP50
    for ln in md.splitlines():
        if f"{V1_PRECISION:.4f}" in ln and ln.startswith("| dent |"):
            assert ln.count(f"{V1_PRECISION:.4f}") == 1
