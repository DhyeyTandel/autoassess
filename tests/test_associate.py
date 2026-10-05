"""Tests for coverage-based damage -> part association (synthetic RLE masks, CPU only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from autoassess.infer.associate import (
    UNASSIGNED,
    MaskInstance,
    associate_damage_to_parts,
    build_association_records,
    mask_coverage,
)
from autoassess.models.severity import SeverityConfig

H, W = 200, 200


def _inst(
    box: tuple[int, int, int, int], name: str = "x", conf: float = 0.5
) -> MaskInstance:
    """Rectangle (x0, y0, x1, y1) as a MaskInstance with an RLE mask."""
    arr = np.zeros((H, W), dtype=np.uint8, order="F")
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = 1
    rle: dict[str, Any] = mask_utils.encode(arr)
    return MaskInstance(
        class_name=name,
        mask_rle=rle,
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        mask_area=float(mask_utils.area(rle)),
        confidence=conf,
    )


def test_small_damage_inside_large_part_is_assigned() -> None:
    damage = _inst((50, 50, 60, 60), "dent")  # 100 px
    part = _inst((0, 0, 200, 200), "door")  # 40000 px, IoU = 0.0025
    iou = float(mask_utils.iou([damage.mask_rle], [part.mask_rle], [0])[0, 0])
    assert iou < 0.10
    [(d, p, cov)] = associate_damage_to_parts([damage], [part], 0.5)
    assert d is damage and p is part
    assert cov == 1.0


def test_highest_coverage_wins() -> None:
    damage = _inst((0, 0, 20, 20))  # 400 px
    part_a = _inst((0, 0, 10, 20), "A")  # covers half
    part_b = _inst((0, 0, 20, 5), "B")  # covers a quarter
    [(_, p, cov)] = associate_damage_to_parts([damage], [part_b, part_a], 0.1)
    assert p is part_a
    assert cov == 0.5


def test_below_threshold_is_unassigned() -> None:
    damage = _inst((0, 0, 20, 20))
    part = _inst((0, 0, 20, 5))  # coverage 0.25
    [(_, p, cov)] = associate_damage_to_parts([damage], [part], 0.5)
    assert p is None
    assert cov == 0.0


def test_no_parts_is_unassigned() -> None:
    [(_, p, cov)] = associate_damage_to_parts([_inst((0, 0, 10, 10))], [], 0.5)
    assert p is None
    assert cov == 0.0


def test_equal_coverage_prefers_higher_confidence_then_input_order() -> None:
    damage = _inst((0, 0, 10, 10))
    low = _inst((0, 0, 100, 100), "low", conf=0.3)
    high = _inst((0, 0, 50, 50), "high", conf=0.9)
    [(_, p, _)] = associate_damage_to_parts([damage], [low, high], 0.5)
    assert p is high
    first = _inst((0, 0, 100, 100), "first", conf=0.5)
    second = _inst((0, 0, 50, 50), "second", conf=0.5)
    [(_, p, _)] = associate_damage_to_parts([damage], [first, second], 0.5)
    assert p is first


def test_zero_area_damage_does_not_crash() -> None:
    empty = _inst((0, 0, 0, 0), "dent")
    part = _inst((0, 0, 50, 50))
    assert mask_coverage(empty.mask_rle, part.mask_rle) == 0.0
    [(_, p, cov)] = associate_damage_to_parts([empty], [part], 0.5)
    assert p is None
    assert cov == 0.0


def test_build_records_emit_part_coverage() -> None:
    cfg = SeverityConfig.load(
        Path(__file__).resolve().parents[1] / "configs" / "severity.yaml"
    )
    d1 = _inst((10, 10, 20, 20), "scratch")
    d2 = _inst((150, 150, 160, 160), "scratch")
    part = _inst((0, 0, 100, 100), "door")
    assoc = associate_damage_to_parts([d1, d2], [part], 0.5)
    records = build_association_records([d1, d2], assoc, W, H, cfg)
    assert "part_iou" not in records[0]
    assert records[0]["part"] == "door"
    assert records[0]["part_coverage"] == 1.0
    assert records[1]["part"] == UNASSIGNED
    assert records[1]["part_coverage"] is None
