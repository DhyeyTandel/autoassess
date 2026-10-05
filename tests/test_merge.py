"""Tests for secondary-damage-model merging and its pipeline wiring (CPU, synthetic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from autoassess.infer import pipeline
from autoassess.infer.associate import MaskInstance
from autoassess.infer.merge import MergeConfig, merge_damage_instances
from autoassess.models.severity import DamageInstance, SeverityConfig, grade_image, severity_score
from tests.conftest import rect_rle

W = H = 100
REPO = Path(__file__).resolve().parent.parent
SEVERITY_YAML = REPO / "configs" / "severity.yaml"


def _inst(name: str, x: int, y: int, w: int, h: int, conf: float = 0.5) -> MaskInstance:
    return MaskInstance(
        class_name=name,
        mask_rle=rect_rle(x, y, w, h, W, H),
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        mask_area=float(w * h),
        confidence=conf,
    )


SEC = frozenset({"torn", "missing_part", "punctured"})


def test_unlisted_secondary_class_dropped() -> None:
    merged, tags = merge_damage_instances([], [_inst("dent", 0, 0, 10, 10)], SEC, 0.5)
    assert merged == [] and tags == []


def test_overlap_threshold() -> None:
    primary = [_inst("dent", 0, 0, 10, 10)]
    same = _inst("torn", 0, 0, 10, 10)  # IoU 1.0
    # 10x10 vs 10x10 shifted 5 in x: inter 50, union 150 -> 1/3
    partial = _inst("torn", 5, 0, 10, 10)
    merged, _ = merge_damage_instances(primary, [same], SEC, 0.5)
    assert len(merged) == 1
    merged, tags = merge_damage_instances(primary, [partial], SEC, 0.5)
    assert len(merged) == 2 and tags == ["primary", "secondary"]
    # Exactly at threshold is dropped (>= suppress_iou).
    merged, _ = merge_damage_instances(primary, [partial], SEC, 1 / 3 - 1e-9)
    assert len(merged) == 1


def test_secondary_cross_class_dedup_keeps_highest_conf() -> None:
    low = _inst("punctured", 0, 0, 10, 10, conf=0.3)
    high = _inst("missing_part", 0, 0, 10, 10, conf=0.6)
    merged, tags = merge_damage_instances([], [low, high], SEC, 0.5)
    assert [m.class_name for m in merged] == ["missing_part"]
    assert merged[0].confidence == 0.6
    assert tags == ["secondary"]


def test_secondary_same_class_dedup_by_iou() -> None:
    a = _inst("torn", 0, 0, 10, 10, conf=0.7)
    # IoU 1/3 with `a` when shifted 5 in x.
    b = _inst("torn", 5, 0, 10, 10, conf=0.4)
    c = _inst("torn", 0, 0, 10, 10, conf=0.4)  # IoU 1.0 with a
    merged, _ = merge_damage_instances([], [a, c], SEC, 0.5)
    assert len(merged) == 1
    merged, tags = merge_damage_instances([], [a, b], SEC, 0.5)
    assert len(merged) == 2 and tags == ["secondary", "secondary"]


def test_secondary_kept_in_descending_confidence_order() -> None:
    p = [_inst("dent", 0, 0, 10, 10)]
    lo = _inst("torn", 20, 20, 10, 10, conf=0.2)
    hi = _inst("punctured", 60, 60, 10, 10, conf=0.9)
    merged, _ = merge_damage_instances(p, [lo, hi], SEC, 0.5)
    assert [m.class_name for m in merged] == ["dent", "punctured", "torn"]


def test_source_tags_align() -> None:
    p = [_inst("dent", 0, 0, 10, 10), _inst("scratch", 50, 50, 10, 10)]
    s = [
        _inst("dent", 80, 0, 5, 5),
        _inst("torn", 20, 20, 10, 10),
        _inst("punctured", 70, 70, 5, 5),
    ]
    merged, tags = merge_damage_instances(p, s, SEC, 0.5)
    assert [m.class_name for m in merged] == ["dent", "scratch", "torn", "punctured"]
    assert tags == ["primary", "primary", "secondary", "secondary"]


def test_merge_config_load_real() -> None:
    cfg = MergeConfig.load(SEVERITY_YAML)
    assert cfg.secondary_classes == SEC
    assert cfg.suppress_iou == 0.5


@pytest.mark.parametrize("name", ["torn", "missing_part", "punctured"])
def test_severity_has_weights(name: str) -> None:
    cfg = SeverityConfig.load(SEVERITY_YAML)
    inst = [DamageInstance(class_name=name, mask_area=100.0, bbox=(0.0, 0.0, 10.0, 10.0))]
    assert severity_score(inst, W, H, cfg) > 0
    assert grade_image(inst, W, H, cfg)["label"] in cfg.labels


def _write_dataset_cfg(path: Path, names: list[str]) -> Path:
    path.write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return path


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    img = tmp_path / "img.png"
    Image.new("RGB", (W, H)).save(img)
    primary_w = tmp_path / "runs" / "yolov8_seg_v1" / "weights" / "best.pt"
    second_w = tmp_path / "runs" / "vehide_seg_v1" / "weights" / "best.pt"
    parts_w = tmp_path / "runs" / "parts_seg_v1" / "weights" / "best.pt"
    fixed = {
        primary_w: [_inst("dent", 0, 0, 10, 10)],
        second_w: [
            _inst("dent", 50, 50, 10, 10),  # not listed
            _inst("torn", 0, 0, 10, 10),  # overlaps primary -> suppressed
            _inst("missing_part", 30, 30, 10, 10),  # kept
        ],
        parts_w: [_inst("door", 0, 0, 50, 50)],
    }

    def fake(weights: Path, source: Path, conf: float, device: str) -> list[MaskInstance]:
        return list(fixed[weights])

    monkeypatch.setattr(pipeline, "run_yolo_seg_inference", fake)
    return {
        "source": img,
        "damage_weights": primary_w,
        "parts_weights": parts_w,
        "secondary": second_w,
        "damage_dataset_config": _write_dataset_cfg(tmp_path / "d.yaml", ["dent", "scratch"]),
        "secondary_cfg": _write_dataset_cfg(
            tmp_path / "s.yaml", ["scratch", "dent", "torn", "missing_part", "punctured"]
        ),
        "parts_dataset_config": _write_dataset_cfg(tmp_path / "p.yaml", ["door"]),
    }


def _run(s: dict[str, Any], **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    result, _d, _p = pipeline.run_pipeline_with_masks(
        source=s["source"],
        damage_weights=s["damage_weights"],
        parts_weights=s["parts_weights"],
        damage_dataset_config=s["damage_dataset_config"],
        parts_dataset_config=s["parts_dataset_config"],
        severity_config_path=SEVERITY_YAML,
        triage_config_path=SEVERITY_YAML,
        **extra,
    )
    return result


def test_pipeline_with_secondary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _setup(tmp_path, monkeypatch)
    result = _run(
        s,
        secondary_damage_weights=s["secondary"],
        secondary_damage_dataset_config=s["secondary_cfg"],
        merge_config_path=SEVERITY_YAML,
    )
    recs = result["instances"]
    assert [r["damage_type"] for r in recs] == ["dent", "missing_part"]
    assert [r["source_model"] for r in recs] == ["yolov8_seg_v1", "vehide_seg_v1"]
    assert result["damage_class_names"] == [
        "dent", "scratch", "torn", "missing_part", "punctured",
    ]


def test_pipeline_without_secondary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _setup(tmp_path, monkeypatch)
    result = _run(s)
    assert [r["source_model"] for r in result["instances"]] == ["yolov8_seg_v1"]
    assert result["damage_class_names"] == ["dent", "scratch"]


def test_missing_secondary_config_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        _run(s, secondary_damage_weights=s["secondary"])
    with pytest.raises(ValueError):
        _run(s, secondary_damage_weights=s["secondary"],
             secondary_damage_dataset_config=s["secondary_cfg"])
