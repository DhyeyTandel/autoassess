"""Tests for autoassess.web.service (no web framework, no real weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from autoassess.web import service

PANELS = ["bonnet", "front_bumper", "rear_bumper", "door", "headlamp", "windshield"]


def _record(part: str, damage: str, severity: str, conf: float, area: float) -> dict[str, Any]:
    return {
        "part": part,
        "damage_type": damage,
        "severity": severity,
        "mask_area_px": area,
        "confidence": conf,
        "part_coverage": 0.9,
        "source_model": "yolov8_seg_v1",
    }


def _result() -> dict[str, Any]:
    return {
        "image_width": 40,
        "image_height": 30,
        "instances": [
            _record("door", "dent", "minor", 0.5, 100.0),
            _record("unassigned", "scratch", "severe", 0.61239, 200.0),
            _record("door", "crack", "moderate", 0.3, 50.0),
            _record("windshield", "glass shatter", "severe", 0.87184, 300.7),
            _record("door", "dent", "moderate", 0.9, 10.0),
        ],
        "triage": {
            "decision": "total_loss_review",
            "total_damaged_area_px": 660.7,
            "n_instances": 5,
            "n_severe": 2,
            "n_moderate": 2,
            "n_minor": 1,
        },
    }


def _build(**kw: object) -> dict[str, Any]:
    args: dict[str, Any] = {
        "result_dict": _result(),
        "overlay_image": Image.new("RGB", (40, 30), "red"),
        "sensitivity": "high_recall",
        "secondary_conf": 0.07,
        "elapsed_ms": 1234.56,
        "panel_names": PANELS,
    }
    args.update(kw)
    return service.build_response(**args)


def test_build_response_top_level_keys() -> None:
    out = _build()
    assert set(out) == {
        "triage", "instances", "panels", "unassigned", "overlay",
        "image", "sensitivity", "secondary_conf", "elapsed_ms",
    }
    assert out["image"] == {"width": 40, "height": 30}
    assert out["sensitivity"] == "high_recall"
    assert out["secondary_conf"] == 0.07
    assert out["elapsed_ms"] == pytest.approx(1234.6, abs=0.1)


def test_build_response_triage_label_and_fields() -> None:
    t = _build()["triage"]
    assert t == {
        "decision": "total_loss_review",
        "label": "Total-loss review",
        "n_instances": 5,
        "n_severe": 2,
        "n_moderate": 2,
        "n_minor": 1,
        "total_damaged_area_px": 661,
    }
    assert isinstance(t["total_damaged_area_px"], int)


@pytest.mark.parametrize(
    ("decision", "label"),
    [("human_review", "Human review"), ("auto_approve", "Auto-approve")],
)
def test_build_response_other_labels(decision: str, label: str) -> None:
    res = _result()
    res["triage"]["decision"] = decision
    assert _build(result_dict=res)["triage"]["label"] == label


def test_build_response_sort_null_panel_rounding() -> None:
    out = _build()
    inst = out["instances"]
    order = [(i["severity"], i["confidence"]) for i in inst]
    assert order == [
        ("severe", 0.8718),
        ("severe", 0.6124),
        ("moderate", 0.9),
        ("moderate", 0.3),
        ("minor", 0.5),
    ]
    assert inst[1]["panel"] is None
    assert inst[0] == {
        "damage": "glass shatter",
        "panel": "windshield",
        "severity": "severe",
        "confidence": 0.8718,
        "mask_area_px": 301,
        "source_model": "yolov8_seg_v1",
    }
    assert all(isinstance(i["mask_area_px"], int) for i in inst)


def test_build_response_panels_all_keys_and_unassigned() -> None:
    out = _build()
    assert out["panels"] == {
        "bonnet": 0, "front_bumper": 0, "rear_bumper": 0,
        "door": 3, "headlamp": 0, "windshield": 1,
    }
    assert out["unassigned"] == 1


def test_build_response_default_panel_names_from_config() -> None:
    out = service.build_response(
        _result(), Image.new("RGB", (4, 4)), "standard", None, 1.0
    )
    assert list(out["panels"]) == PANELS
    assert out["secondary_conf"] is None


def test_build_response_overlay_is_jpeg_data_url() -> None:
    import base64
    import io

    overlay = _build()["overlay"]
    prefix = "data:image/jpeg;base64,"
    assert overlay.startswith(prefix)
    img = Image.open(io.BytesIO(base64.b64decode(overlay[len(prefix):])))
    assert img.format == "JPEG"
    assert img.size == (40, 30)


def test_sensitivity_mapping() -> None:
    settings = service.resolve_damage_models({"SECONDARY_DAMAGE_CONF": "0.2"})
    assert service.secondary_conf_for("standard", settings) == 0.15
    assert service.secondary_conf_for("high_recall", settings) == 0.2
    default = service.resolve_damage_models({})
    assert service.secondary_conf_for("high_recall", default) == 0.07
    with pytest.raises(ValueError):
        service.secondary_conf_for("High recall", default)


def test_resolve_damage_models_defaults_and_errors() -> None:
    s = service.resolve_damage_models({})
    assert s.source == "merged"
    assert s.primary_weights == Path("runs/yolov8_seg_v1/weights/best.pt")
    assert s.secondary_weights == Path("runs/vehide_seg_v1/weights/best.pt")
    c = service.resolve_damage_models({"DAMAGE_MODEL_SOURCE": "cardd"})
    assert c.secondary_weights is None
    with pytest.raises(ValueError):
        service.resolve_damage_models({"DAMAGE_MODEL_SOURCE": "nope"})


def test_save_upload_lossless_roundtrip(tmp_path: Path) -> None:
    img = Image.new("RGB", (5, 5), (1, 2, 3))
    p = service.save_upload_lossless(img, tmp_path)
    assert p.suffix == ".png"
    assert Image.open(p).getpixel((0, 0)) == (1, 2, 3)


def test_temp_file_deleted_when_pipeline_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def boom(**kwargs: Path) -> None:
        seen.append(kwargs["source"])
        assert kwargs["source"].exists()
        raise RuntimeError("pipeline failed")

    a = service.Assessor(env={}, device="cpu", temp_dir=tmp_path, check_weights=False)
    monkeypatch.setattr(a, "_run_pipeline", boom)
    with pytest.raises(RuntimeError, match="pipeline failed"):
        a.assess(Image.new("RGB", (8, 8)), "high_recall")
    assert len(seen) == 1
    assert not seen[0].exists()
    assert list(tmp_path.iterdir()) == []


def test_assessor_missing_weights_raises(tmp_path: Path) -> None:
    a = service.Assessor(
        env={"DAMAGE_WEIGHTS": str(tmp_path / "nope.pt")}, device="cpu", temp_dir=tmp_path
    )
    with pytest.raises(FileNotFoundError):
        a.assess(Image.new("RGB", (8, 8)), "standard")


def test_load_image_rgb_and_bad_bytes() -> None:
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (6, 4), (1, 2, 3, 255)).save(buf, format="PNG")
    img = service.load_image(buf.getvalue())
    assert img.mode == "RGB" and img.size == (6, 4)
    with pytest.raises(ValueError):
        service.load_image(b"hello, not an image")
