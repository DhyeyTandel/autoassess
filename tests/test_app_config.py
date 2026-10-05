"""Tests for the demo's env-to-settings resolution (no Gradio server is launched).

`app.py` builds a `gr.Blocks` at import time but only launches under
`__main__`, so importing it is side-effect free apart from a little start-up
cost. It is loaded by file path so the test does not depend on sys.path.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

_APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("autoassess_demo_app", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def test_default_is_merged(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models({})
    assert s.source == "merged"
    assert s.primary_weights == Path("runs/yolov8_seg_v1/weights/best.pt")
    assert s.primary_config == Path("configs/cardd.yaml")
    assert s.secondary_weights == Path("runs/vehide_seg_v1/weights/best.pt")
    assert s.secondary_config == Path("configs/vehide.yaml")
    assert s.secondary_conf == 0.07


def test_cardd_single_model(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models({"DAMAGE_MODEL_SOURCE": "cardd"})
    assert s.primary_weights == Path("runs/yolov8_seg_v1/weights/best.pt")
    assert s.primary_config == Path("configs/cardd.yaml")
    assert s.secondary_weights is None and s.secondary_config is None


def test_vehide_single_model(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models({"DAMAGE_MODEL_SOURCE": "vehide"})
    assert s.primary_weights == Path("runs/vehide_seg_v1/weights/best.pt")
    assert s.primary_config == Path("configs/vehide.yaml")
    assert s.secondary_weights is None and s.secondary_config is None


def test_unknown_source_raises(app_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="nope"):
        app_module.resolve_damage_models({"DAMAGE_MODEL_SOURCE": "nope"})


def test_overrides_apply(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models(
        {
            "SECONDARY_DAMAGE_CONF": "0.3",
            "DAMAGE_WEIGHTS": "x/primary.pt",
            "SECONDARY_DAMAGE_WEIGHTS": "x/secondary.pt",
        }
    )
    assert s.secondary_conf == 0.3
    assert s.primary_weights == Path("x/primary.pt")
    assert s.secondary_weights == Path("x/secondary.pt")
    # Overrides replace weights only; configs stay keyed to the source.
    assert s.primary_config == Path("configs/cardd.yaml")
    assert s.secondary_config == Path("configs/vehide.yaml")


def test_secondary_weights_override_ignored_for_single_model(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models(
        {"DAMAGE_MODEL_SOURCE": "cardd", "SECONDARY_DAMAGE_WEIGHTS": "x/secondary.pt"}
    )
    assert s.secondary_weights is None


def test_secondary_conf_env_restores_precision_setting(app_module: ModuleType) -> None:
    s = app_module.resolve_damage_models({"SECONDARY_DAMAGE_CONF": "0.15"})
    assert s.secondary_conf == 0.15


def _noisy_rgb(size: tuple[int, int] = (32, 24)) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_save_upload_lossless_roundtrips_pixels(app_module: ModuleType, tmp_path: Path) -> None:
    img = _noisy_rgb()
    path = app_module.save_upload_lossless(img, tmp_path)
    assert path.parent == tmp_path and path.suffix == ".png"
    with Image.open(path) as saved:
        assert saved.mode == "RGB"
        assert np.array_equal(np.asarray(saved), np.asarray(img))


def test_save_upload_lossless_unique_paths(app_module: ModuleType, tmp_path: Path) -> None:
    img = _noisy_rgb()
    a = app_module.save_upload_lossless(img, tmp_path)
    b = app_module.save_upload_lossless(img, tmp_path)
    assert a != b


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_save_upload_lossless_converts_to_rgb(
    app_module: ModuleType, tmp_path: Path, mode: str
) -> None:
    img = _noisy_rgb().convert(mode)
    path = app_module.save_upload_lossless(img, tmp_path)
    with Image.open(path) as saved:
        assert saved.mode == "RGB"
        assert np.array_equal(np.asarray(saved), np.asarray(img.convert("RGB")))


def test_assess_removes_temp_file_when_pipeline_raises(
    app_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "w.pt"
    weights.touch()
    monkeypatch.setattr(app_module, "DAMAGE_WEIGHTS", weights)
    monkeypatch.setattr(app_module, "PARTS_WEIGHTS", weights)
    monkeypatch.setattr(app_module, "SECONDARY_DAMAGE_WEIGHTS", None)

    seen: list[Path] = []

    def boom(**kwargs: Path) -> None:
        seen.append(kwargs["source"])
        assert kwargs["source"].exists()
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "run_pipeline_with_masks", boom)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        app_module.assess(_noisy_rgb())
    assert len(seen) == 1 and not seen[0].exists()


def test_normalize_upload_none(app_module: ModuleType) -> None:
    assert app_module.normalize_upload(None) is None


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_normalize_upload_returns_rgb(app_module: ModuleType, mode: str) -> None:
    img = _noisy_rgb().convert(mode)
    out = app_module.normalize_upload(img)
    assert out is not img
    assert out.mode == "RGB"
    assert out.size == img.size


def test_normalize_upload_applies_exif_orientation(app_module: ModuleType) -> None:
    img = _noisy_rgb((32, 24))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW to display -> width/height swap
    reloaded = Image.open(io.BytesIO(_save_with_exif(img, exif)))
    out = app_module.normalize_upload(reloaded)
    assert out.size == (24, 32)
    # Idempotent: gradio already transposes, and the tag is gone after one pass.
    assert app_module.normalize_upload(out).size == (24, 32)


def _save_with_exif(img: Image.Image, exif: Image.Exif) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif)
    return buf.getvalue()


def test_demo_has_upload_dependency_on_image_input(app_module: ModuleType) -> None:
    image_id = app_module.image_input._id
    deps = app_module.demo.config["dependencies"]
    matches = [
        d
        for d in deps
        if any(t[0] == image_id and t[1] == "upload" for t in d["targets"])
    ]
    assert len(matches) == 1
    assert matches[0]["inputs"] == [image_id]
    assert matches[0]["outputs"] == [image_id]
    assert app_module.image_input.format == "png"
