"""Tests for the demo's env-to-settings resolution (no Gradio server is launched).

`app.py` builds a `gr.Blocks` at import time but only launches under
`__main__`, so importing it is side-effect free apart from a little start-up
cost. It is loaded by file path so the test does not depend on sys.path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
    assert s.secondary_conf == 0.15


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
