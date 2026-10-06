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
from typing import Any

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
        app_module.assess(_noisy_rgb(), app_module.SENSITIVITY_STANDARD)
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


def test_secondary_conf_for_modes(app_module: ModuleType) -> None:
    default = app_module.resolve_damage_models({})
    assert app_module.secondary_conf_for(app_module.SENSITIVITY_STANDARD, default) == 0.15
    assert app_module.secondary_conf_for(app_module.SENSITIVITY_HIGH_RECALL, default) == 0.07
    custom = app_module.resolve_damage_models({"SECONDARY_DAMAGE_CONF": "0.3"})
    assert app_module.secondary_conf_for(app_module.SENSITIVITY_HIGH_RECALL, custom) == 0.3
    # Standard is fixed regardless of the env override.
    assert app_module.secondary_conf_for(app_module.SENSITIVITY_STANDARD, custom) == 0.15


def test_secondary_conf_for_unknown_mode_raises(app_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="Turbo"):
        app_module.secondary_conf_for("Turbo", app_module.resolve_damage_models({}))


def _stub_pipeline(
    app_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    merged: bool,
) -> dict[str, Any]:
    weights = tmp_path / "w.pt"
    weights.touch()
    monkeypatch.setattr(app_module, "DAMAGE_WEIGHTS", weights)
    monkeypatch.setattr(app_module, "PARTS_WEIGHTS", weights)
    monkeypatch.setattr(app_module, "SECONDARY_DAMAGE_WEIGHTS", weights if merged else None)
    monkeypatch.setattr(app_module, "_damage_models", app_module.resolve_damage_models({}))
    recorded: dict[str, Any] = {}

    def fake_pipeline(**kwargs: object) -> tuple[dict[str, Any], list[Any], list[Any]]:
        recorded.update(kwargs)
        triage = {
            "decision": "auto_approve",
            "n_severe": 0,
            "n_moderate": 0,
            "n_minor": 0,
            "n_instances": 0,
            "total_damaged_area_px": 0.0,
        }
        return {"instances": [], "triage": triage}, [], []

    monkeypatch.setattr(app_module, "run_pipeline_with_masks", fake_pipeline)
    monkeypatch.setattr(app_module, "draw_overlay", lambda image, *a, **k: image)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return recorded


@pytest.mark.parametrize(
    ("mode_attr", "expected"),
    [("SENSITIVITY_STANDARD", 0.15), ("SENSITIVITY_HIGH_RECALL", 0.07)],
)
def test_assess_passes_sensitivity_cutoff(
    app_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode_attr: str,
    expected: float,
) -> None:
    recorded = _stub_pipeline(app_module, tmp_path, monkeypatch, merged=True)
    mode = getattr(app_module, mode_attr)
    _, result, _, _ = app_module.assess(_noisy_rgb(), mode)
    assert recorded["secondary_damage_conf"] == expected
    assert result["settings"] == {"sensitivity": mode, "secondary_conf": expected}


def test_assess_single_model_has_no_secondary(
    app_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _stub_pipeline(app_module, tmp_path, monkeypatch, merged=False)
    _, result, _, _ = app_module.assess(_noisy_rgb(), app_module.SENSITIVITY_STANDARD)
    assert recorded["secondary_damage_weights"] is None
    assert recorded["secondary_damage_conf"] is None
    assert result["settings"] == {"sensitivity": None, "secondary_conf": None}


def test_demo_has_sensitivity_radio_wired_to_submit(app_module: ModuleType) -> None:
    components = app_module.demo.config["components"]
    radios = [c for c in components if c["type"] == "radio"]
    assert len(radios) == 1
    props = radios[0]["props"]
    assert props["label"] == "Sensitivity"
    assert [c[1] for c in props["choices"]] == [
        app_module.SENSITIVITY_HIGH_RECALL,
        app_module.SENSITIVITY_STANDARD,
    ]
    assert props["value"] == app_module.SENSITIVITY_HIGH_RECALL
    submit_id = app_module.submit_btn._id
    deps = [
        d
        for d in app_module.demo.config["dependencies"]
        if any(t[0] == submit_id and t[1] == "click" for t in d["targets"])
    ]
    assert len(deps) == 1
    assert deps[0]["inputs"] == [app_module.image_input._id, radios[0]["id"]]


def _triage(decision: str = "total_loss_review", **over: object) -> dict[str, Any]:
    t = {
        "decision": decision,
        "n_severe": 2,
        "n_moderate": 1,
        "n_minor": 0,
        "n_instances": 3,
        "total_damaged_area_px": 12345.0,
    }
    t.update(over)
    return t


def test_assess_returns_results_visibility_update(
    app_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pipeline(app_module, tmp_path, monkeypatch, merged=False)
    out = app_module.assess(_noisy_rgb(), app_module.SENSITIVITY_STANDARD)
    assert len(out) == 4
    assert out[3]["visible"] is True


def test_results_column_hidden_at_load_and_wired_to_submit(app_module: ModuleType) -> None:
    results_id = app_module.results_column._id
    comp = next(c for c in app_module.demo.config["components"] if c["id"] == results_id)
    assert comp["props"]["visible"] is False
    submit_id = app_module.submit_btn._id
    deps = [
        d
        for d in app_module.demo.config["dependencies"]
        if any(t[0] == submit_id and t[1] == "click" for t in d["targets"])
    ]
    assert len(deps) == 1
    assert results_id in deps[0]["outputs"]
    assert len(deps[0]["outputs"]) == 4


def _non_ascii_emoji(text: str) -> list[str]:
    allowed = {"\u2192", "\u25b8", "\u2022"}
    bad = []
    for ch in text:
        cp = ord(ch)
        if ch in allowed:
            continue
        if 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
            bad.append(ch)
    return bad


@pytest.mark.parametrize("decision", ["auto_approve", "human_review", "total_loss_review"])
def test_triage_card_content_and_no_emoji(app_module: ModuleType, decision: str) -> None:
    html = app_module.render_triage_card(_triage(decision))
    label = app_module.TRIAGE_META[decision]["label"]
    assert label in html
    assert "heuristic, not adjuster-validated" in html
    assert "VERDICT" in html
    assert _non_ascii_emoji(html) == []


def test_triage_card_zero_detection_copy_kept(app_module: ModuleType) -> None:
    html = app_module.render_triage_card(
        _triage("auto_approve", n_severe=0, n_moderate=0, n_minor=0, n_instances=0)
    )
    assert "Nothing detected" in html
    assert "verify manually" in html


def test_image_input_sources_upload_only(app_module: ModuleType) -> None:
    comp = next(
        c for c in app_module.demo.config["components"] if c["id"] == app_module.image_input._id
    )
    assert comp["props"]["sources"] == ["upload"]


def test_results_column_has_aa_results_class(app_module: ModuleType) -> None:
    comp = next(
        c for c in app_module.demo.config["components"] if c["id"] == app_module.results_column._id
    )
    assert "aa-results" in comp["props"]["elem_classes"]


def test_submit_chains_client_side_scroll_to_results(app_module: ModuleType) -> None:
    deps = app_module.demo.config["dependencies"]
    submit_id = app_module.submit_btn._id
    submit_dep = next(
        d for d in deps if any(t[0] == submit_id and t[1] == "click" for t in d["targets"])
    )
    chained = [d for d in deps if d.get("trigger_after") == submit_dep["id"]]
    assert len(chained) == 1
    js = chained[0]["js"]
    assert js is not None
    assert "scrollIntoView" in js
    assert "aa-results" in js
    assert "prefers-reduced-motion" in js


def test_css_neutralises_form_wrapper_of_sensitivity_only(app_module: ModuleType) -> None:
    css = app_module.CSS
    assert ".form:has(> .aa-sens)" in css
    start = css.index(".form:has(> .aa-sens)")
    rule = css[start : css.index("}", start)]
    for decl in ("background: transparent", "border: none", "box-shadow: none"):
        assert decl in rule
    # No blanket .form restyle elsewhere.
    assert css.count(".form") == css.count(".form:has(> .aa-sens)")


def test_css_mobile_media_block_enlarges_image_containers(app_module: ModuleType) -> None:
    # String-level check only; the rendered widths are verified in a real browser.
    css = app_module.CSS
    start = css.index("@media (max-width: 640px)")
    block = css[start : css.index("\n}\n", start)]
    assert '[data-testid="image"]' in block
    assert ".aa-card" in block and ".aa-results" in block
    assert "padding: 12px" in block
    assert "padding: 0" in block


def test_head_style_holds_unscoped_main_padding_and_footer_rules(app_module: ModuleType) -> None:
    head = app_module.HEAD_STYLE
    assert head.startswith("<style>") and head.rstrip().endswith("</style>")
    media = head[head.index("@media (max-width: 640px)") :]
    rule = media[media.index(".gradio-container .main") : media.index("}")]
    assert "padding-left: 0 !important" in rule
    assert "padding-right: 0 !important" in rule
    assert "footer .divider { display: none !important; }" in head


def test_head_style_is_passed_to_launch(app_module: ModuleType) -> None:
    # gradio 6.22 accepts `head` on launch() only (gr.Blocks has no such parameter), so
    # assert on the launch call via source inspection.
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    launch = source[source.index("demo.launch(") :]
    assert "head=HEAD_STYLE" in launch[: launch.index(")")]


def test_css_no_longer_holds_unscopable_rules(app_module: ModuleType) -> None:
    assert ".main {" not in app_module.CSS
    assert "footer" not in app_module.CSS
