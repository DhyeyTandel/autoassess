"""Guards for the demo restyle: banned colours/fonts, palettes, instance table."""

from __future__ import annotations

import colorsys
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoassess.infer import visualize

_APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("autoassess_restyle_app", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    return _APP_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("banned", ["#EE5308", "#C24303", "#FF8A4D", "fonts.googleapis.com"])
def test_old_accent_and_google_fonts_removed(banned: str) -> None:
    assert banned.lower() not in _source().lower()


def test_no_google_font_objects() -> None:
    assert "GoogleFont" not in _source()


def test_no_orange_or_purple_hex_in_app() -> None:
    hexes = set(re.findall(r"#([0-9A-Fa-f]{6})\b", _source()))
    assert hexes, "expected colour tokens in app.py"
    for h in hexes:
        r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
        hue, _light, sat = colorsys.rgb_to_hls(r, g, b)
        deg = hue * 360
        assert not (15 <= deg <= 45 and sat > 0.5), f"#{h} is orange"
        assert not (260 <= deg <= 310 and sat > 0.3), f"#{h} is purple"


def test_no_gradients_shadows_or_animation_in_app() -> None:
    src = _source()
    assert "gradient" not in src
    assert "@keyframes" not in src
    assert "box-shadow:" not in src.replace("box-shadow: none", "")


def test_overlay_palettes() -> None:
    assert visualize.SEVERITY_COLORS == {
        "minor": (26, 127, 55),
        "moderate": (143, 112, 0),
        "severe": (207, 34, 46),
    }
    assert visualize.PART_COLOR == (100, 116, 139)


def _luminance(rgb: tuple[int, int, int]) -> float:
    def lin(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_label_text_readable_on_every_fill() -> None:
    text = _luminance(visualize._LABEL_TEXT_COLOR)
    for color in visualize.SEVERITY_COLORS.values():
        lum = _luminance(color)
        hi, lo = max(text, lum), min(text, lum)
        assert (hi + 0.05) / (lo + 0.05) >= 4.5


def test_triage_meta_hex_values(app_module: ModuleType) -> None:
    meta = app_module.TRIAGE_META
    assert meta["auto_approve"]["hex"] == "#1A7F37"
    assert meta["total_loss_review"]["hex"] == "#CF222E"
    # Same colour as the overlay's moderate fill, rgb (143, 112, 0).
    assert meta["human_review"]["hex"] == "#8F7000"


def test_instances_table_rows_order_and_format(app_module: ModuleType) -> None:
    result = {
        "instances": [
            {"damage_type": "scratch", "part": "door", "severity": "minor", "confidence": 0.9},
            {"damage_type": "dent", "part": "unassigned", "severity": "severe", "confidence": 0.5},
            {"damage_type": "crack", "part": "hood", "severity": "severe", "confidence": 0.871},
            {"damage_type": "dent", "part": "bumper", "severity": "moderate", "confidence": 0.3},
        ]
    }
    assert app_module.instances_table(result) == [
        ["crack", "hood", "severe", "0.87"],
        ["dent", "—", "severe", "0.50"],
        ["dent", "bumper", "moderate", "0.30"],
        ["scratch", "door", "minor", "0.90"],
    ]


def test_instances_table_empty(app_module: ModuleType) -> None:
    assert app_module.instances_table({"instances": []}) == []


def test_header_and_notice(app_module: ModuleType) -> None:
    assert "AutoAssess" in app_module.HEADER_HTML
    assert "Research demo" in app_module.HEADER_HTML
    assert "grades severity and suggests how to route the claim." in app_module.HEADER_HTML


def test_moderate_colour_matches_triage_amber(app_module: ModuleType) -> None:
    r, g, b = visualize.SEVERITY_COLORS["moderate"]
    assert f"#{r:02X}{g:02X}{b:02X}" == app_module.TRIAGE_META["human_review"]["hex"]


def test_css_prevents_mobile_overflow(app_module: ModuleType) -> None:
    css = app_module.CSS
    assert ".gradio-container, .gradio-container * { min-width: 0; }" in css
    start = css.index(".aa-table {")
    rule = css[start : css.index("}", start)]
    assert "overflow-x: auto" in rule and "max-width: 100%" in rule
    assert "overflow-wrap: anywhere" in css


def test_css_table_uses_sans_tabular_figures(app_module: ModuleType) -> None:
    css = app_module.CSS
    start = css.index(".aa-table, .aa-table *")
    rule = css[start : css.index("}", start)]
    assert "font-family: var(--font)" in rule
    assert "font-variant-numeric: tabular-nums" in rule
    th = css[css.index(".aa-table th") :]
    th = th[: th.index("}")]
    assert "text-transform: none" in th and "font-weight: 500" in th
    assert "color: var(--text-2)" in th
