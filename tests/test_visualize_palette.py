"""Palette tests for the overlay renderer: no orange/purple, readable labels."""

from __future__ import annotations

import colorsys

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from autoassess.infer.associate import MaskInstance
from autoassess.infer.visualize import (
    _LABEL_TEXT_COLOR,
    PART_COLOR,
    SEVERITY_COLORS,
    draw_overlay,
)

Rgb = tuple[int, int, int]
OLD_ORANGE = (240, 140, 30)
ALL_COLORS: dict[str, Rgb] = {**SEVERITY_COLORS, "part": PART_COLOR}


def _hsl(rgb: Rgb) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in rgb)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s, lightness


def _luminance(rgb: Rgb) -> float:
    def lin(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_palette_values() -> None:
    assert SEVERITY_COLORS == {
        "minor": (26, 127, 55),
        "moderate": (143, 112, 0),
        "severe": (207, 34, 46),
    }
    assert PART_COLOR == (100, 116, 139)
    assert _LABEL_TEXT_COLOR == (255, 255, 255)


@pytest.mark.parametrize("name", list(ALL_COLORS))
def test_no_orange_or_purple(name: str) -> None:
    hue, sat, _ = _hsl(ALL_COLORS[name])
    assert not (15 <= hue <= 45 and sat > 0.5), f"{name} is orange"
    assert not (260 <= hue <= 310 and sat > 0.3), f"{name} is purple"


@pytest.mark.parametrize("name", list(SEVERITY_COLORS))
def test_white_label_text_wcag_aa(name: str) -> None:
    lum = _luminance(SEVERITY_COLORS[name])
    ratio = (_luminance(_LABEL_TEXT_COLOR) + 0.05) / (lum + 0.05)
    assert ratio >= 4.5


def test_draw_overlay_moderate_has_no_old_orange() -> None:
    size = (100, 100)
    m = np.zeros((size[1], size[0]), dtype=np.uint8, order="F")
    m[40:80, 20:60] = 1
    inst = MaskInstance(
        class_name="dent", mask_rle=mask_utils.encode(m), bbox=(20, 40, 60, 80),
        mask_area=float(m.sum()), confidence=0.9,
    )
    out = draw_overlay(Image.new("RGB", size, (255, 255, 255)), [inst], [], ["moderate"], ["hood"])
    pixels = {out.getpixel((x, y)) for x in range(size[0]) for y in range(size[1])}
    assert OLD_ORANGE not in pixels
    assert SEVERITY_COLORS["moderate"] in pixels  # outline / label are opaque
