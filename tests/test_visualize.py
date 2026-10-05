"""Tests for the demo overlay renderer (CPU only, synthetic data)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from autoassess.infer.associate import MaskInstance
from autoassess.infer.visualize import (
    SEVERITY_COLORS,
    draw_overlay,
    label_font_size,
    layout_labels,
    outline_width,
)


def _inst(name: str, box: tuple[int, int, int, int], size: tuple[int, int]) -> MaskInstance:
    w, h = size
    m = np.zeros((h, w), dtype=np.uint8, order="F")
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(m)
    return MaskInstance(
        class_name=name, mask_rle=rle, bbox=(x0, y0, x1, y1),
        mask_area=float(m.sum()), confidence=0.9,
    )


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_layout_overlapping_boxes_do_not_collide() -> None:
    size = (400, 300)
    boxes = [(100 + 5 * i, 100 + 3 * i, 220 + 5 * i, 200 + 3 * i) for i in range(6)]
    sizes = [(120, 24)] * 6
    pos = layout_labels(boxes, sizes, size)
    rects = [(x, y, x + w, y + h) for (x, y), (w, h) in zip(pos, sizes, strict=True)]
    for i, r in enumerate(rects):
        assert 0 <= r[0] and 0 <= r[1] and r[2] <= size[0] and r[3] <= size[1]
        for r2 in rects[i + 1 :]:
            assert not _intersects(r, r2)


def test_layout_clamps_at_edges() -> None:
    size = (200, 150)
    boxes = [(10, 0, 60, 40), (170, 60, 200, 100)]
    sizes = [(80, 20), (90, 20)]
    pos = layout_labels(boxes, sizes, size)
    for (x, y), (w, h) in zip(pos, sizes, strict=True):
        assert x >= 0 and y >= 0 and x + w <= size[0] and y + h <= size[1]


def test_font_size_scales_with_image() -> None:
    assert label_font_size((100, 100)) == 14
    assert label_font_size((1000, 800)) == round(0.028 * 800)
    assert label_font_size((4000, 3000)) > label_font_size((1000, 800)) > 14
    assert outline_width(14) == 2
    assert outline_width(56) == 8


def test_draw_overlay_label_is_opaque_severity_colour() -> None:
    size = (100, 100)
    img = Image.new("RGB", size, (255, 255, 255))
    d1 = _inst("dent", (20, 40, 60, 80), size)
    d2 = _inst("scratch", (30, 45, 70, 85), size)
    part = _inst("hood", (10, 30, 90, 95), size)
    out = draw_overlay(img, [d1, d2], [part], ["severe", "minor"], ["hood", "unassigned"])
    assert out.mode == "RGB" and out.size == size
    # First instance (larger-or-equal area, placed first) label sits just above its box.
    assert out.getpixel((21, 37)) == SEVERITY_COLORS["severe"]
