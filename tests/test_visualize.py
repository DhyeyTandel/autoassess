"""Tests for the demo overlay renderer (CPU only, synthetic data)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

from autoassess.infer.associate import MaskInstance
from autoassess.infer.visualize import (
    SEVERITY_COLORS,
    draw_overlay,
    label_font_size,
    layout_labels,
    leader_line,
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


Rect = tuple[int, int, int, int]


def _gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _crowded_boxes(n: int = 14) -> list[tuple[int, int, int, int]]:
    # Heavily overlapping boxes clustered in the middle of an 800x600 image.
    return [(300 + 6 * i, 220 + 5 * (i % 7), 480 + 6 * i, 360 + 5 * (i % 7)) for i in range(n)]


def test_layout_crowded_labels_stay_close_to_boxes() -> None:
    size = (800, 600)
    boxes = _crowded_boxes()
    sizes = [(200, 24)] * len(boxes)
    pos = layout_labels(boxes, sizes, size)
    rects = [(x, y, x + w, y + h) for (x, y), (w, h) in zip(pos, sizes, strict=True)]
    for i, r in enumerate(rects):
        assert 0 <= r[0] and 0 <= r[1] and r[2] <= size[0] and r[3] <= size[1]
        for r2 in rects[i + 1 :]:
            assert not _intersects(r, r2)
    max_gap = max(_gap(r, b) for r, b in zip(rects, boxes, strict=True))
    assert max_gap <= 4 * 24, max_gap


def test_leader_line_none_when_touching_or_overlapping() -> None:
    box = (100.0, 100.0, 200.0, 200.0)
    assert leader_line((120, 80, 180, 100), box) is None  # touching
    assert leader_line((120, 90, 180, 110), box) is None  # overlapping
    assert leader_line((120, 150, 180, 170), box) is None  # inside
    assert leader_line((120, 78, 180, 98), box) is None  # gap of 2 px


def test_leader_line_endpoints_on_edges() -> None:
    box = (100.0, 100.0, 200.0, 200.0)
    # Directly above (x ranges overlap).
    rect = (120, 40, 180, 60)
    seg = leader_line(rect, box)
    assert seg is not None
    (lx, ly), (bx, by) = seg
    assert ly == rect[3] and rect[0] <= lx <= rect[2]
    assert by == box[1] and box[0] <= bx <= box[2]
    # Diagonal (below-right): corner to corner.
    rect = (260, 300, 320, 320)
    seg = leader_line(rect, box)
    assert seg is not None
    assert seg == ((260, 300), (200, 200))


def test_draw_overlay_leader_line_has_severity_colour() -> None:
    size = (800, 600)
    img = Image.new("RGB", size, (255, 255, 255))
    boxes = _crowded_boxes(40)
    insts = [_inst("dent", b, size) for b in boxes]
    sevs = ["severe" if i % 2 else "minor" for i in range(len(insts))]
    out = draw_overlay(img, insts, [], sevs, ["hood"] * len(insts))

    font_size = label_font_size(size)
    font = ImageFont.load_default(size=font_size)
    pad = max(2, round(font_size / 4))
    draw = ImageDraw.Draw(img)
    sizes = []
    for s in sevs:
        x0, t, x1, b = draw.textbbox((0, 0), f"dent · {s} · hood", font=font)
        sizes.append((int(x1 - x0) + 2 * pad, int(b - t) + 2 * pad))
    pos = layout_labels(boxes, sizes, size)
    rects = [(x, y, x + w, y + h) for (x, y), (w, h) in zip(pos, sizes, strict=True)]
    segs = [leader_line(r, tuple(map(float, b))) for r, b in zip(rects, boxes, strict=True)]
    assert any(s is not None for s in segs)
    checked = 0
    for seg, sev, own in zip(segs, sevs, range(len(segs)), strict=True):
        if seg is None:
            continue
        mx = round((seg[0][0] + seg[1][0]) / 2)
        my = round((seg[0][1] + seg[1][1]) / 2)
        if any(r[0] - 1 <= mx <= r[2] and r[1] - 1 <= my <= r[3] for r in rects):
            continue
        others = [s for k, s in enumerate(segs) if s is not None and k != own]
        if any(
            abs((s[1][1] - s[0][1]) * (mx - s[0][0]) - (s[1][0] - s[0][0]) * (my - s[0][1]))
            < 4 * max(1, np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
            for s in others
        ):
            continue
        assert out.getpixel((mx, my)) == SEVERITY_COLORS[sev]
        checked += 1
    assert checked > 0
