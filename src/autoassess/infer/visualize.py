"""Overlay renderer for the demo: part/damage fills, outlines, readable labels."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autoassess.infer.associate import MaskInstance

# Severity/triage palette is semantic (maps to real risk), kept separate from
# the UI's accent color. RGB tuples feed PIL overlay drawing; hex feeds CSS.
SEVERITY_COLORS = {"minor": (240, 200, 50), "moderate": (240, 140, 30), "severe": (220, 50, 50)}
PART_COLOR = (60, 140, 230)

_FALLBACK_COLOR = (150, 150, 150)
_LABEL_TEXT_COLOR = (30, 30, 30)

Box = tuple[float, float, float, float]


def label_font_size(image_size: tuple[int, int]) -> int:
    """Font size in px for an image of (width, height)."""
    return max(14, round(0.028 * min(image_size)))


def outline_width(font_size: int) -> int:
    """Outline width in px for a given label font size."""
    return max(2, round(font_size / 7))


def _decode_mask(inst: MaskInstance) -> np.ndarray:
    from pycocotools import mask as mask_utils

    return mask_utils.decode(inst.mask_rle).astype(bool)  # type: ignore[no-any-return]


def _collides(rect: tuple[int, int, int, int], placed: Sequence[tuple[int, int, int, int]]) -> bool:
    return any(
        rect[0] < p[2] and p[0] < rect[2] and rect[1] < p[3] and p[1] < rect[3] for p in placed
    )


def _rect_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance between two (x0, y0, x1, y1) rectangles; 0 when they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(math.hypot(dx, dy))


def _facing_coord(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float]:
    """Nearest coordinates on interval a and interval b along one axis."""
    if a1 < b0:
        return a1, b0
    if b1 < a0:
        return a0, b1
    mid = (max(a0, b0) + min(a1, b1)) / 2
    return mid, mid


def leader_line(
    label_rect: tuple[int, int, int, int], box: Box
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Segment from the label edge to the box edge, or None if they touch (gap <= 2 px)."""
    if _rect_gap(label_rect, box) <= 2:
        return None
    lx, bx = _facing_coord(label_rect[0], label_rect[2], box[0], box[2])
    ly, by = _facing_coord(label_rect[1], label_rect[3], box[1], box[3])
    return (round(lx), round(ly)), (round(bx), round(by))


def layout_labels(
    anchor_boxes: Sequence[Box],
    label_sizes: Sequence[tuple[int, int]],
    image_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Top-left (x, y) for each label so labels do not overlap and stay in the image.

    Labels are placed greedily in descending order of anchor-box area (ties keep
    input order); the result is returned in input order. Candidates per label:
    above the box's top-left, inside its top-left, below its bottom-left. If all
    collide, shift down from the first candidate by label height until one fits;
    if nothing fits, use the clamped first candidate.
    """
    img_w, img_h = image_size

    def clamp(x: float, y: float, w: int, h: int) -> tuple[int, int]:
        return (
            int(max(0, min(round(x), img_w - w))),
            int(max(0, min(round(y), img_h - h))),
        )

    order = sorted(
        range(len(anchor_boxes)),
        key=lambda i: -(
            (anchor_boxes[i][2] - anchor_boxes[i][0]) * (anchor_boxes[i][3] - anchor_boxes[i][1])
        ),
    )
    result: list[tuple[int, int]] = [(0, 0)] * len(anchor_boxes)
    placed: list[tuple[int, int, int, int]] = []
    for i in order:
        x0, y0, _, y1 = anchor_boxes[i]
        w, h = label_sizes[i]
        candidates = [
            clamp(x0, y0 - h, w, h),
            clamp(x0, y0, w, h),
            clamp(x0, y1, w, h),
        ]
        chosen: tuple[int, int] | None = None
        for cx, cy in candidates:
            if not _collides((cx, cy, cx + w, cy + h), placed):
                chosen = (cx, cy)
                break
        if chosen is None:
            chosen = _search_nearby(anchor_boxes[i], (w, h), placed, clamp)
        if chosen is None:
            cx, cy = candidates[0]
            step = max(1, h)
            while cy + h <= img_h:
                cy += step
                if cy + h > img_h:
                    break
                if not _collides((cx, cy, cx + w, cy + h), placed):
                    chosen = (cx, cy)
                    break
            if chosen is None:
                chosen = candidates[0]
        result[i] = chosen
        placed.append((chosen[0], chosen[1], chosen[0] + w, chosen[1] + h))
    return result


def _search_nearby(
    box: Box,
    label_size: tuple[int, int],
    placed: Sequence[tuple[int, int, int, int]],
    clamp: Callable[[float, float, int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    """Collision-free label position closest to ``box`` (ties: generation order)."""
    x0, y0, x1, _ = box
    w, h = label_size
    step = max(1, h)
    raw: list[tuple[float, float]] = [(x1, y0), (x0 - w, y0)]
    radius = 6
    for r in range(1, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) == r:
                    raw.append((x0 + dx * step, y0 + dy * step))
    best: tuple[int, int] | None = None
    best_gap = math.inf
    for rx, ry in raw:
        cx, cy = clamp(rx, ry, w, h)
        rect = (cx, cy, cx + w, cy + h)
        if _collides(rect, placed):
            continue
        gap = _rect_gap(rect, box)
        if gap < best_gap:
            best, best_gap = (cx, cy), gap
    return best


def draw_overlay(
    image: Image.Image,
    damage_instances: list[MaskInstance],
    part_instances: list[MaskInstance],
    severities: list[str],
    parts_assigned: list[str],
) -> Image.Image:
    """Fill part masks (light blue) and damage masks (colored by severity) into
    one RGBA layer, draw damage outlines, composite once, then draw opaque
    severity-colored labels on the composited RGB image."""
    fill = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    for part in part_instances:
        fill[_decode_mask(part)] = (*PART_COLOR, 60)
    for damage, severity in zip(damage_instances, severities, strict=True):
        color = SEVERITY_COLORS.get(severity, _FALLBACK_COLOR)
        fill[_decode_mask(damage)] = (*color, 110)

    font_size = label_font_size(image.size)
    width = outline_width(font_size)
    font = ImageFont.load_default(size=font_size)
    pad = max(2, round(font_size / 4))

    overlay = Image.fromarray(fill, "RGBA")
    odraw = ImageDraw.Draw(overlay)
    for damage, severity in zip(damage_instances, severities, strict=True):
        color = SEVERITY_COLORS.get(severity, _FALLBACK_COLOR)
        x0, y0, x1, y1 = damage.bbox
        odraw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=width)

    out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(out)

    texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    sizes: list[tuple[int, int]] = []
    for damage, severity, part_name in zip(
        damage_instances, severities, parts_assigned, strict=True
    ):
        shown = "no part" if part_name == "unassigned" else part_name
        text = f"{damage.class_name} · {severity} · {shown}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        texts.append(text)
        offsets.append((int(left), int(top)))
        sizes.append((int(right - left) + 2 * pad, int(bottom - top) + 2 * pad))

    positions = layout_labels([d.bbox for d in damage_instances], sizes, image.size)
    for damage, severity, (x, y), (w, h) in zip(
        damage_instances, severities, positions, sizes, strict=True
    ):
        seg = leader_line((x, y, x + w, y + h), damage.bbox)
        if seg is not None:
            color = SEVERITY_COLORS.get(severity, _FALLBACK_COLOR)
            draw.line([seg[0], seg[1]], fill=color, width=width)
    for text, severity, (x, y), (w, h), (ox, oy) in zip(
        texts, severities, positions, sizes, offsets, strict=True
    ):
        color = SEVERITY_COLORS.get(severity, _FALLBACK_COLOR)
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color)
        draw.text((x + pad - ox, y + pad - oy), text, font=font, fill=_LABEL_TEXT_COLOR)
    return out
