"""Gradio demo — upload an image, see overlaid masks, structured JSON, and the triage decision.

Runs the same pipeline as the CLI (`autoassess-pipeline`, backed by
`autoassess.infer.pipeline.run_pipeline_with_masks`); this file only adds an
upload UI and a mask-overlay renderer around that one call. Weight paths
come from env vars (see below) since this is a thin demo shell, not a
config-driven training script.

Usage
-----
    DAMAGE_WEIGHTS=runs/yolov8_seg_v1/weights/best.pt \\
    PARTS_WEIGHTS=runs/parts_seg_v1/weights/best.pt \\
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from autoassess.infer.associate import MaskInstance
from autoassess.infer.pipeline import (
    AUTO_APPROVE,
    HUMAN_REVIEW,
    TOTAL_LOSS_REVIEW,
    run_pipeline_with_masks,
)

DAMAGE_WEIGHTS = Path(os.environ.get("DAMAGE_WEIGHTS", "runs/yolov8_seg_v1/weights/best.pt"))
PARTS_WEIGHTS = Path(os.environ.get("PARTS_WEIGHTS", "runs/parts_seg_v1/weights/best.pt"))
DAMAGE_DATASET_CONFIG = Path("configs/cardd.yaml")
PARTS_DATASET_CONFIG = Path("configs/carparts.yaml")
SEVERITY_CONFIG = Path("configs/severity.yaml")
DEVICE = os.environ.get("AUTOASSESS_DEVICE", "cpu")
UPLOAD_TMP_PATH = Path("/tmp/autoassess_upload.jpg")

# Severity/triage palette is semantic (maps to real risk), kept separate from
# the UI's accent color. RGB tuples feed PIL overlay drawing; hex feeds CSS.
SEVERITY_COLORS = {"minor": (240, 200, 50), "moderate": (240, 140, 30), "severe": (220, 50, 50)}
PART_COLOR = (60, 140, 230)

TRIAGE_META = {
    AUTO_APPROVE: {"label": "Auto-approve", "hex": "#2f9e58", "glyph": "check"},
    HUMAN_REVIEW: {"label": "Flag for human review", "hex": "#c8871a", "glyph": "flag"},
    TOTAL_LOSS_REVIEW: {"label": "Total-loss candidate", "hex": "#c1392b", "glyph": "alert"},
}

_GLYPH_PATHS = {
    "check": '<path d="M5 13l4 4L19 7" stroke="currentColor" stroke-width="2.5" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "flag": '<path d="M6 21V4h11l-3 4 3 4H6" stroke="currentColor" stroke-width="2.5" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "alert": '<path d="M12 4l9 16H3z" stroke="currentColor" stroke-width="2.5" fill="none" '
    'stroke-linejoin="round"/><path d="M12 10v4" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round"/><circle cx="12" cy="17" r="1.15" fill="currentColor"/>',
}


def _icon(glyph: str) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
        f"{_GLYPH_PATHS[glyph]}</svg>"
    )


def _decode_mask(inst: MaskInstance) -> np.ndarray:
    from pycocotools import mask as mask_utils

    return mask_utils.decode(inst.mask_rle).astype(bool)  # type: ignore[no-any-return]


def draw_overlay(
    image: Image.Image,
    damage_instances: list[MaskInstance],
    part_instances: list[MaskInstance],
    severities: list[str],
    parts_assigned: list[str],
) -> Image.Image:
    """Fill part masks (light blue) and damage masks (colored by severity)
    into one RGBA layer, then draw a labeled outline per damage instance,
    and composite the layer onto the image once at the end."""
    fill = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    for part in part_instances:
        fill[_decode_mask(part)] = (*PART_COLOR, 60)
    for damage, severity in zip(damage_instances, severities, strict=True):
        color = SEVERITY_COLORS.get(severity, (150, 150, 150))
        fill[_decode_mask(damage)] = (*color, 110)

    overlay = Image.fromarray(fill, "RGBA")
    draw = ImageDraw.Draw(overlay)
    triples = zip(damage_instances, severities, parts_assigned, strict=True)
    for damage, severity, part_name in triples:
        color = SEVERITY_COLORS.get(severity, (150, 150, 150))
        x0, y0, x1, y1 = damage.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=2)
        label = f"{damage.class_name} ({severity}) @ {part_name}"
        draw.text((x0 + 2, max(y0 - 14, 0)), label, fill=(*color, 255))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_triage_card(triage: dict[str, Any]) -> str:
    """Build the triage verdict as a status card (severity-striped, iconed) rather
    than markdown bullets — this is the one number a reviewer scans for first,
    so it gets the visual weight instead of sitting below the raw JSON."""
    decision = triage["decision"]
    meta = TRIAGE_META.get(decision, {"label": decision, "hex": "#6b7280", "glyph": "flag"})
    counts = [
        ("severe", triage["n_severe"], SEVERITY_COLORS["severe"]),
        ("moderate", triage["n_moderate"], SEVERITY_COLORS["moderate"]),
        ("minor", triage["n_minor"], SEVERITY_COLORS["minor"]),
    ]
    chips = "".join(
        f'<span class="aa-chip" style="--chip:{_rgb_hex(rgb)}">'
        f'<span class="aa-chip-dot"></span>{count} {name}</span>'
        for name, count, rgb in counts
        if count > 0
    ) or '<span class="aa-chip aa-chip-muted">No damage instances detected</span>'

    return f"""
<div class="aa-verdict" style="--verdict:{meta['hex']}">
  <div class="aa-verdict-icon">{_icon(meta['glyph'])}</div>
  <div class="aa-verdict-body">
    <div class="aa-verdict-label">{meta['label']}</div>
    <div class="aa-verdict-chips">{chips}</div>
    <div class="aa-verdict-meta">
      {triage['n_instances']} instance{'s' if triage['n_instances'] != 1 else ''} detected
      &middot; {triage['total_damaged_area_px']:,.0f}px total damaged area
    </div>
  </div>
</div>
""".strip()


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def assess(image: Image.Image) -> tuple[Image.Image, dict[str, Any], str]:
    if image is None:
        raise gr.Error("Upload an image first.")
    if not DAMAGE_WEIGHTS.exists() or not PARTS_WEIGHTS.exists():
        raise gr.Error(
            f"Model weights not found (looked for {DAMAGE_WEIGHTS} and {PARTS_WEIGHTS}). "
            "Set DAMAGE_WEIGHTS / PARTS_WEIGHTS env vars, or train the models first."
        )

    image.convert("RGB").save(UPLOAD_TMP_PATH)

    result, damage_instances, part_instances = run_pipeline_with_masks(
        source=UPLOAD_TMP_PATH,
        damage_weights=DAMAGE_WEIGHTS,
        parts_weights=PARTS_WEIGHTS,
        damage_dataset_config=DAMAGE_DATASET_CONFIG,
        parts_dataset_config=PARTS_DATASET_CONFIG,
        severity_config_path=SEVERITY_CONFIG,
        triage_config_path=SEVERITY_CONFIG,
        device=DEVICE,
    )

    severities = [rec["severity"] for rec in result["instances"]]
    parts_assigned = [rec["part"] for rec in result["instances"]]
    overlay_image = draw_overlay(
        image, damage_instances, part_instances, severities, parts_assigned
    )

    return overlay_image, result, render_triage_card(result["triage"])


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    text_size=gr.themes.sizes.text_md,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
    font=("ui-sans-serif", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"),
    font_mono=("ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"),
).set(
    body_background_fill="#f4f1eb",
    body_background_fill_dark="#1b1815",
    background_fill_primary="#fffdf9",
    background_fill_primary_dark="#242019",
    border_color_primary="#ded6c6",
    border_color_primary_dark="#3a342a",
    block_background_fill="#fffdf9",
    block_background_fill_dark="#242019",
    block_border_color="#ded6c6",
    block_border_color_dark="#3a342a",
    block_label_text_color="#7a6f58",
    block_label_text_color_dark="#a89b7f",
    block_title_text_color="#3d3626",
    block_title_text_color_dark="#e8e1d3",
    body_text_color="#3d3626",
    body_text_color_dark="#e8e1d3",
    body_text_color_subdued="#8a7f68",
    body_text_color_subdued_dark="#a89b7f",
    button_primary_background_fill="#c15a2e",
    button_primary_background_fill_hover="#a84c25",
    button_primary_text_color="#fffdf9",
    button_primary_border_color="#c15a2e",
)

CSS = """
:root {
  --aa-verdict-approve: #2f9e58;
  --aa-verdict-review: #c8871a;
  --aa-verdict-loss: #c1392b;
}

.aa-eyebrow {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: #b0714a;
  margin: 0 0 0.35rem 0;
}
:root:not([data-theme="light"]) .aa-eyebrow { color: #e0995f; }
:root[data-theme="dark"] .aa-eyebrow { color: #e0995f; }

.aa-title {
  font-size: 1.9rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin: 0 0 0.4rem 0;
  text-wrap: balance;
}

.aa-subtitle {
  font-size: 0.98rem;
  color: var(--body-text-color-subdued);
  max-width: 62ch;
  line-height: 1.5;
  margin: 0 0 1.1rem 0;
}

/* Triage verdict card: severity color as a left stripe + icon chip, not text alone */
.aa-verdict {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  padding: 1.1rem 1.25rem;
  border-radius: var(--radius-lg);
  border: 1px solid var(--verdict);
  border-left-width: 5px;
  background: color-mix(in srgb, var(--verdict) 8%, var(--block-background-fill));
}

.aa-verdict-icon {
  flex: none;
  width: 2.1rem;
  height: 2.1rem;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  background: var(--verdict);
}

.aa-verdict-label {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--verdict);
  line-height: 1.25;
}

.aa-verdict-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.55rem 0 0.5rem 0;
}

.aa-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.82rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem 0.2rem 0.5rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--chip) 16%, var(--block-background-fill));
  color: var(--body-text-color);
}

.aa-chip-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: var(--chip);
  flex: none;
}

.aa-chip-muted {
  background: transparent;
  border: 1px dashed var(--border-color-primary);
  color: var(--body-text-color-subdued);
  font-weight: 500;
}

.aa-verdict-meta {
  font-size: 0.84rem;
  color: var(--body-text-color-subdued);
  font-variant-numeric: tabular-nums;
}

.aa-footer {
  text-align: center;
  font-size: 0.8rem;
  color: var(--body-text-color-subdued);
  margin-top: 0.5rem;
}
"""

with gr.Blocks(title="AutoAssess — Damage Triage") as demo:
    gr.HTML(
        '<div class="aa-eyebrow">Claims triage &middot; computer vision</div>'
        '<div class="aa-title">AutoAssess</div>'
        '<p class="aa-subtitle">Upload a vehicle photo to localize damage, '
        "identify the affected panel, grade severity, and get a routing "
        "decision for the claim.</p>"
    )
    with gr.Row(equal_height=False):
        with gr.Column(min_width=320):
            image_input = gr.Image(type="pil", label="Vehicle photo")
            submit_btn = gr.Button("Assess damage", variant="primary")
        with gr.Column(min_width=320):
            overlay_output = gr.Image(type="pil", label="Detected damage + parts")
            triage_output = gr.HTML()
    with gr.Accordion("Structured result (JSON)", open=False):
        json_output = gr.JSON(label=None, show_label=False)
    gr.HTML(
        '<div class="aa-footer">Severity grades and triage thresholds are '
        "heuristic, not adjuster-validated &mdash; route every decision "
        "through human review until independently confirmed.</div>"
    )

    submit_btn.click(
        fn=assess, inputs=[image_input], outputs=[overlay_output, json_output, triage_output]
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
