"""Gradio demo — upload an image, see overlaid masks, structured JSON, and the triage decision.

Runs the same pipeline as the CLI (`autoassess-pipeline`, backed by
`autoassess.infer.pipeline.run_pipeline_with_masks`); this file only adds an
upload UI and a mask-overlay renderer around that one call. Weight paths
come from env vars (see below) since this is a thin demo shell, not a
config-driven training script.

Usage
-----
    python app.py                               # merged: CarDD + VehiDE (default)
    DAMAGE_MODEL_SOURCE=cardd python app.py     # CarDD damage model only
    DAMAGE_MODEL_SOURCE=vehide python app.py    # VehiDE damage model only

    # Merged mode: CarDD is primary; VehiDE's extra classes (see the `merge:`
    # section of configs/severity.yaml) are merged in. SECONDARY_DAMAGE_CONF
    # (default 0.15) is the VehiDE confidence cutoff. 0.15 was chosen from a
    # VehiDE test-split sweep: missing_part P0.59/R0.69 at 0.15 versus
    # P0.37/R0.75 at 0.05, i.e. a much cleaner precision for a small recall cost.
    SECONDARY_DAMAGE_CONF=0.25 python app.py

    # Or override weights/parts paths individually (DAMAGE_WEIGHTS overrides only
    # the primary model, SECONDARY_DAMAGE_WEIGHTS only the secondary):
    DAMAGE_WEIGHTS=runs/yolov8_seg_v1/weights/best.pt \\
    SECONDARY_DAMAGE_WEIGHTS=runs/vehide_seg_v1/weights/best.pt \\
    PARTS_WEIGHTS=runs/parts_seg_v1/weights/best.pt \\
    python app.py
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pillow_heif
from PIL import Image

from autoassess.infer.pipeline import (
    AUTO_APPROVE,
    HUMAN_REVIEW,
    TOTAL_LOSS_REVIEW,
    run_pipeline_with_masks,
)
from autoassess.infer.visualize import SEVERITY_COLORS, draw_overlay

pillow_heif.register_heif_opener()  # lets PIL.Image.open decode iPhone .heic/.heif uploads

# Two damage models exist — CarDD (6 classes, fully converged) and VehiDE (7
# classes incl. torn/punctured/missing_part, which CarDD has no equivalent
# for, but only trained to 48/50 epochs). Weights and dataset config must be
# switched together — a weights/config mismatch would silently mislabel every
# detection (wrong class count/names) rather than erroring, so each (weights,
# config) pair is keyed off one env var instead of two independent ones. The
# "merged" source runs CarDD as primary and VehiDE as secondary; each entry is
# (primary_weights, primary_config, secondary_weights, secondary_config), with
# the secondary pair None for single-model sources.
_DAMAGE_MODEL_SOURCES: dict[str, tuple[str, str, str | None, str | None]] = {
    "merged": (
        "runs/yolov8_seg_v1/weights/best.pt",
        "configs/cardd.yaml",
        "runs/vehide_seg_v1/weights/best.pt",
        "configs/vehide.yaml",
    ),
    "cardd": ("runs/yolov8_seg_v1/weights/best.pt", "configs/cardd.yaml", None, None),
    "vehide": ("runs/vehide_seg_v1/weights/best.pt", "configs/vehide.yaml", None, None),
}
_DEFAULT_DAMAGE_SOURCE = "merged"
_DEFAULT_SECONDARY_CONF = 0.15


@dataclass(frozen=True)
class DamageModelSettings:
    """Resolved damage-model choice; secondary fields are None for single-model sources."""

    source: str
    primary_weights: Path
    primary_config: Path
    secondary_weights: Path | None
    secondary_config: Path | None
    secondary_conf: float


def resolve_damage_models(env: Mapping[str, str]) -> DamageModelSettings:
    """Resolve damage-model settings from env-style vars (pure; no I/O).

    Raises ValueError for an unknown DAMAGE_MODEL_SOURCE.
    """
    source = env.get("DAMAGE_MODEL_SOURCE", _DEFAULT_DAMAGE_SOURCE)
    if source not in _DAMAGE_MODEL_SOURCES:
        raise ValueError(
            f"DAMAGE_MODEL_SOURCE={source!r} is not one of {sorted(_DAMAGE_MODEL_SOURCES)}."
        )
    p_weights, p_config, s_weights, s_config = _DAMAGE_MODEL_SOURCES[source]
    secondary_weights: Path | None = None
    secondary_config: Path | None = None
    if s_weights is not None and s_config is not None:
        secondary_weights = Path(env.get("SECONDARY_DAMAGE_WEIGHTS", s_weights))
        secondary_config = Path(s_config)
    return DamageModelSettings(
        source=source,
        primary_weights=Path(env.get("DAMAGE_WEIGHTS", p_weights)),
        primary_config=Path(p_config),
        secondary_weights=secondary_weights,
        secondary_config=secondary_config,
        secondary_conf=float(env.get("SECONDARY_DAMAGE_CONF", _DEFAULT_SECONDARY_CONF)),
    )


_damage_models = resolve_damage_models(os.environ)
DAMAGE_WEIGHTS = _damage_models.primary_weights
DAMAGE_DATASET_CONFIG = _damage_models.primary_config
SECONDARY_DAMAGE_WEIGHTS = _damage_models.secondary_weights
SECONDARY_DAMAGE_DATASET_CONFIG = _damage_models.secondary_config
SECONDARY_DAMAGE_CONF = _damage_models.secondary_conf
PARTS_WEIGHTS = Path(os.environ.get("PARTS_WEIGHTS", "runs/parts_seg_v1/weights/best.pt"))
PARTS_DATASET_CONFIG = Path("configs/carparts.yaml")
SEVERITY_CONFIG = Path("configs/severity.yaml")
DEVICE = os.environ.get("AUTOASSESS_DEVICE", "cpu")
UPLOAD_TMP_PATH = Path("/tmp/autoassess_upload.jpg")

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
    ) or (
        '<span class="aa-chip aa-chip-muted">Nothing detected &mdash; '
        "not the same as confirmed damage-free, verify manually</span>"
    )

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
    required = [DAMAGE_WEIGHTS, PARTS_WEIGHTS]
    if SECONDARY_DAMAGE_WEIGHTS is not None:
        required.append(SECONDARY_DAMAGE_WEIGHTS)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise gr.Error(
            f"Model weights not found: {', '.join(str(p) for p in missing)}. "
            "Set DAMAGE_WEIGHTS / SECONDARY_DAMAGE_WEIGHTS / PARTS_WEIGHTS env vars, "
            "or train the models first."
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
        secondary_damage_weights=SECONDARY_DAMAGE_WEIGHTS,
        secondary_damage_dataset_config=SECONDARY_DAMAGE_DATASET_CONFIG,
        secondary_damage_conf=SECONDARY_DAMAGE_CONF,
        merge_config_path=SEVERITY_CONFIG if SECONDARY_DAMAGE_WEIGHTS is not None else None,
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
