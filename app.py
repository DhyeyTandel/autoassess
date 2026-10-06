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
    # section of configs/severity.yaml) are merged in. The UI has a "Sensitivity"
    # radio (shown only when a secondary model is loaded) that picks the VehiDE
    # confidence cutoff per request:
    #   Standard    0.15 (precision-leaning; missing_part P 0.59 / R 0.69 on the
    #               VehiDE test split)
    #   High recall SECONDARY_DAMAGE_CONF, default 0.07 (the UI default). At 0.15
    #               the demo misses a hanging/detached front bumper on CarDD test
    #               image 000042, which VehiDE detects as missing_part at 0.07.
    #               The cost is precision (missing_part P ~0.45, torn P ~0.20);
    #               roughly half the extra VehiDE detections are false alarms.
    # SECONDARY_DAMAGE_CONF therefore sets the High recall value only.
    # `autoassess-pipeline` keeps its own 0.15 default. The chosen mode and
    # cutoff are echoed under "settings" in the result JSON.
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
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pillow_heif
from PIL import Image, ImageOps

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
_DEFAULT_SECONDARY_CONF = 0.07

SENSITIVITY_STANDARD = "Standard"
SENSITIVITY_HIGH_RECALL = "High recall"
STANDARD_SECONDARY_CONF = 0.15


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


def secondary_conf_for(mode: str, settings: DamageModelSettings) -> float:
    """Map a sensitivity mode to the secondary-model confidence cutoff (pure).

    Standard is fixed at STANDARD_SECONDARY_CONF; High recall uses the resolved
    (env-overridable) `settings.secondary_conf`. Raises ValueError otherwise.
    """
    if mode == SENSITIVITY_STANDARD:
        return STANDARD_SECONDARY_CONF
    if mode == SENSITIVITY_HIGH_RECALL:
        return settings.secondary_conf
    raise ValueError(
        f"Unknown sensitivity {mode!r}; expected {SENSITIVITY_STANDARD!r} "
        f"or {SENSITIVITY_HIGH_RECALL!r}."
    )


_damage_models = resolve_damage_models(os.environ)
DAMAGE_WEIGHTS = _damage_models.primary_weights
DAMAGE_DATASET_CONFIG = _damage_models.primary_config
SECONDARY_DAMAGE_WEIGHTS = _damage_models.secondary_weights
SECONDARY_DAMAGE_DATASET_CONFIG = _damage_models.secondary_config
PARTS_WEIGHTS = Path(os.environ.get("PARTS_WEIGHTS", "runs/parts_seg_v1/weights/best.pt"))
PARTS_DATASET_CONFIG = Path("configs/carparts.yaml")
SEVERITY_CONFIG = Path("configs/severity.yaml")
DEVICE = os.environ.get("AUTOASSESS_DEVICE", "cpu")

TRIAGE_META = {
    AUTO_APPROVE: {"label": "Auto-approve", "hex": "#1E8A5A", "glyph": "check"},
    HUMAN_REVIEW: {"label": "Flag for human review", "hex": "#B7791F", "glyph": "flag"},
    TOTAL_LOSS_REVIEW: {"label": "Total-loss candidate", "hex": "#D0342C", "glyph": "alert"},
}

def render_triage_card(triage: dict[str, Any]) -> str:
    """Build the triage verdict as a square card with a left stripe in the verdict
    colour. This is the one thing a reviewer scans for first, so it gets the
    visual weight; the heuristic disclaimer lives inside it."""
    decision = triage["decision"]
    meta = TRIAGE_META.get(decision, {"label": decision, "hex": "#7C7367", "glyph": "flag"})
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
    n = triage["n_instances"]

    return f"""
<div class="aa-verdict" style="--verdict:{meta['hex']}">
  <div class="aa-eyebrow">VERDICT</div>
  <div class="aa-verdict-label">{meta['label']}</div>
  <div class="aa-verdict-chips">{chips}</div>
  <div class="aa-verdict-meta">
    {n} instance{'s' if n != 1 else ''} &middot;
    {triage['total_damaged_area_px']:,.0f} px total damaged
  </div>
  <div class="aa-verdict-note">▸ Severity and triage are heuristic, not adjuster-validated.
  Route decisions through human review.</div>
</div>
""".strip()


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def save_upload_lossless(image: Image.Image, directory: Path | None = None) -> Path:
    """Write `image` as RGB PNG to a unique temp file and return its path.

    Lossless on purpose: a JPEG re-encode shifts low-confidence scores enough to
    drop borderline detections. The caller owns (and must delete) the file.
    """
    fd, name = tempfile.mkstemp(suffix=".png", dir=directory)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            image.convert("RGB").save(fh, format="PNG")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def normalize_upload(image: Image.Image | None) -> Image.Image | None:
    """Return a new RGB, EXIF-upright copy of `image` (None passes through).

    Used on upload so the input preview shows a browser-renderable PNG instead
    of the raw .heic gradio would otherwise serve back. Gradio's own
    preprocess (image_utils.preprocess_image) already applies
    ImageOps.exif_transpose when Orientation != 1 and the transposed image
    carries no Orientation tag, so by the time a value reaches Python it is
    upright; transposing again here is a no-op for those images and only
    matters for callers passing an untransposed image.
    """
    if image is None:
        return None
    return ImageOps.exif_transpose(image).convert("RGB")


def assess(
    image: Image.Image, sensitivity: str
) -> tuple[Image.Image, dict[str, Any], str, dict[str, Any]]:
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

    has_secondary = SECONDARY_DAMAGE_WEIGHTS is not None
    secondary_conf = secondary_conf_for(sensitivity, _damage_models) if has_secondary else None

    upload_path = save_upload_lossless(image)
    try:
        result, damage_instances, part_instances = run_pipeline_with_masks(
            source=upload_path,
            damage_weights=DAMAGE_WEIGHTS,
            parts_weights=PARTS_WEIGHTS,
            damage_dataset_config=DAMAGE_DATASET_CONFIG,
            parts_dataset_config=PARTS_DATASET_CONFIG,
            severity_config_path=SEVERITY_CONFIG,
            triage_config_path=SEVERITY_CONFIG,
            device=DEVICE,
            secondary_damage_weights=SECONDARY_DAMAGE_WEIGHTS,
            secondary_damage_dataset_config=SECONDARY_DAMAGE_DATASET_CONFIG,
            secondary_damage_conf=secondary_conf,
            merge_config_path=SEVERITY_CONFIG if has_secondary else None,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    severities = [rec["severity"] for rec in result["instances"]]
    parts_assigned = [rec["part"] for rec in result["instances"]]
    overlay_image = draw_overlay(
        image, damage_instances, part_instances, severities, parts_assigned
    )

    # Record which setting produced this output (None when there is no secondary model).
    result["settings"] = {
        "sensitivity": sensitivity if has_secondary else None,
        "secondary_conf": secondary_conf,
    }

    return overlay_image, result, render_triage_card(result["triage"]), gr.update(visible=True)


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    text_size=gr.themes.sizes.text_md,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_sm,
    font=(
        gr.themes.GoogleFont("Sofia Sans"),
        "ui-sans-serif",
        "Helvetica Neue",
        "Arial",
        "sans-serif",
    ),
    font_mono=(
        gr.themes.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "SFMono-Regular",
        "Menlo",
        "monospace",
    ),
).set(
    button_primary_background_fill="#EE5308",
    button_primary_background_fill_hover="#C24303",
    button_primary_background_fill_dark="#EE5308",
    button_primary_background_fill_hover_dark="#C24303",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="#EE5308",
    button_primary_border_color_dark="#EE5308",
)

# These rules live in a head <style> instead of CSS because Gradio prefixes every
# selector in `css=` with `.gradio-container.gradio-container-<ver> .contain`, and both
# `.main` (32px side padding on phones) and `<footer>` (stray "·" dividers) are outside
# `.contain`, so scoped versions can never match. Settings is the only footer link
# (footer_links=["settings"]), so hiding every footer divider is intended.
HEAD_STYLE = """<style>
@media (max-width: 640px) {
  .gradio-container .main { padding-left: 0 !important; padding-right: 0 !important; }
}
footer .divider { display: none !important; }
</style>"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;1,400&family=Sofia+Sans:wght@1..1000&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --paper: #F4F0E9; --paper-lift: #FCFAF6; --card: #FFFFFF;
  --ink: #17140F; --ink-soft: #2B2620; --body-color: #4B443A;
  --muted: #7C7367; --muted-soft: #A89E90;
  --faint: #E6DFD2; --faint-soft: #EFEAE0;
  --accent: #EE5308; --accent-deep: #C24303; --accent-soft: #FBE6D9;
  --success: #1E8A5A; --error: #D0342C;
  --ease-spring: cubic-bezier(.34,1.56,.64,1);
  --ease-swift: cubic-bezier(.2,.7,.2,1);
  --d1: .18s; --d2: .32s; --d3: .6s;
  --font-display: 'Newsreader', Georgia, serif;
  --font-body: 'Sofia Sans', ui-sans-serif, 'Helvetica Neue', Arial, sans-serif;
  --font-mono-aa: 'JetBrains Mono', ui-monospace, Menlo, monospace;
}
.dark {
  --paper: #17140F; --paper-lift: #1E1912; --card: #201B13;
  --ink: #F4F0E9; --ink-soft: #EFEAE0; --body-color: #C9C0B2;
  --muted: #A89E90; --muted-soft: #6E6455;
  --faint: #332D23; --faint-soft: #2A2519;
  --accent-soft: #3A2313; --accent-deep: #FF8A4D;
}

/* Map Gradio's variables onto the tokens so every built-in component follows
   light/dark without per-component rules. */
body, .gradio-container {
  background: var(--paper) !important;
  color: var(--body-color);
  font-family: var(--font-body);
  font-size: 16.5px;
  line-height: 1.6;
  font-weight: 450;
  --body-background-fill: var(--paper);
  --background-fill-primary: var(--card);
  --background-fill-secondary: var(--paper-lift);
  --block-background-fill: var(--card);
  --block-border-color: var(--faint);
  --block-border-width: 1px;
  --block-radius: 4px;
  --block-shadow: none;
  --block-label-text-color: var(--muted);
  --block-label-background-fill: transparent;
  --block-title-text-color: var(--ink-soft);
  --body-text-color: var(--body-color);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: var(--faint);
  --input-background-fill: var(--paper-lift);
  --input-border-color: var(--faint);
  --input-radius: 8px;
  --input-shadow: none;
  --input-shadow-focus: none;
  --button-large-radius: 999px;
  --button-medium-radius: 999px;
  --button-small-radius: 999px;
  --button-shadow: none;
  --button-primary-background-fill: var(--accent);
  --button-primary-background-fill-hover: var(--accent-deep);
  --button-primary-text-color: #fff;
  --button-primary-border-color: var(--accent);
  --panel-background-fill: var(--card);
  --panel-border-color: var(--faint);
  --shadow-drop: none;
  --shadow-drop-lg: none;
  --layout-gap: 16px;
}
.gradio-container {
  max-width: 1160px !important;
  margin: 0 auto !important;
  padding: 24px 16px 48px !important;
  box-sizing: border-box;
}
.gradio-container, .gradio-container * { min-width: 0; }
html, body { overflow-x: hidden; }

/* Header */
.aa-header { text-align: left; margin: 8px 0 8px; }
.aa-eyebrow {
  font-family: var(--font-mono-aa);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 0 0 12px 0;
}
.aa-eyebrow::before {
  content: "";
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent);
  flex: none;
}
.aa-title {
  font-family: var(--font-display);
  font-weight: 400;
  letter-spacing: -0.02em;
  font-size: clamp(2.1rem, 6vw, 3.6rem);
  line-height: 1.05;
  color: var(--ink);
  margin: 0 0 14px 0;
  text-wrap: balance;
}
.aa-title em { font-style: italic; font-weight: 400; color: var(--accent); }
.aa-subtitle {
  font-size: 16.5px; line-height: 1.6; color: var(--muted);
  max-width: 62ch; margin: 0 0 8px 0;
}

/* Cards and frames */
.aa-card, .aa-results {
  background: var(--card);
  border: 1px solid var(--faint);
  border-radius: 4px;
  padding: 16px !important;
  box-shadow: none;
  gap: 16px;
}
.aa-card .block, .aa-results .block { box-shadow: none; }
.aa-card .block.padded { background: transparent; }
.aa-card .image-container, .aa-card [data-testid="image"],
.aa-results [data-testid="image"], .aa-results .image-container,
.aa-card .image-frame, .aa-results .image-frame {
  border-radius: 24px;
  overflow: hidden;
}
.aa-card .block:has([data-testid="image"]), .aa-results .block:has([data-testid="image"]) {
  border-radius: 24px;
  overflow: hidden;
}
input, textarea, select { border-radius: 8px !important; }
label > span, .gradio-container label span[data-testid="block-info"] {
  font-weight: 580;
}

/* Image label badges: solid chip so they stay readable over any photo */
.gradio-container [data-testid="block-label"] {
  background: var(--card) !important;
  border: 1px solid var(--faint) !important;
  border-radius: 8px !important;
  color: var(--body-color) !important;
  font-weight: 560;
  font-size: var(--block-label-text-size, 0.85em);
  top: 10px !important;
  left: 10px !important;
  z-index: 5;
}
.gradio-container [data-testid="block-label"] * { color: var(--body-color) !important; }

/* Sensitivity block: flat, hairline-separated, no card fill */
.gradio-container .aa-sens {
  background: transparent !important;
  border: none !important;
  border-top: 1px solid var(--faint) !important;
  box-shadow: none !important;
  padding-top: 14px;
  margin-top: 6px;
}

/* Gradio's own wrapper around the radio carries the solid fill; flatten only that one */
.gradio-container .form:has(> .aa-sens) {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Sensitivity radio as pill segments */
.aa-sens .wrap { display: flex; flex-wrap: wrap; gap: 8px; }
.aa-sens label {
  border-radius: 999px !important;
  border: 1px solid var(--faint) !important;
  background: var(--paper-lift) !important;
  padding: 6px 16px !important;
  font-weight: 580;
  color: var(--body-color);
  cursor: pointer;
  transition: background var(--d1) var(--ease-swift), border-color var(--d1) var(--ease-swift),
    color var(--d1) var(--ease-swift), transform var(--d1) var(--ease-spring);
}
.aa-sens label:hover { border-color: var(--muted-soft) !important; }
.aa-sens label:active { transform: scale(.97); }
.aa-sens label input[type="radio"] {
  position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}
.aa-sens label.selected, .aa-sens label:has(input:checked) {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: var(--paper) !important;
}
.aa-sens label:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Assess button */
.aa-assess, .aa-assess button {
  border-radius: 999px !important;
  font-weight: 600;
  width: 100%;
}
button.aa-assess {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  padding: 12px 24px;
  transition: transform var(--d2) var(--ease-spring), box-shadow var(--d2) var(--ease-swift),
    background var(--d1) var(--ease-swift);
}
button.aa-assess:hover {
  background: var(--accent-deep) !important;
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(238, 83, 8, .28);
}
button.aa-assess:active { transform: scale(.97); box-shadow: none; }

/* Layout: stack on mobile, side by side from 900px */
.aa-shell { flex-direction: column !important; gap: 16px; }
@media (min-width: 900px) {
  .aa-shell { flex-direction: row !important; align-items: flex-start; }
  .aa-shell > .aa-input-col { flex: 1 1 0 !important; }
  .aa-shell > .aa-results { flex: 1.35 1 0 !important; }
}

/* Phones: shrink stacked padding so the two images get most of the 375px width */
@media (max-width: 640px) {
  .aa-card, .aa-results { padding: 12px !important; gap: 12px; }
  .aa-card .block:has([data-testid="image"]),
  .aa-results .block:has([data-testid="image"]) { padding: 0 !important; }
}

/* Results reveal */
.aa-results { animation: aa-fade-up var(--d3) var(--ease-swift) both; }
@keyframes aa-fade-up {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Verdict card: square, 4px stripe in the verdict colour */
.aa-verdict {
  background: var(--card);
  border: 1px solid var(--faint);
  border-left: 4px solid var(--verdict);
  border-radius: 0;
  padding: 16px 18px;
}
.aa-verdict .aa-eyebrow { margin-bottom: 8px; }
.aa-verdict-label {
  font-family: var(--font-display);
  font-weight: 400;
  letter-spacing: -0.02em;
  font-size: 1.9rem;
  line-height: 1.15;
  color: var(--verdict);
}
.aa-verdict-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.aa-chip {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 14px; font-weight: 580;
  padding: 3px 12px 3px 9px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--chip) 16%, var(--card));
  color: var(--ink-soft);
}
.aa-chip-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--chip); flex: none; }
.aa-chip-muted {
  background: transparent; border: 1px dashed var(--faint);
  color: var(--muted); font-weight: 450;
}
.aa-verdict-meta {
  font-family: var(--font-mono-aa);
  font-size: 13px;
  color: var(--body-color);
  font-variant-numeric: tabular-nums;
}
.aa-verdict-note {
  margin-top: 12px; padding-top: 10px;
  border-top: 1px solid var(--faint-soft);
  font-size: 13.5px; line-height: 1.5; color: var(--muted);
}

/* Hide JSON mono text overflow on narrow screens */
.aa-results .json-holder, .aa-results pre { overflow-x: auto; max-width: 100%; }

@media (prefers-reduced-motion: reduce) {
  .aa-results { animation: none; }
  *, *::before, *::after { transition-duration: .01ms !important; }
  button.aa-assess:hover, button.aa-assess:active, .aa-sens label:active { transform: none; }
}
"""

HEADER_HTML = (
    '<div class="aa-header">'
    '<div class="aa-eyebrow">Claims triage &middot; computer vision</div>'
    '<h1 class="aa-title">Assess vehicle <em>damage</em> from one photo</h1>'
    '<p class="aa-subtitle">Locates the damage, names the affected panel, grades '
    "severity and routes the claim.</p>"
    "</div>"
)

SCROLL_TO_RESULTS_JS = """() => {
  const el = document.querySelector('.aa-results');
  if (el) {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block: 'start'});
  }
}"""

with gr.Blocks(title="AutoAssess — Damage Triage") as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row(equal_height=False, elem_classes="aa-shell"):
        with gr.Column(min_width=0, elem_classes="aa-card aa-input-col"):
            # format="png": the default (webp) is lossy; the re-served preview should
            # be lossless and browser-renderable (raw .heic is not).
            image_input = gr.Image(
                type="pil", label="Vehicle photo", format="png", sources=["upload"]
            )
            sensitivity_input = gr.Radio(
                choices=[SENSITIVITY_HIGH_RECALL, SENSITIVITY_STANDARD],
                value=SENSITIVITY_HIGH_RECALL,
                label="Sensitivity",
                info=(
                    "High recall flags more possible damage (including torn or missing "
                    "parts) but raises more false alarms; Standard is stricter."
                ),
                visible=SECONDARY_DAMAGE_WEIGHTS is not None,
                elem_classes="aa-sens",
            )
            submit_btn = gr.Button("Assess damage →", variant="primary", elem_classes="aa-assess")
        with gr.Column(min_width=0, visible=False, elem_classes="aa-results") as results_column:
            overlay_output = gr.Image(type="pil", label="Detected damage and parts")
            triage_output = gr.HTML()
            with gr.Accordion("Structured result (JSON)", open=False):
                json_output = gr.JSON(label=None, show_label=False)

    # .upload (not .change): the returned value does not re-fire .upload, so no loop.
    image_input.upload(normalize_upload, inputs=image_input, outputs=image_input)
    submit_btn.click(
        fn=assess,
        inputs=[image_input, sensitivity_input],
        outputs=[overlay_output, json_output, triage_output, results_column],
    ).success(
        # Gradio's scroll_to_output only scrolls when the output is fully off-screen; on
        # mobile the results column's top edge is barely visible, so scroll explicitly.
        fn=None,
        inputs=None,
        outputs=None,
        js=SCROLL_TO_RESULTS_JS,
    )

if __name__ == "__main__":
    # footer_links drops "Built with Gradio" and "Use via API" (whose hidden-on-mobile
    # separator left a stray leading dot) and keeps Settings.
    demo.launch(theme=THEME, css=CSS, head=HEAD_STYLE, footer_links=["settings"])
