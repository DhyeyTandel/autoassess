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
from autoassess.infer.visualize import draw_overlay

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

# hex is the light-mode colour; dark_hex is a lighter variant that stays legible
# on the dark card. Status colours only, not accents; the human-review amber is
# kept at hue > 45 deg so it never reads as orange.
TRIAGE_META = {
    AUTO_APPROVE: {"label": "Auto-approve", "hex": "#1A7F37", "dark_hex": "#3FB950"},
    HUMAN_REVIEW: {"label": "Flag for human review", "hex": "#8F7000", "dark_hex": "#E0C341"},
    TOTAL_LOSS_REVIEW: {
        "label": "Total-loss candidate",
        "hex": "#CF222E",
        "dark_hex": "#FF7B72",
    },
}

_SEVERITY_ORDER = {"severe": 0, "moderate": 1, "minor": 2}
_UNASSIGNED = "unassigned"


def render_triage_card(triage: dict[str, Any]) -> str:
    """Build the triage verdict as a plain bordered panel: status dot and label,
    then counts and total damaged area in secondary text, then the heuristic
    disclaimer in muted text."""
    decision = triage["decision"]
    meta = TRIAGE_META.get(
        decision, {"label": decision, "hex": "#71717A", "dark_hex": "#A1A1AA"}
    )
    parts = [
        f"{triage[key]} {name}"
        for key, name in (
            ("n_severe", "severe"),
            ("n_moderate", "moderate"),
            ("n_minor", "minor"),
        )
        if triage[key] > 0
    ]
    n = triage["n_instances"]
    if parts:
        summary = (
            f"{n} instance{'s' if n != 1 else ''} ({', '.join(parts)}) "
            f"&middot; {triage['total_damaged_area_px']:,.0f} px total damaged"
        )
    else:
        summary = "Nothing detected &mdash; not the same as confirmed damage-free, verify manually"

    return f"""
<div class="aa-verdict" style="--verdict:{meta['hex']};--verdict-dark:{meta['dark_hex']}">
  <div class="aa-verdict-label"><span class="aa-dot"></span>{meta['label']}</div>
  <div class="aa-verdict-meta">{summary}</div>
  <div class="aa-verdict-note">Severity and triage are heuristic, not adjuster-validated.
  Route decisions through human review.</div>
</div>
""".strip()


def instances_table(result: dict[str, Any]) -> list[list[str]]:
    """Rows of [Damage, Panel, Severity, Confidence] for the results table (pure).

    Sorted severe, moderate, minor, then confidence descending. Unassigned
    panels show as an em dash; confidence is formatted to two decimals.
    """
    ordered = sorted(
        result["instances"],
        key=lambda r: (_SEVERITY_ORDER.get(r["severity"], len(_SEVERITY_ORDER)), -r["confidence"]),
    )
    return [
        [
            str(r["damage_type"]),
            "—" if r["part"] == _UNASSIGNED else str(r["part"]),
            str(r["severity"]),
            f"{r['confidence']:.2f}",
        ]
        for r in ordered
    ]


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


STATUS_TEXT = "Analysing photo, about 7 seconds."


def show_status() -> dict[str, Any]:
    """Reveal the plain status line; chained before `assess` so it shows while it runs."""
    return gr.update(value=STATUS_TEXT, visible=True)


def hide_status() -> dict[str, Any]:
    """Hide the status line once `assess` has finished (also after an error)."""
    return gr.update(value="", visible=False)


def assess(
    image: Image.Image, sensitivity: str
) -> tuple[Image.Image, str, list[list[str]], dict[str, Any], dict[str, Any]]:
    """Run the pipeline; returns overlay, triage HTML, instance rows, raw result, column update."""
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

    table_rows = instances_table(result)
    return (
        overlay_image,
        render_triage_card(result["triage"]),
        table_rows,
        result,
        gr.update(visible=True),
    )


_SYSTEM_FONT = (
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
)
_MONO_FONT = ("ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace")

# Plain family names, so no font is fetched from a third party.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.zinc,
    text_size=gr.themes.sizes.text_md,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_sm,
    font=_SYSTEM_FONT,
    font_mono=_MONO_FONT,
).set(
    button_primary_background_fill="#0051C3",
    button_primary_background_fill_hover="#003D99",
    button_primary_background_fill_dark="#4C9AFF",
    button_primary_background_fill_hover_dark="#79B4FF",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#0E0E10",
    button_primary_border_color="#0051C3",
    button_primary_border_color_dark="#4C9AFF",
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
:root {
  --page: #F7F7F8; --card: #FFFFFF; --line: #E4E4E7;
  --text: #18181B; --text-2: #52525B; --muted: #71717A;
  --accent: #0051C3; --accent-hover: #003D99;
}
.dark {
  --page: #0E0E10; --card: #18181B; --line: #2E2E33;
  --text: #F4F4F5; --text-2: #A1A1AA; --muted: #A1A1AA;
  --accent: #4C9AFF; --accent-hover: #79B4FF;
}

/* Map Gradio's variables onto the tokens so built-in components follow
   light/dark without per-component rules. */
body, .gradio-container {
  background: var(--page) !important;
  color: var(--text-2);
  font-size: 15px;
  line-height: 1.5;
  --body-background-fill: var(--page);
  --background-fill-primary: var(--card);
  --background-fill-secondary: var(--page);
  --block-background-fill: var(--card);
  --block-border-color: var(--line);
  --block-border-width: 1px;
  --block-radius: 8px;
  --block-shadow: none;
  --block-label-text-color: var(--text-2);
  --block-label-background-fill: transparent;
  --block-title-text-color: var(--text);
  --body-text-color: var(--text-2);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: var(--line);
  --input-background-fill: var(--card);
  --input-border-color: var(--line);
  --input-radius: 6px;
  --input-shadow: none;
  --input-shadow-focus: none;
  --button-large-radius: 6px;
  --button-medium-radius: 6px;
  --button-small-radius: 6px;
  --button-shadow: none;
  --button-primary-background-fill: var(--accent);
  --button-primary-background-fill-hover: var(--accent-hover);
  --button-primary-border-color: var(--accent);
  --panel-background-fill: var(--card);
  --panel-border-color: var(--line);
  --table-border-color: var(--line);
  --table-even-background-fill: var(--card);
  --table-odd-background-fill: var(--card);
  --table-text-color: var(--text);
  --table-row-focus: var(--page);
  --shadow-drop: none;
  --shadow-drop-lg: none;
  --layout-gap: 16px;
}
.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 24px 16px 40px !important;
  box-sizing: border-box;
}
.gradio-container, .gradio-container * { min-width: 0; }
html, body { overflow-x: hidden; }
.gradio-container .main, .aa-shell, .aa-card, .aa-results, .aa-verdict {
  max-width: 100%;
  box-sizing: border-box;
}
.aa-results { width: 100%; }
.aa-card, .aa-results, .aa-verdict, .aa-header { overflow-wrap: anywhere; }

/* Header: a plain top bar */
.aa-header { margin: 0 0 8px; }
.aa-bar { display: flex; align-items: center; gap: 10px; }
.aa-name { font-size: 18px; font-weight: 600; color: var(--text); letter-spacing: -0.01em; }
.aa-tag {
  font-size: 12px; line-height: 1; color: var(--muted);
  border: 1px solid var(--line); border-radius: 4px; padding: 3px 6px;
}
.aa-lede { margin: 8px 0 0; font-size: 14px; color: var(--text-2); max-width: 70ch; }

/* Cards */
.aa-card, .aa-results {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px !important;
  gap: 16px;
}
.aa-card .block, .aa-results .block { box-shadow: none; }
.aa-card .block.padded { background: transparent; }
.aa-card .block:has([data-testid="image"]), .aa-results .block:has([data-testid="image"]) {
  border-radius: 6px;
  overflow: hidden;
}
label > span, .gradio-container label span[data-testid="block-info"] {
  font-weight: 500;
  color: var(--text);
}

/* Image label badges: solid chip so they stay readable over any photo */
.gradio-container [data-testid="block-label"] {
  background: var(--card) !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  font-weight: 500;
  font-size: 12px;
  top: 10px !important;
  left: 10px !important;
  z-index: 5;
}
.gradio-container [data-testid="block-label"] * { color: var(--text) !important; }

/* Sensitivity block: flat, hairline-separated, no card fill */
.gradio-container .aa-sens {
  background: transparent !important;
  border: none !important;
  border-top: 1px solid var(--line) !important;
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

/* Sensitivity radio as a two-segment control */
.aa-sens .wrap { display: inline-flex; flex-wrap: nowrap; gap: 0; }
.aa-sens label {
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
  background: var(--page) !important;
  padding: 6px 14px !important;
  font-weight: 500;
  font-size: 14px;
  color: var(--text-2);
  cursor: pointer;
  margin: 0;
}
.aa-sens label + label { margin-left: -1px; }
.aa-sens label:first-child { border-radius: 6px 0 0 6px !important; }
.aa-sens label:last-child { border-radius: 0 6px 6px 0 !important; }
.aa-sens label input[type="radio"] {
  position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}
.aa-sens label.selected, .aa-sens label:has(input:checked) {
  background: var(--card) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  position: relative;
  z-index: 1;
}
.aa-sens label:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Assess button */
.aa-assess, .aa-assess button { width: 100%; }
button.aa-assess {
  background: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 6px !important;
  font-weight: 500;
  padding: 10px 16px;
}
button.aa-assess:hover { background: var(--accent-hover) !important; }

/* Status line shown while the pipeline runs */
.aa-status { font-size: 14px; color: var(--text-2); }
.aa-status p { margin: 0; }

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

/* Triage panel */
.aa-verdict {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}
.aa-verdict-label {
  display: flex; align-items: center; gap: 8px;
  font-size: 16px; font-weight: 600; color: var(--text);
}
.aa-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--verdict); flex: none; }
.dark .aa-dot { background: var(--verdict-dark); }
.aa-verdict-meta { margin-top: 6px; font-size: 14px; color: var(--text-2); }
.aa-verdict-note {
  margin-top: 10px; padding-top: 10px;
  border-top: 1px solid var(--line);
  font-size: 12.5px; line-height: 1.5; color: var(--muted);
}

/* Instances table */
.aa-h { margin: 4px 0 0; font-size: 14px; font-weight: 600; color: var(--text); }
.aa-table {
  max-width: 100%;
  overflow-x: auto;
}
.aa-table, .aa-table * {
  font-family: var(--font);
  font-variant-numeric: tabular-nums;
}
.aa-table table { font-size: 14px; }
.aa-table th {
  font-weight: 500;
  text-transform: none;
  color: var(--text-2);
}
.aa-results .json-holder, .aa-results pre { overflow-x: auto; max-width: 100%; }

.aa-notice {
  margin: 16px 0 0;
  font-size: 12.5px; line-height: 1.5; color: var(--muted);
}
"""

HEADER_HTML = (
    '<div class="aa-header">'
    '<div class="aa-bar"><span class="aa-name">AutoAssess</span>'
    '<span class="aa-tag">Research demo</span></div>'
    '<p class="aa-lede">Upload a photo of a damaged vehicle. AutoAssess segments the '
    "damage, assigns each instance to a panel, grades severity and suggests how to "
    "route the claim.</p>"
    "</div>"
)

# (sweep every, delete files older than) seconds: uploaded photos can show licence
# plates and must not accumulate on a public server.
CACHE_SWEEP = (600, 600)

# Gradio 6.22 ignores GRADIO_MAX_FILE_SIZE (page config shows max_file_size null),
# so read it here and pass it to launch() explicitly.
MAX_UPLOAD = os.environ.get("GRADIO_MAX_FILE_SIZE", "15mb")

NOTICE_HTML = (
    '<p class="aa-notice">Research demo, non-commercial. Models trained on CarDD, '
    "VehiDE and Carparts-Seg. Uploaded photos are deleted from the server within "
    "about 20 minutes.</p>"
)

SCROLL_TO_RESULTS_JS = """() => {
  const el = document.querySelector('.aa-results');
  if (el) {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block: 'start'});
  }
}"""

TABLE_HEADERS = ["Damage", "Panel", "Severity", "Confidence"]

with gr.Blocks(title="AutoAssess — Damage Triage", delete_cache=CACHE_SWEEP) as demo:
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
            submit_btn = gr.Button("Assess damage", variant="primary", elem_classes="aa-assess")
            status_output = gr.Markdown(visible=False, elem_classes="aa-status")
        with gr.Column(min_width=0, visible=False, elem_classes="aa-results") as results_column:
            overlay_output = gr.Image(type="pil", label="Detected damage and parts")
            triage_output = gr.HTML()
            gr.HTML('<h2 class="aa-h">Instances</h2>')
            table_output = gr.Dataframe(
                headers=TABLE_HEADERS,
                datatype="str",
                interactive=False,
                show_row_numbers=False,
                show_label=False,
                wrap=True,
                buttons=[],
                show_search="none",
                elem_classes="aa-table",
            )
            with gr.Accordion("Raw result", open=False):
                json_output = gr.JSON(label=None, show_label=False)
    gr.HTML(NOTICE_HTML)

    # .upload (not .change): the returned value does not re-fire .upload, so no loop.
    image_input.upload(normalize_upload, inputs=image_input, outputs=image_input)
    # The status line is a separate, always-visible component updated before and after
    # `assess`: Gradio's built-in progress overlay draws on the output components, and
    # those sit in the results column, which is hidden until the first result arrives.
    # show_status is queue=False so it paints at once instead of waiting behind the job.
    assess_event = submit_btn.click(
        fn=show_status,
        inputs=None,
        outputs=status_output,
        queue=False,
        show_progress="hidden",
    ).then(
        fn=assess,
        inputs=[image_input, sensitivity_input],
        outputs=[overlay_output, triage_output, table_output, json_output, results_column],
        show_progress="hidden",
    )
    # .then (not .success) so the status line also clears when assess raises gr.Error.
    assess_event.then(fn=hide_status, inputs=None, outputs=status_output, queue=False)
    assess_event.success(
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
    demo.launch(
        theme=THEME,
        css=CSS,
        head=HEAD_STYLE,
        footer_links=["settings"],
        max_file_size=MAX_UPLOAD,
    )
