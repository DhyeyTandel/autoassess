"""Framework-free assessment service: model loading, the pipeline call, and the
JSON response contract served by `autoassess.web.api`.

Env vars, defaults and sensitivity cutoffs are documented per function below.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError

from autoassess.eval.coco_eval import load_class_names
from autoassess.infer.associate import _load_yolo
from autoassess.infer.pipeline import (
    AUTO_APPROVE,
    HUMAN_REVIEW,
    TOTAL_LOSS_REVIEW,
    run_pipeline_with_masks,
)
from autoassess.infer.visualize import draw_overlay
from autoassess.utils.logging import get_logger

log = get_logger(__name__)

pillow_heif.register_heif_opener()  # lets PIL.Image.open decode iPhone .heic/.heif uploads

# (primary_weights, primary_config, secondary_weights, secondary_config); the
# secondary pair is None for single-model sources. Weights and dataset config
# are keyed off one env var so they can never be mismatched.
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

SENSITIVITY_STANDARD = "standard"
SENSITIVITY_HIGH_RECALL = "high_recall"
SENSITIVITIES = (SENSITIVITY_HIGH_RECALL, SENSITIVITY_STANDARD)
STANDARD_SECONDARY_CONF = 0.15

PARTS_DATASET_CONFIG = Path("configs/carparts.yaml")
SEVERITY_CONFIG = Path("configs/severity.yaml")
DEFAULT_PARTS_WEIGHTS = "runs/parts_seg_v1/weights/best.pt"

TRIAGE_LABELS = {
    AUTO_APPROVE: "Auto-approve",
    HUMAN_REVIEW: "Human review",
    TOTAL_LOSS_REVIEW: "Total-loss review",
}
_SEVERITY_RANK = {"severe": 0, "moderate": 1, "minor": 2}
OVERLAY_JPEG_QUALITY = 85


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
    """Map a sensitivity id to the secondary-model confidence cutoff (pure).

    "standard" is fixed at 0.15; "high_recall" uses the resolved
    (env-overridable) `settings.secondary_conf`. Raises ValueError otherwise.
    """
    if mode == SENSITIVITY_STANDARD:
        return STANDARD_SECONDARY_CONF
    if mode == SENSITIVITY_HIGH_RECALL:
        return settings.secondary_conf
    raise ValueError(
        f"Unknown sensitivity {mode!r}; expected {SENSITIVITY_HIGH_RECALL!r} "
        f"or {SENSITIVITY_STANDARD!r}."
    )


def load_image(data: bytes) -> Image.Image:
    """Decode upload bytes (JPEG/PNG/HEIC/...) to an RGB, EXIF-upright image.

    Raises ValueError if the bytes are not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Not a valid image file.") from exc


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


@dataclass
class AssessResult:
    """Raw pipeline result dict, the drawn overlay, and the secondary cutoff used."""

    result: dict[str, Any]
    overlay: Image.Image
    secondary_conf: float | None


class Assessor:
    """Loads the damage (one or two) and parts YOLO models once and runs the pipeline."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        device: str | None = None,
        temp_dir: Path | None = None,
        check_weights: bool = True,
    ) -> None:
        self._env: Mapping[str, str] = os.environ if env is None else env
        self.settings = resolve_damage_models(self._env)
        self.parts_weights = Path(self._env.get("PARTS_WEIGHTS", DEFAULT_PARTS_WEIGHTS))
        self.device = device if device is not None else self._env.get("AUTOASSESS_DEVICE", "cpu")
        self._temp_dir = temp_dir
        self._check_weights = check_weights
        self._loaded = False

    @property
    def has_secondary(self) -> bool:
        return self.settings.secondary_weights is not None

    def _weights(self) -> list[Path]:
        paths = [self.settings.primary_weights, self.parts_weights]
        if self.settings.secondary_weights is not None:
            paths.append(self.settings.secondary_weights)
        return paths

    def load(self) -> None:
        """Load all models into memory (idempotent). Raises FileNotFoundError."""
        if self._loaded:
            return
        if self._check_weights:
            missing = [p for p in self._weights() if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Model weights not found: {', '.join(str(p) for p in missing)}. "
                    "Set DAMAGE_WEIGHTS / SECONDARY_DAMAGE_WEIGHTS / PARTS_WEIGHTS, "
                    "or train the models first."
                )
            for p in self._weights():
                _load_yolo(str(p))
        self._loaded = True

    def _run_pipeline(self, **kwargs: Any) -> tuple[dict[str, Any], list[Any], list[Any]]:  # noqa: ANN401
        return run_pipeline_with_masks(**kwargs)

    def assess(self, image: Image.Image, sensitivity: str) -> AssessResult:
        self.load()
        s = self.settings
        secondary_conf = secondary_conf_for(sensitivity, s) if self.has_secondary else None
        kwargs: dict[str, Any] = {
            "damage_weights": s.primary_weights,
            "parts_weights": self.parts_weights,
            "damage_dataset_config": s.primary_config,
            "parts_dataset_config": PARTS_DATASET_CONFIG,
            "severity_config_path": SEVERITY_CONFIG,
            "triage_config_path": SEVERITY_CONFIG,
            "device": self.device,
            "secondary_damage_weights": s.secondary_weights,
            "secondary_damage_dataset_config": s.secondary_config,
            "merge_config_path": SEVERITY_CONFIG if self.has_secondary else None,
        }
        if secondary_conf is not None:
            kwargs["secondary_damage_conf"] = secondary_conf

        upload_path = save_upload_lossless(image, self._temp_dir)
        try:
            result, damage_instances, part_instances = self._run_pipeline(
                source=upload_path, **kwargs
            )
        finally:
            upload_path.unlink(missing_ok=True)

        records = result["instances"]
        overlay = draw_overlay(
            image,
            damage_instances,
            part_instances,
            [r["severity"] for r in records],
            [r["part"] for r in records],
        )
        return AssessResult(result=result, overlay=overlay, secondary_conf=secondary_conf)


def _overlay_data_url(overlay: Image.Image) -> str:
    buf = io.BytesIO()
    overlay.convert("RGB").save(buf, format="JPEG", quality=OVERLAY_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_response(
    result_dict: dict[str, Any],
    overlay_image: Image.Image,
    sensitivity: str,
    secondary_conf: float | None,
    elapsed_ms: float,
    panel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the POST /api/assess JSON body from a pipeline result (pure)."""
    if panel_names is None:
        panel_names = load_class_names(PARTS_DATASET_CONFIG)
    triage = result_dict["triage"]
    records = sorted(
        result_dict["instances"],
        key=lambda r: (_SEVERITY_RANK.get(r["severity"], len(_SEVERITY_RANK)), -r["confidence"]),
    )
    instances = [
        {
            "damage": r["damage_type"],
            "panel": None if r["part"] == "unassigned" else r["part"],
            "severity": r["severity"],
            "confidence": round(float(r["confidence"]), 4),
            "mask_area_px": int(round(r["mask_area_px"])),
            "source_model": r.get("source_model"),
        }
        for r in records
    ]
    panels = dict.fromkeys(panel_names, 0)
    for inst in instances:
        if inst["panel"] in panels:
            panels[inst["panel"]] += 1
    return {
        "triage": {
            "decision": triage["decision"],
            "label": TRIAGE_LABELS.get(triage["decision"], triage["decision"]),
            "n_instances": int(triage["n_instances"]),
            "n_severe": int(triage["n_severe"]),
            "n_moderate": int(triage["n_moderate"]),
            "n_minor": int(triage["n_minor"]),
            "total_damaged_area_px": int(round(triage["total_damaged_area_px"])),
        },
        "instances": instances,
        "panels": panels,
        "unassigned": sum(1 for i in instances if i["panel"] is None),
        "overlay": _overlay_data_url(overlay_image),
        "image": {
            "width": int(result_dict["image_width"]),
            "height": int(result_dict["image_height"]),
        },
        "sensitivity": sensitivity,
        "secondary_conf": secondary_conf,
        "elapsed_ms": round(float(elapsed_ms), 1),
    }
