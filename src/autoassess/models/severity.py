"""Heuristic severity grading from detected damage instances.

Formula
-------
Given the damage instances detected in one image (class + pixel mask area
for each), the severity score is a **class-weighted area fraction**:

    score = sum(class_weight[c_i] * mask_area_i for i in instances) / vehicle_region_area

- `mask_area_i` is the pixel area of instance i's segmentation mask.
- `class_weight[c_i]` is a per-class severity prior from `configs/severity.yaml`
  (e.g. glass shatter and crack weighted higher than scratch): a given area of
  structural or safety-relevant damage counts for more than the same area of
  cosmetic damage.
- `vehicle_region_area` approximates "how much of the frame is vehicle," so
  the score reads as roughly "fraction of the vehicle that's damaged,"
  independent of how tightly the photo is cropped. CarDD has no vehicle
  bounding-box annotation, so this is **approximated** as the union of all
  detected damage boxes in the image, padded outward by
  `vehicle_region_padding` (see `configs/severity.yaml`) — not a real vehicle
  detection. This approximation is biased in both directions: a single small,
  off-center defect makes the padded union much smaller than the true
  vehicle, inflating the score; heavy damage spread across the frame makes
  the union approach the true vehicle extent, which is the case the padding
  constant is loosely tuned for. Treat the resulting score as a rough,
  monotonic severity ranking, not a calibrated "% of vehicle damaged."

The continuous score is then bucketed into minor / moderate / severe using
two thresholds from the same config file.

Circularity warning
--------------------
This heuristic is a hand-authored prior, not fit to any ground-truth severity
label — CarDD has no severity annotation at all. It is used in two ways in
this project:

1. As the only severity signal available for evaluation, since there is
   nothing else to compare against.
2. As **weak supervision** to train a learned classifier
   (`autoassess.train.train_severity`).

Using this heuristic for both purposes is circular: a learned model trained
on heuristic labels and then "evaluated" against those same heuristic labels
can only ever be measured on how well it imitates the heuristic's arithmetic,
not on whether the heuristic itself is a good measure of real-world claim
severity. See `autoassess.train.train_severity` and
`autoassess.eval.severity_eval` module docstrings for how this shows up in
practice and what fixing it would require (real adjuster-labeled severity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEVERITY_LABELS = ["minor", "moderate", "severe"]


@dataclass
class DamageInstance:
    """One detected (or ground-truth) damage instance, framework-agnostic.

    `class_name` must match a key in `configs/severity.yaml`'s `class_weights`
    (underscored form, e.g. "glass shatter" -> "glass_shatter" — see
    `normalise_class_name`). `mask_area` is in pixels. `bbox` is
    `(x0, y0, x1, y1)` in absolute pixel coordinates, used only to build the
    vehicle-region proxy, not the severity score itself.
    """

    class_name: str
    mask_area: float
    bbox: tuple[float, float, float, float]


@dataclass
class SeverityConfig:
    class_weights: dict[str, float]
    vehicle_region_padding: float
    minor_max: float
    moderate_max: float
    labels: list[str]

    @classmethod
    def load(cls, path: Path) -> SeverityConfig:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        class_weights: dict[str, float] = OmegaConf.to_container(  # type: ignore[assignment]
            cfg.class_weights, resolve=True
        )
        labels: list[str] = OmegaConf.to_container(cfg.labels, resolve=True)  # type: ignore[assignment]
        return cls(
            class_weights=class_weights,
            vehicle_region_padding=float(cfg.vehicle_region_padding),
            minor_max=float(cfg.thresholds.minor_max),
            moderate_max=float(cfg.thresholds.moderate_max),
            labels=labels,
        )


def normalise_class_name(name: str) -> str:
    """Map a human-readable class name ("glass shatter") to the config's
    underscored key ("glass_shatter"). CarDD class names contain spaces;
    YAML keys don't, so this is the single place that bridges the two."""
    return name.strip().lower().replace(" ", "_")


def vehicle_region_area(
    instances: list[DamageInstance], image_width: int, image_height: int, padding: float
) -> float:
    """Union bounding box of all instance boxes, padded outward by `padding`
    (fraction of the union box's own width/height on each side) and clipped
    to the image bounds. See module docstring for why this is only a proxy
    for the true vehicle extent, and its known biases.

    Falls back to the full image area when there are no instances (nothing
    to union), since a score of 0/0 is undefined and "no damage" should
    always grade as minor regardless of the denominator.
    """
    if not instances:
        return float(image_width * image_height)

    x0 = min(inst.bbox[0] for inst in instances)
    y0 = min(inst.bbox[1] for inst in instances)
    x1 = max(inst.bbox[2] for inst in instances)
    y1 = max(inst.bbox[3] for inst in instances)

    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = box_w * padding
    pad_y = box_h * padding

    px0 = max(0.0, x0 - pad_x)
    py0 = max(0.0, y0 - pad_y)
    px1 = min(float(image_width), x1 + pad_x)
    py1 = min(float(image_height), y1 + pad_y)

    return max(px1 - px0, 1.0) * max(py1 - py0, 1.0)


def severity_score(
    instances: list[DamageInstance],
    image_width: int,
    image_height: int,
    config: SeverityConfig,
) -> float:
    """Class-weighted damaged-area fraction — see module docstring for the
    formula. Returns 0.0 for an image with no detected instances."""
    if not instances:
        return 0.0

    region_area = vehicle_region_area(
        instances, image_width, image_height, config.vehicle_region_padding
    )

    weighted_area = 0.0
    for inst in instances:
        key = normalise_class_name(inst.class_name)
        weight = config.class_weights.get(key)
        if weight is None:
            raise KeyError(
                f"No severity weight configured for class '{inst.class_name}' "
                f"(normalised key '{key}'). Add it to configs/severity.yaml class_weights."
            )
        weighted_area += weight * inst.mask_area

    return weighted_area / region_area


def bucket_score(score: float, config: SeverityConfig) -> str:
    """Bucket a continuous severity score into minor / moderate / severe
    using the two thresholds in `configs/severity.yaml`."""
    if score < config.minor_max:
        return config.labels[0]
    if score < config.moderate_max:
        return config.labels[1]
    return config.labels[2]


def grade_image(
    instances: list[DamageInstance],
    image_width: int,
    image_height: int,
    config: SeverityConfig,
) -> dict[str, Any]:
    """Compute the full heuristic result for one image: raw score, bucketed
    label, and the per-instance weighted-area breakdown (for auditability —
    so a human reviewing a grading decision can see which instances drove it)."""
    score = severity_score(instances, image_width, image_height, config)
    label = bucket_score(score, config)

    region_area = vehicle_region_area(
        instances, image_width, image_height, config.vehicle_region_padding
    )
    breakdown = []
    for inst in instances:
        key = normalise_class_name(inst.class_name)
        weight = config.class_weights.get(key, 0.0)
        breakdown.append({
            "class_name": inst.class_name,
            "mask_area": inst.mask_area,
            "class_weight": weight,
            "weighted_area": weight * inst.mask_area,
            "weighted_area_fraction": (
                (weight * inst.mask_area) / region_area if region_area else 0.0
            ),
        })

    return {
        "score": score,
        "label": label,
        "vehicle_region_area": region_area,
        "instance_breakdown": breakdown,
    }
