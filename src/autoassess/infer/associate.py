"""Associate damage instances with vehicle parts — autoassess-associate.

Runs the CarDD damage model (`train_yolo.py`) and the parts model
(`train_parts.py`) over the same image, then assigns each damage instance to
the part it is most covered by — "which panel is this damage on."

Matching rule: coverage = area(damage ∩ part) / area(damage). Each damage
instance goes to the part with the highest coverage, provided it is at least
`--coverage-threshold` (default 0.5). Ties go to the higher part confidence,
then to input order. Otherwise the instance falls back to `"unassigned"`
rather than being forced onto the nearest-but-still-wrong panel.

Why coverage and not mask IoU: IoU penalises a small damage mask on a large
panel even when the damage sits entirely inside it. A dent fully inside a
door has IoU = dent_area / door_area, so a 2k px dent on an 80k px door gets
IoU 0.025 and was left unassigned under the old 0.10 IoU threshold.

Measured on the first 150 CarDD test images (431 damage instances), share
assigned to a part:
    14.4%  IoU >= 0.10,      parts conf 0.25  (old rule)
    19.5%  coverage >= 0.5,  parts conf 0.25
    25.8%  coverage >= 0.5,  parts conf 0.15  (current defaults)
On the parts test split, lowering the parts cutoff from 0.25 to 0.15 costs a
little precision (box P 0.873 -> 0.841) and gains a little recall
(R 0.967 -> 0.971).

Known gap: the remaining unassigned damage is mostly on panels the 6-class
parts model does not have (fender, quarter panel, grille) — see
`configs/carparts.yaml` "Known gap". Those should read "unassigned" rather
than be mislabelled.

Output is a structured JSON list, one entry per damage instance:

    {
        "part": str,              # panel name, or "unassigned"
        "damage_type": str,       # CarDD class name (e.g. "scratch")
        "severity": str,          # minor / moderate / severe — this instance
                                   # alone, via autoassess.models.severity
                                   # (see that module for the heuristic and
                                   # its circularity caveat)
        "mask_area_px": float,
        "confidence": float,      # damage model's own detection confidence
        "part_coverage": float | null  # fraction of the damage mask inside the
                                       # assigned part, null if unassigned
    }

Usage
-----
    autoassess-associate \\
        --damage-weights runs/yolov8_seg_v1/weights/best.pt \\
        --parts-weights  runs/parts_seg_v1/weights/best.pt \\
        --source image.jpg \\
        --out    reports/associate/image.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoassess.eval.coco_eval import load_class_names
from autoassess.models.severity import DamageInstance, SeverityConfig, grade_image
from autoassess.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_COVERAGE_THRESHOLD = 0.5
UNASSIGNED = "unassigned"


@dataclass
class MaskInstance:
    """One segmented instance, framework-agnostic — a damage detection or a
    part detection. `mask_rle` is a COCO RLE dict (pycocotools format),
    kept as RLE rather than a dense array since that's what pycocotools'
    mask routines consume directly and it's far cheaper to carry around."""

    class_name: str
    mask_rle: dict[str, Any]
    bbox: tuple[float, float, float, float]
    mask_area: float
    confidence: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Associate damage instances with vehicle parts by mask coverage."
    )
    p.add_argument("--damage-weights", type=Path, required=True,
                   help="Path to a trained CarDD damage model checkpoint (.pt).")
    p.add_argument("--parts-weights", type=Path, required=True,
                   help="Path to a trained parts model checkpoint (.pt).")
    p.add_argument("--damage-dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Dataset YAML for damage class names (default: configs/cardd.yaml).")
    p.add_argument("--parts-dataset-config", type=Path, default=Path("configs/carparts.yaml"),
                   help="Dataset YAML for part class names (default: configs/carparts.yaml).")
    p.add_argument("--severity-config", type=Path, default=Path("configs/severity.yaml"),
                   help="Heuristic severity config (default: configs/severity.yaml).")
    p.add_argument("--source", type=Path, required=True,
                   help="Path to a single input image.")
    p.add_argument("--damage-conf", type=float, default=0.15,
                   help="Confidence threshold for the damage model (default: 0.15).")
    p.add_argument("--parts-conf", type=float, default=0.15,
                   help="Confidence threshold for the parts model (default: 0.15).")
    p.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD,
                   help="Minimum coverage (damage area inside the part / damage area) for a "
                        f"damage instance to be assigned to a part rather than falling back "
                        f"to \"{UNASSIGNED}\" (default: {DEFAULT_COVERAGE_THRESHOLD}).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Inference device (default: cpu).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path (default: print to stdout only).")
    return p.parse_args()


def run_yolo_seg_inference(
    weights_path: Path, source: Path, conf: float, device: str
) -> list[MaskInstance]:
    """Run a trained YOLOv8-seg checkpoint over one image and return its
    instances as framework-agnostic MaskInstance objects with RLE masks at
    the image's original resolution (not the model's input resolution)."""
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO  # type: ignore[attr-defined]

    from autoassess.eval.yolo_eval import polygon_xy_to_rle

    model: Any = YOLO(str(weights_path))
    result = model.predict(str(source), conf=conf, device=device, verbose=False)[0]

    instances: list[MaskInstance] = []
    if result.masks is None:
        return instances

    height, width = result.orig_shape
    class_names = result.names

    for poly_xy, cls, box_conf in zip(
        result.masks.xy, result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
    ):
        if poly_xy.shape[0] < 3:
            continue
        rle = polygon_xy_to_rle(poly_xy, width, height)
        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
        x, y, w, h = bbox
        instances.append(MaskInstance(
            class_name=class_names[int(cls)],
            mask_rle=rle,
            bbox=(x, y, x + w, y + h),
            mask_area=area,
            confidence=float(box_conf),
        ))
    return instances


def mask_coverage(damage_rle: dict[str, Any], part_rle: dict[str, Any]) -> float:
    """Fraction of the damage mask lying inside the part mask:
    area(damage ∩ part) / area(damage). 0.0 when the damage area is 0."""
    from pycocotools import mask as mask_utils

    damage_area = float(mask_utils.area(damage_rle))
    if damage_area <= 0.0:
        return 0.0
    inter = mask_utils.merge([damage_rle, part_rle], intersect=True)
    return float(mask_utils.area(inter)) / damage_area


def associate_damage_to_parts(
    damage_instances: list[MaskInstance],
    part_instances: list[MaskInstance],
    coverage_threshold: float,
) -> list[tuple[MaskInstance, MaskInstance | None, float]]:
    """For each damage instance, find the part instance with the highest
    coverage (see `mask_coverage`). Ties go to the higher part confidence,
    then to input order. Returns (damage, best_part_or_None, coverage)
    triples — best_part is None (and coverage is 0.0) when there are no
    part instances, or the best coverage is below `coverage_threshold`
    (or zero).

    This is independent per damage instance (not a one-to-one assignment
    problem) — multiple damage instances legitimately map to the same part
    (e.g. two separate scratches on one door), so there is no "part already
    taken" bookkeeping here, unlike bipartite matching.
    """
    results: list[tuple[MaskInstance, MaskInstance | None, float]] = []
    for damage in damage_instances:
        best_part: MaskInstance | None = None
        best_cov = 0.0
        for part in part_instances:
            cov = mask_coverage(damage.mask_rle, part.mask_rle)
            if cov <= 0.0:
                continue
            if (
                best_part is None
                or cov > best_cov
                or (cov == best_cov and part.confidence > best_part.confidence)
            ):
                best_cov = cov
                best_part = part

        if best_part is None or best_cov < coverage_threshold:
            results.append((damage, None, 0.0))
        else:
            results.append((damage, best_part, best_cov))
    return results


def build_association_records(
    damage_instances: list[MaskInstance],
    associations: list[tuple[MaskInstance, MaskInstance | None, float]],
    image_width: int,
    image_height: int,
    severity_config: SeverityConfig,
) -> list[dict[str, Any]]:
    """Build the final per-instance JSON records, computing severity for
    each damage instance independently (see module docstring: this is a
    deliberate choice — a single instance's severity should not be inflated
    by unrelated damage elsewhere in the same photo)."""
    records = []
    for damage, part, coverage in associations:
        single_instance = [DamageInstance(
            class_name=damage.class_name, mask_area=damage.mask_area, bbox=damage.bbox
        )]
        grade = grade_image(single_instance, image_width, image_height, severity_config)

        records.append({
            "part": part.class_name if part is not None else UNASSIGNED,
            "damage_type": damage.class_name,
            "severity": grade["label"],
            "mask_area_px": damage.mask_area,
            "confidence": damage.confidence,
            "part_coverage": coverage if part is not None else None,
        })
    return records


def main() -> None:
    args = parse_args()

    damage_class_names = load_class_names(args.damage_dataset_config)
    part_class_names = load_class_names(args.parts_dataset_config)
    severity_config = SeverityConfig.load(args.severity_config)

    damage_instances = run_yolo_seg_inference(
        args.damage_weights, args.source, args.damage_conf, args.device
    )
    part_instances = run_yolo_seg_inference(
        args.parts_weights, args.source, args.parts_conf, args.device
    )

    from PIL import Image
    with Image.open(args.source) as im:
        width, height = im.size

    associations = associate_damage_to_parts(
        damage_instances, part_instances, args.coverage_threshold
    )
    records = build_association_records(
        damage_instances, associations, width, height, severity_config
    )

    n_unassigned = sum(1 for r in records if r["part"] == UNASSIGNED)
    log.info(
        "%s: %d damage instance(s), %d part instance(s), %d unassigned",
        args.source, len(damage_instances), len(part_instances), n_unassigned,
    )
    summary = {
        "source": str(args.source),
        "n_damage_instances": len(damage_instances),
        "n_part_instances": len(part_instances),
        "n_unassigned": n_unassigned,
        "damage_class_names": damage_class_names,
        "part_class_names": part_class_names,
        "coverage_threshold": args.coverage_threshold,
        "instances": records,
    }

    output_json = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_json, encoding="utf-8")
        print(f"Written: {args.out}")
    print(output_json)


if __name__ == "__main__":
    main()
