"""Associate damage instances with vehicle parts — autoassess-associate.

Runs the CarDD damage model (`train_yolo.py`) and the parts model
(`train_parts.py`) over the same image, then assigns each damage instance to
whichever part instance it overlaps most (by pixel mask IoU) — "which panel
is this damage on." Instances with no part overlapping above
`--iou-threshold` fall back to `"unassigned"` rather than being forced onto
the nearest-but-still-wrong panel: a low-confidence or genuinely
out-of-frame damage detection (e.g. damage on a wheel or mirror, panels the
parts model was never trained to segment — see `configs/carparts.yaml`
"Known gap") should say so rather than silently mislabel.

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
        "part_iou": float | null  # IoU with the assigned part, null if unassigned
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

DEFAULT_IOU_THRESHOLD = 0.10
UNASSIGNED = "unassigned"


@dataclass
class MaskInstance:
    """One segmented instance, framework-agnostic — a damage detection or a
    part detection. `mask_rle` is a COCO RLE dict (pycocotools format),
    kept as RLE rather than a dense array since that's what pycocotools'
    IoU routine consumes directly and it's far cheaper to carry around."""

    class_name: str
    mask_rle: dict[str, Any]
    bbox: tuple[float, float, float, float]
    mask_area: float
    confidence: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Associate damage instances with vehicle parts by mask IoU."
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
    p.add_argument("--damage-conf", type=float, default=0.25,
                   help="Confidence threshold for the damage model (default: 0.25).")
    p.add_argument("--parts-conf", type=float, default=0.25,
                   help="Confidence threshold for the parts model (default: 0.25).")
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                   help="Minimum mask IoU for a damage instance to be assigned to a part "
                        f"rather than falling back to \"{UNASSIGNED}\" (default: "
                        f"{DEFAULT_IOU_THRESHOLD}).")
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


def mask_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    from pycocotools import mask as mask_utils

    iou_matrix = mask_utils.iou([a], [b], [0])
    return float(iou_matrix[0, 0])


def associate_damage_to_parts(
    damage_instances: list[MaskInstance],
    part_instances: list[MaskInstance],
    iou_threshold: float,
) -> list[tuple[MaskInstance, MaskInstance | None, float]]:
    """For each damage instance, find the part instance with maximum mask
    IoU. Returns (damage, best_part_or_None, best_iou) triples — best_part
    is None (and best_iou is 0.0) when either there are no part instances at
    all, or the best IoU found is below `iou_threshold`.

    This is independent per damage instance (not a one-to-one assignment
    problem) — multiple damage instances legitimately map to the same part
    (e.g. two separate scratches on one door), so there is no "part already
    taken" bookkeeping here, unlike bipartite matching.
    """
    results: list[tuple[MaskInstance, MaskInstance | None, float]] = []
    for damage in damage_instances:
        best_part: MaskInstance | None = None
        best_iou = 0.0
        for part in part_instances:
            iou = mask_iou(damage.mask_rle, part.mask_rle)
            if iou > best_iou:
                best_iou = iou
                best_part = part

        if best_part is None or best_iou < iou_threshold:
            results.append((damage, None, 0.0))
        else:
            results.append((damage, best_part, best_iou))
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
    for damage, part, iou in associations:
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
            "part_iou": iou if part is not None else None,
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

    associations = associate_damage_to_parts(damage_instances, part_instances, args.iou_threshold)
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
        "iou_threshold": args.iou_threshold,
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
