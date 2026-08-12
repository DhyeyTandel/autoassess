"""End-to-end claims pipeline — autoassess-pipeline.

Chains the whole system into one call per image:

    image -> damage detection -> part detection -> association
          -> per-instance severity grade -> triage decision

The first four stages are exactly `autoassess.infer.associate` (damage
model + parts model, IoU-matched, graded per instance — see that module's
docstring for the matching rule and the severity heuristic's circularity
caveat, which applies here unchanged). This module adds the last stage:
turning the per-instance severity grades into a single triage decision for
the whole claim.

Triage rule (thresholds in `configs/severity.yaml` under `triage:`)
---------------------------------------------------------------------
    auto_approve:  every instance is "minor" AND total damaged mask area
                   (summed across instances, in px) is below
                   auto_approve_max_total_area_px
    total_loss_review: any instance is "severe" OR total area exceeds
                   total_loss_min_total_area_px
    human_review:  everything else — the default, safe fallback

This is a heuristic on top of a heuristic: severity grades come from
`autoassess.models.severity`'s hand-authored area-weighted formula (not
ground truth — see that module's circularity warning), and the triage
thresholds here are likewise hand-picked, not fit to real claims outcomes.
Treat "auto_approve" and "total_loss_review" as *routing* decisions that
still deserve a human spot-check until both layers are validated against
real adjuster decisions, not as authorization to skip human review in
production.

Usage
-----
    autoassess-pipeline \\
        --damage-weights runs/yolov8_seg_v1/weights/best.pt \\
        --parts-weights  runs/parts_seg_v1/weights/best.pt \\
        --source image.jpg \\
        --out    reports/pipeline/image.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from autoassess.eval.coco_eval import load_class_names
from autoassess.infer.associate import (
    MaskInstance,
    associate_damage_to_parts,
    build_association_records,
    run_yolo_seg_inference,
)
from autoassess.models.severity import SeverityConfig
from autoassess.utils.logging import get_logger

log = get_logger(__name__)

AUTO_APPROVE = "auto_approve"
HUMAN_REVIEW = "human_review"
TOTAL_LOSS_REVIEW = "total_loss_review"


@dataclass
class TriageConfig:
    auto_approve_max_total_area_px: float
    total_loss_min_total_area_px: float

    @classmethod
    def load(cls, path: Path) -> TriageConfig:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return cls(
            auto_approve_max_total_area_px=float(cfg.triage.auto_approve_max_total_area_px),
            total_loss_min_total_area_px=float(cfg.triage.total_loss_min_total_area_px),
        )


def triage_decision(instances: list[dict[str, Any]], config: TriageConfig) -> dict[str, Any]:
    """Apply the triage rule (see module docstring) to a list of
    `associate.build_association_records` output dicts. Returns the decision
    plus the numbers that drove it, so a reviewer can see why."""
    total_area = sum(r["mask_area_px"] for r in instances)
    any_severe = any(r["severity"] == "severe" for r in instances)
    all_minor = all(r["severity"] == "minor" for r in instances)

    if any_severe or total_area > config.total_loss_min_total_area_px:
        decision = TOTAL_LOSS_REVIEW
    elif all_minor and total_area <= config.auto_approve_max_total_area_px:
        decision = AUTO_APPROVE
    else:
        decision = HUMAN_REVIEW

    return {
        "decision": decision,
        "total_damaged_area_px": total_area,
        "n_instances": len(instances),
        "n_severe": sum(1 for r in instances if r["severity"] == "severe"),
        "n_moderate": sum(1 for r in instances if r["severity"] == "moderate"),
        "n_minor": sum(1 for r in instances if r["severity"] == "minor"),
    }


def run_pipeline_with_masks(
    source: Path,
    damage_weights: Path,
    parts_weights: Path,
    damage_dataset_config: Path,
    parts_dataset_config: Path,
    severity_config_path: Path,
    triage_config_path: Path,
    damage_conf: float = 0.25,
    parts_conf: float = 0.25,
    iou_threshold: float = 0.10,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[MaskInstance], list[MaskInstance]]:
    """Run the full image -> triage-decision pipeline. Returns the
    JSON-safe result dict plus the raw damage/part `MaskInstance` lists
    (RLE masks + boxes) for callers that need to draw them, e.g. `app.py`.
    `run_pipeline` below is the JSON-only convenience wrapper the CLI uses.
    """
    damage_class_names = load_class_names(damage_dataset_config)
    part_class_names = load_class_names(parts_dataset_config)
    severity_config = SeverityConfig.load(severity_config_path)
    triage_config = TriageConfig.load(triage_config_path)

    damage_instances = run_yolo_seg_inference(damage_weights, source, damage_conf, device)
    part_instances = run_yolo_seg_inference(parts_weights, source, parts_conf, device)

    with Image.open(source) as im:
        width, height = im.size

    associations = associate_damage_to_parts(damage_instances, part_instances, iou_threshold)
    records = build_association_records(
        damage_instances, associations, width, height, severity_config
    )
    triage = triage_decision(records, triage_config)

    log.info(
        "%s: %d damage instance(s), %d part instance(s) -> %s",
        source, len(damage_instances), len(part_instances), triage["decision"],
    )

    result = {
        "source": str(source),
        "image_width": width,
        "image_height": height,
        "damage_class_names": damage_class_names,
        "part_class_names": part_class_names,
        "instances": records,
        "triage": triage,
    }
    return result, damage_instances, part_instances


def run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """JSON-only convenience wrapper around `run_pipeline_with_masks`, for
    the CLI and any caller that doesn't need the raw masks."""
    result, _damage_instances, _part_instances = run_pipeline_with_masks(*args, **kwargs)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full damage -> part -> severity -> triage pipeline on one image."
    )
    p.add_argument("--damage-weights", type=Path, required=True)
    p.add_argument("--parts-weights", type=Path, required=True)
    p.add_argument("--damage-dataset-config", type=Path, default=Path("configs/cardd.yaml"))
    p.add_argument("--parts-dataset-config", type=Path, default=Path("configs/carparts.yaml"))
    p.add_argument("--severity-config", type=Path, default=Path("configs/severity.yaml"))
    p.add_argument("--triage-config", type=Path, default=Path("configs/severity.yaml"),
                   help="Config containing the triage: section (default: configs/severity.yaml, "
                        "same file as --severity-config).")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--damage-conf", type=float, default=0.25)
    p.add_argument("--parts-conf", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        source=args.source,
        damage_weights=args.damage_weights,
        parts_weights=args.parts_weights,
        damage_dataset_config=args.damage_dataset_config,
        parts_dataset_config=args.parts_dataset_config,
        severity_config_path=args.severity_config,
        triage_config_path=args.triage_config,
        damage_conf=args.damage_conf,
        parts_conf=args.parts_conf,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )

    output_json = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_json, encoding="utf-8")
        print(f"Written: {args.out}")
    print(output_json)


if __name__ == "__main__":
    main()
