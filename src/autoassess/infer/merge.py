"""Merge detections from a secondary damage model into the primary one.

Rule
----
Keep every primary-model instance. Keep a secondary-model instance only if

  1. its class is listed in `secondary_classes`, AND
  2. its mask IoU (pycocotools, iscrowd=0) with *every* primary instance is
     below `suppress_iou`, AND
  3. its mask IoU with every secondary instance already kept is below
     `suppress_iou`. Candidates are processed in descending confidence (stable
     for ties), and class is ignored in this check.

The result is the primary instances in their original order followed by the
kept secondary ones in descending confidence order, plus a parallel list of
source tags ("primary" / "secondary").

Why
---
The primary model (CarDD) is clearly stronger on the damage types both
datasets share (e.g. dent AP50 0.53 vs 0.24 for VehiDE), so VehiDE is only
trusted for what CarDD cannot detect at all: torn, missing_part and punctured.
At conf 0.15 on the VehiDE test split, box precision/recall is torn 0.33/0.39,
missing_part 0.59/0.69, punctured 0.43/0.58. Lowering the confidence to 0.05
gains little recall and costs a lot of precision, so the secondary default is
0.15, the same as the primary model. Overlap suppression stops a secondary
box from double-counting damage the primary model already found.

Rule 3 exists because Ultralytics NMS is per-class: on the VehiDE model,
different classes (e.g. punctured and missing_part on one hole) or several
instances of one class (e.g. five torn masks on one grille) all survive NMS and
stack on the same region, so they are de-duplicated across classes here.

Thresholds live in `configs/severity.yaml` under `merge:`.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from pycocotools import mask as mask_utils

from autoassess.infer.associate import MaskInstance

PRIMARY = "primary"
SECONDARY = "secondary"


@dataclass
class MergeConfig:
    secondary_classes: frozenset[str]
    suppress_iou: float

    @classmethod
    def load(cls, path: Path) -> MergeConfig:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return cls(
            secondary_classes=frozenset(str(c) for c in cfg.merge.secondary_classes),
            suppress_iou=float(cfg.merge.suppress_iou),
        )


def merge_damage_instances(
    primary: list[MaskInstance],
    secondary: list[MaskInstance],
    secondary_classes: Collection[str],
    suppress_iou: float,
) -> tuple[list[MaskInstance], list[str]]:
    """Merge `secondary` into `primary` per the module docstring rule.
    Returns (merged instances, parallel source tags)."""
    merged = list(primary)
    tags = [PRIMARY] * len(primary)
    kept: list[MaskInstance] = []
    candidates = sorted(
        (i for i in secondary if i.class_name in secondary_classes),
        key=lambda i: i.confidence,
        reverse=True,
    )
    for inst in candidates:
        others = [*primary, *kept]
        if others:
            ious = mask_utils.iou(
                [inst.mask_rle], [o.mask_rle for o in others], [0] * len(others)
            )
            if float(ious.max()) >= suppress_iou:
                continue
        kept.append(inst)
    merged.extend(kept)
    tags.extend([SECONDARY] * len(kept))
    return merged, tags
