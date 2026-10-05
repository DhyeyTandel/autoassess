"""Shared Ultralytics-YOLO evaluation helpers.

Extracted from `train_yolo.py` so `train_parts.py` (a second YOLOv8-seg
model, trained on the vehicle-parts dataset instead of CarDD) can reuse the
same COCO-eval-via-pycocotools pipeline, latency benchmarking, and
peak-VRAM tracking without duplicating them — both models are scored
identically regardless of which dataset they were trained on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoassess.eval.coco_eval import build_coco_ground_truth, run_coco_eval
from autoassess.eval.metrics import measure_inference_latency
from autoassess.eval.scoring import OPERATING_CONF, SCORING_CONF_THRESHOLD, SCORING_MAX_DETS


def resolve_dataset_yaml_for_ultralytics(dataset_config_path: Path) -> str:
    """Return a path to a dataset YAML that Ultralytics will resolve correctly.

    Every checked-in dataset YAML in this project (`configs/cardd.yaml`,
    `configs/carparts.yaml`) documents its `path:` as "relative to this
    file's location" — but Ultralytics' own `check_det_dataset` does *not*
    resolve a relative `path:` against the YAML's directory. It resolves
    against the YAML's directory only if that combination already exists as
    a real path; otherwise it silently falls back to resolving relative to
    Ultralytics' own global datasets cache dir (`~/.../Ultralytics/settings
    -> datasets_dir`, a sibling of this repo, not inside it). Since this
    project always writes converted data under `data/processed/<name>`
    inside the repo, that fallback resolves to the wrong directory (or
    nowhere), and training fails with a "images not found" error that gives
    no hint the `path:` was ever misresolved.

    Rather than hardcode an absolute, machine-specific path into the
    checked-in config (which would break on every other checkout), this
    writes a temporary copy of the YAML with `path:` rewritten to an
    absolute path resolved against the *source* YAML's own directory — the
    semantics the comment in every one of these files already promises.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(dataset_config_path)
    if "path" in cfg:
        raw_path = str(cfg.path)
        resolved = (dataset_config_path.parent / raw_path).resolve()
        cfg.path = str(resolved)

    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file must outlive this function
        mode="w", suffix=".yaml", delete=False, prefix="autoassess_resolved_"
    )
    OmegaConf.save(cfg, tmp.name)
    return tmp.name


def resolve_processed_dir(dataset_config_path: Path) -> Path:
    """Return the dataset's processed-data root (the directory containing
    images/ and labels/) by resolving the dataset YAML's own `path:` field
    against that file's directory — the same semantics
    `resolve_dataset_yaml_for_ultralytics` promises, but returned as a Path
    for callers that need to build eval image/label paths directly rather
    than hand the YAML to Ultralytics. Deriving this from the YAML (instead
    of hardcoding e.g. `processed_dir / "cardd"`) is what lets train_yolo.py
    evaluate whichever dataset --dataset-config actually points at.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(dataset_config_path)
    raw_path = str(cfg.path)
    return (dataset_config_path.parent / raw_path).resolve()


def reset_peak_vram_for_yolo(device_arg: str) -> None:
    if device_arg not in ("cpu", "mps") and not device_arg.startswith("mps"):
        torch.cuda.reset_peak_memory_stats(int(device_arg) if device_arg.isdigit() else device_arg)


def peak_vram_mb_for_yolo(device_arg: str) -> float | None:
    """Peak VRAM in MB since the last `reset_peak_vram_for_yolo` call.

    Only CUDA exposes a true peak-memory counter; Ultralytics' own device
    string ('0', 'cpu', 'mps') doesn't map to a `torch.device` 1:1, so this
    mirrors `autoassess.eval.metrics.peak_vram_mb`'s CUDA-only behaviour
    using Ultralytics' own device argument convention.
    """
    if device_arg in ("cpu", "mps") or device_arg.startswith("mps"):
        return None
    device = int(device_arg) if device_arg.isdigit() else device_arg
    return torch.cuda.max_memory_allocated(device) / 1e6


def collect_yolo_detections(
    eval_model: Any,  # noqa: ANN401 — ultralytics.YOLO has no exported public type
    images_dir: Path,
    coco_gt_dict: dict[str, Any],
    imgsz: int = 640,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run `eval_model` over every image in `coco_gt_dict` and return
    COCO-format detections.

    Inference uses the scoring settings (`SCORING_CONF_THRESHOLD`,
    `SCORING_MAX_DETS`) so the full PR curve reaches pycocotools, and the
    same `imgsz` as latency measurement. `device` (Ultralytics convention:
    '0', 'cpu', 'mps') is forwarded to `predict` when not None; otherwise
    Ultralytics picks its own default device.
    """
    from pycocotools import mask as mask_utils

    file_name_to_id = {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}

    predict_kwargs: dict[str, Any] = {
        "conf": SCORING_CONF_THRESHOLD,
        "max_det": SCORING_MAX_DETS,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device is not None:
        predict_kwargs["device"] = device

    detections: list[dict[str, Any]] = []
    for file_name, image_id in file_name_to_id.items():
        results = eval_model.predict(str(images_dir / file_name), **predict_kwargs)
        if not results:
            # Ultralytics returns an empty list rather than raising when the image
            # itself fails to decode (truncated/corrupt JPEG) — treat as zero
            # detections for this image rather than crashing the whole eval run.
            continue
        result = results[0]
        if result.masks is None:
            continue
        height, width = result.orig_shape
        for poly_xy, cls, conf in zip(
            result.masks.xy, result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
        ):
            if poly_xy.shape[0] < 3:
                continue
            rle = polygon_xy_to_rle(poly_xy, width, height)
            bbox = mask_utils.toBbox(rle).tolist()
            detections.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "segmentation": rle,
                "bbox": bbox,
                "score": float(conf),
            })
    return detections


def run_yolo_coco_eval(
    eval_model: Any,  # noqa: ANN401 — ultralytics.YOLO has no exported public type
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    imgsz: int = 640,
    device: str | None = None,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Run inference over every image in `images_dir` (minus `exclude`) and score the
    predictions against COCO ground truth reconstructed from the converted
    YOLO-seg labels, using the same pycocotools pipeline as Mask R-CNN's
    trainer so every model in this project is scored identically.

    Predictions are made at the scoring threshold (see
    `autoassess.eval.scoring`), not Ultralytics' default conf=0.25.
    """
    coco_gt_dict = build_coco_ground_truth(images_dir, labels_dir, class_names, exclude)
    detections = collect_yolo_detections(eval_model, images_dir, coco_gt_dict, imgsz, device)
    return run_coco_eval(coco_gt_dict, detections, class_names)


def polygon_xy_to_rle(poly_xy: np.ndarray, width: int, height: int) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rles = mask_utils.frPyObjects([poly_xy.flatten().tolist()], height, width)
    rle = mask_utils.merge(rles)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle  # type: ignore[no-any-return]


def measure_yolo_latency(
    eval_model: Any,  # noqa: ANN401
    images_dir: Path,
    imgsz: int,
    n_images: int = 30,
    device: str | None = None,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, float]:
    image_paths = sorted(
        p for p in [*images_dir.glob("*.jpg"), *images_dir.glob("*.png")] if p.name not in exclude
    )[:n_images]

    def _predict(image_path: Path) -> None:
        # Latency is measured at the deployment operating point (conf=0.25),
        # not the near-zero scoring threshold, which would inflate NMS cost.
        kwargs: dict[str, Any] = {"imgsz": imgsz, "conf": OPERATING_CONF, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        eval_model.predict(str(image_path), **kwargs)

    return measure_inference_latency(_predict, image_paths)
