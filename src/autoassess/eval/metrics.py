"""Shared metrics schema and COCO-eval helpers for cross-model comparison.

Both ``train_yolo.py`` and ``train_maskrcnn.py`` write ``runs/<name>/metrics.json``
in this normalised schema so ``autoassess.eval.compare`` can load either file
without framework-specific branching:

    {
        "run_name": str,
        "model_type": "yolov8-seg" | "maskrcnn",
        "model": str,                          # checkpoint / architecture id
        "class_names": [str, ...],
        "epochs_requested": int,
        "epochs_run_this_invocation": int,
        "batch": int,
        "imgsz": int,
        "device": str,
        "patience": int,
        "best_epoch": int,
        "early_stopped": bool,
        "wall_time_seconds_total": float,
        "wall_time_seconds_per_epoch_mean": float | null,
        "epoch_wall_times_seconds": [float, ...],
        "metrics": {
            "box_map50": float,
            "box_map50_95": float,
            "mask_map50": float,
            "mask_map50_95": float,
            "mask_iou_mean": float,
            "per_class": {
                "<class_name>": {"precision": float, "recall": float, "ap50": float}
            }
        },
        "inference": {
            "latency_ms_mean": float,
            "latency_ms_p50": float,
            "latency_ms_p95": float,
            "n_images": int
        },
        "model_info": {
            "params_total": int,
            "params_trainable": int,
            "vram_peak_mb": float | null   # null on non-CUDA devices (no peak-memory API)
        }
    }
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

COCO_IOU_THRESHOLDS_50 = 0.5


def count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device: torch.device) -> float | None:
    """Peak VRAM in MB since the last `reset_peak_vram` call.

    Only CUDA exposes a true peak-memory counter (`max_memory_allocated`).
    MPS only exposes *current* allocated memory, which would systematically
    understate the true peak (fragmentation, freed-but-not-yet-reclaimed
    buffers) — returning null there rather than a misleading number.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return None


def measure_inference_latency(
    predict_fn: Any,  # noqa: ANN401 — callable signature varies per framework (YOLO vs torchvision)
    images: list[Any],
    warmup: int = 5,
) -> dict[str, float]:
    """Benchmark per-image inference latency in milliseconds.

    `predict_fn` is called once per image (not batched), matching the
    single-image-at-a-time latency a real claims-intake pipeline would see.
    """
    for img in images[:warmup]:
        predict_fn(img)

    latencies_ms = []
    for img in images:
        start = time.perf_counter()
        predict_fn(img)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "latency_ms_mean": statistics.mean(latencies_ms) if latencies_ms else float("nan"),
        "latency_ms_p50": latencies_ms[n // 2] if latencies_ms else float("nan"),
        "latency_ms_p95": latencies_ms[min(int(n * 0.95), n - 1)] if latencies_ms else float("nan"),
        "n_images": n,
    }


def coco_eval_summary(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    iou_type: str,
    class_ids: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Run pycocotools COCOeval for one IoU type ("bbox" or "segm") and
    return overall mAP50 / mAP50-95 plus per-class AP50 and precision/recall
    at IoU=0.5, conf-independent (COCOeval's own precision/recall arrays).
    """
    from pycocotools.cocoeval import COCOeval

    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    map50_95 = float(ev.stats[0])
    map50 = float(ev.stats[1])

    # precision/recall arrays: [T, R, K, A, M] = IoU thresholds x recall thresholds x
    # classes x area ranges x max dets. Index for IoU=0.5, all areas, max dets=last.
    iou_idx = 0  # ev.params.iouThrs[0] == 0.50
    area_idx = 0  # 'all'
    maxdet_idx = -1

    per_class: dict[str, dict[str, float]] = {}
    for k, (class_id, name) in enumerate(zip(class_ids, class_names, strict=True)):
        precision = ev.eval["precision"][iou_idx, :, k, area_idx, maxdet_idx]
        precision = precision[precision > -1]
        recall_curve = ev.params.recThrs
        recall = ev.eval["recall"][iou_idx, k, area_idx, maxdet_idx]

        ap50 = float(precision.mean()) if precision.size else 0.0
        mean_precision = ap50  # COCO AP at a fixed IoU *is* the mean precision over recall levels
        mean_recall = float(recall) if recall > -1 else 0.0
        per_class[name] = {
            "precision": mean_precision,
            "recall": mean_recall,
            "ap50": ap50,
        }
        del recall_curve  # unused beyond documenting the axis; kept for readability

    return {
        "map50": map50,
        "map50_95": map50_95,
        "per_class": per_class,
    }


def mask_iou_mean(coco_gt: Any, coco_dt: Any, iou_thr: float = 0.5) -> float:  # noqa: ANN401
    """Mean IoU of matched (IoU >= iou_thr) predicted-vs-GT mask pairs across
    the whole eval set, using pycocotools' own mask IoU computation so it's
    consistent with the mAP numbers above rather than a separately
    hand-rolled IoU routine.
    """
    from pycocotools import mask as mask_utils

    ious_all: list[float] = []
    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        if not gt_anns or not dt_anns:
            continue

        gt_rles = [ann["segmentation"] for ann in gt_anns]
        dt_rles = [ann["segmentation"] for ann in dt_anns]
        iscrowd = [0] * len(gt_rles)
        iou_matrix = mask_utils.iou(dt_rles, gt_rles, iscrowd)
        if iou_matrix.size == 0:
            continue

        # greedy one-to-one matching by descending IoU, mirroring COCOeval's own matcher
        matched_gt: set[int] = set()
        for dt_idx in range(iou_matrix.shape[0]):
            best_gt, best_iou = -1, iou_thr
            for gt_idx in range(iou_matrix.shape[1]):
                if gt_idx in matched_gt:
                    continue
                if iou_matrix[dt_idx, gt_idx] > best_iou:
                    best_iou, best_gt = iou_matrix[dt_idx, gt_idx], gt_idx
            if best_gt >= 0:
                matched_gt.add(best_gt)
                ious_all.append(float(best_iou))

    return statistics.mean(ious_all) if ious_all else 0.0


def build_metrics_json(
    run_name: str,
    model_type: str,
    model: str,
    class_names: list[str],
    epochs_requested: int,
    epochs_run_this_invocation: int,
    batch: int,
    imgsz: int,
    device: str,
    patience: int,
    best_epoch: int,
    early_stopped: bool,
    wall_time_seconds_total: float,
    epoch_wall_times_seconds: list[float],
    box_metrics: dict[str, Any],
    mask_metrics: dict[str, Any],
    mask_iou: float,
    inference: dict[str, float],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the normalised metrics.json payload documented at module level."""
    default_pc = {"precision": 0.0, "recall": 0.0, "ap50": 0.0}
    per_class: dict[str, dict[str, float]] = {}
    for name in class_names:
        box_pc = box_metrics["per_class"].get(name, default_pc)
        mask_pc = mask_metrics["per_class"].get(name, default_pc)
        per_class[name] = {
            "precision": mask_pc["precision"],
            "recall": mask_pc["recall"],
            "box_ap50": box_pc["ap50"],
            "mask_ap50": mask_pc["ap50"],
        }

    return {
        "run_name": run_name,
        "model_type": model_type,
        "model": model,
        "class_names": class_names,
        "epochs_requested": epochs_requested,
        "epochs_run_this_invocation": epochs_run_this_invocation,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "patience": patience,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "wall_time_seconds_total": wall_time_seconds_total,
        "wall_time_seconds_per_epoch_mean": (
            sum(epoch_wall_times_seconds) / len(epoch_wall_times_seconds)
            if epoch_wall_times_seconds else None
        ),
        "epoch_wall_times_seconds": epoch_wall_times_seconds,
        "metrics": {
            "box_map50": box_metrics["map50"],
            "box_map50_95": box_metrics["map50_95"],
            "mask_map50": mask_metrics["map50"],
            "mask_map50_95": mask_metrics["map50_95"],
            "mask_iou_mean": mask_iou,
            "per_class": per_class,
        },
        "inference": inference,
        "model_info": model_info,
    }


def write_metrics_json(run_dir: Path, payload: dict[str, Any]) -> Path:
    import json

    metrics_path = run_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path
