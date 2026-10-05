"""Shared metrics schema and COCO-eval helpers for cross-model comparison.

Both ``train_yolo.py`` and ``train_maskrcnn.py`` write ``runs/<name>/metrics.json``
in this normalised schema (version 2; legacy v1 files lack ``schema_version``
and their per-class "precision" is really AP50) so ``autoassess.eval.compare`` can load either file
without framework-specific branching:

    {
        "schema_version": 2,                   # absent in legacy (v1) files
        "eval_split": "val" | "test",          # split the metrics were scored on
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
            "mask_iou_true_positives": float,  # mean IoU of matched pairs (>= 0.5)
            "mask_iou_per_gt": float,          # mean over all GTs, misses count 0
            "operating_point": {
                "conf": float,                 # deployment threshold (0.25)
                "iou": float,                  # match IoU (0.5)
                "scoring_conf": float,         # conf the detections were kept at (AP)
                "box": {"precision": float, "recall": float, "f1": float},   # macro
                "mask": {"precision": float, "recall": float, "f1": float}   # macro
            },
            "per_class": {
                "<class_name>": {
                    "box_ap50": float,
                    "mask_ap50": float,
                    "max_recall": float,       # mask, no conf cutoff (not P/R)
                    "precision": float,        # mask, at operating_point.conf
                    "recall": float,
                    "f1": float,
                    "tp": int, "fp": int, "fn": int,
                    "f1_opt_conf": float,      # mask, conf maximising F1
                    "f1_opt": float
                }
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
from dataclasses import dataclass, field
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
    """Run pycocotools COCOeval for one IoU type ("bbox" or "segm").

    Returns overall ``map50`` / ``map50_95`` and, per class, ``ap50`` (area
    under the interpolated precision-recall curve at IoU 0.5) and
    ``max_recall``: COCOeval's recall at IoU 0.5, all areas, maxDets=100, with
    no confidence cutoff. ``max_recall`` is the best recall reachable by
    keeping every detection; it is not an operating-point recall (see
    ``operating_point_metrics`` for that).
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
    for k, name in enumerate(class_names):
        precision = ev.eval["precision"][iou_idx, :, k, area_idx, maxdet_idx]
        precision = precision[precision > -1]
        recall = ev.eval["recall"][iou_idx, k, area_idx, maxdet_idx]
        per_class[name] = {
            "ap50": float(precision.mean()) if precision.size else 0.0,
            "max_recall": float(recall) if recall > -1 else 0.0,
        }

    return {
        "map50": map50,
        "map50_95": map50_95,
        "per_class": per_class,
    }


@dataclass
class ClassMatches:
    """Greedy-matching result for one class.

    ``dets`` holds one ``(score, matched, iou)`` per kept detection (iou is
    None when unmatched). ``gt_ids`` lists the class's non-crowd GT ids in id
    order and ``gt_ious`` the best matched IoU for each (0.0 for a miss).
    """

    dets: list[tuple[float, bool, float | None]] = field(default_factory=list)
    gt_ids: list[int] = field(default_factory=list)
    gt_ious: list[float] = field(default_factory=list)


def greedy_match(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    iou_type: str,
    conf_thr: float,
    iou_thr: float = 0.5,
    max_dets: int = 100,
) -> dict[int, ClassMatches]:
    """Greedy one-to-one detection/GT matching, per image and per class.

    For every (image, category): keep detections with ``score >= conf_thr``,
    take the top ``max_dets`` by score (as COCOeval's maxDets does per
    category), process them in descending score order, and let each claim the
    still-unmatched same-class GT with the highest IoU ``>= iou_thr`` (ties go
    to the lower GT id). Detections of another class never match. IoU is
    ``pycocotools.mask.iou`` on segmentation RLEs ("segm") or xywh boxes
    ("bbox"). Crowd GTs are skipped entirely. Returns ``{category_id:
    ClassMatches}`` for every category in ``coco_gt``.
    """
    from pycocotools import mask as mask_utils

    key = "segmentation" if iou_type == "segm" else "bbox"
    result: dict[int, ClassMatches] = {c: ClassMatches() for c in coco_gt.getCatIds()}
    gt_iou_by_id: dict[int, dict[int, float]] = {c: {} for c in result}

    for img_id in coco_gt.getImgIds():
        for cat_id in result:
            gts = [
                a
                for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=cat_id))
                if not a.get("iscrowd", 0)
            ]
            gts.sort(key=lambda a: int(a["id"]))
            for g in gts:
                gt_iou_by_id[cat_id][int(g["id"])] = 0.0

            dts = [
                a
                for a in coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id, catIds=cat_id))
                if float(a["score"]) >= conf_thr
            ]
            dts.sort(key=lambda a: -float(a["score"]))  # stable: ties keep load order
            dts = dts[:max_dets]
            if not dts:
                continue

            if gts:
                ious = mask_utils.iou(
                    [d[key] for d in dts], [g[key] for g in gts], [0] * len(gts)
                )
            taken: set[int] = set()
            for i, d in enumerate(dts):
                best_j, best_iou = -1, iou_thr
                if gts:
                    for j in range(len(gts)):
                        v = float(ious[i, j])
                        if j in taken or v < iou_thr:
                            continue
                        if best_j < 0 or v > best_iou:
                            best_j, best_iou = j, v
                if best_j >= 0:
                    taken.add(best_j)
                    gt_iou_by_id[cat_id][int(gts[best_j]["id"])] = best_iou
                    result[cat_id].dets.append((float(d["score"]), True, best_iou))
                else:
                    result[cat_id].dets.append((float(d["score"]), False, None))

    for cat_id, cm in result.items():
        cm.gt_ids = sorted(gt_iou_by_id[cat_id])
        cm.gt_ious = [gt_iou_by_id[cat_id][g] for g in cm.gt_ids]
    return result


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def operating_point_metrics(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    iou_type: str,
    class_ids: list[int],
    class_names: list[str],
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    """Per-class TP/FP/FN, precision, recall and F1 at one operating point.

    Uses ``greedy_match`` at ``conf_thr``/``iou_thr``. A TP is a matched
    detection, an FP an unmatched kept detection, an FN a GT left unmatched.
    Precision, recall and F1 are 0.0 when undefined. Returns
    ``{"per_class": {name: {...}}, "macro": {precision, recall, f1}}`` where
    macro is the unweighted mean over classes with at least one GT.
    """
    matches = greedy_match(coco_gt, coco_dt, iou_type, conf_thr, iou_thr)
    per_class: dict[str, dict[str, float]] = {}
    with_gt: list[str] = []
    for cid, name in zip(class_ids, class_names, strict=True):
        cm = matches.get(cid, ClassMatches())
        tp = sum(1 for _, ok, _ in cm.dets if ok)
        fp = len(cm.dets) - tp
        fn = len(cm.gt_ids) - tp
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}
        if cm.gt_ids:
            with_gt.append(name)
    macro = {
        f: (sum(per_class[n][f] for n in with_gt) / len(with_gt) if with_gt else 0.0)
        for f in ("precision", "recall", "f1")
    }
    return {"per_class": per_class, "macro": macro}


def f1_optimal_conf(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    iou_type: str,
    class_ids: list[int],
    class_names: list[str],
    iou_thr: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Per class, the confidence threshold that maximises F1.

    Matches once at ``conf_thr=0`` and sweeps cumulative TP/FP over the
    score-sorted detections, evaluating after each unique score (so ties are
    included whole, as ``score >= conf``). Returns ``{"conf", "f1",
    "precision", "recall"}`` at the max-F1 threshold; F1 ties go to the higher
    conf. A class with no detections reports conf 0.0 and zeros.

    Caveat: a sweep over one matching is only guaranteed equal to re-matching
    at each threshold when matching is prefix-determined, i.e. each detection's
    outcome depends only on higher-scored detections. That holds for this
    greedy rule (including the max_dets cap, which keeps a score prefix), and
    the tests check agreement on the fixture, but it would break for any
    matcher that looks at lower-scored detections. Matching is also assumed
    to break score ties by load order in both paths.
    """
    matches = greedy_match(coco_gt, coco_dt, iou_type, 0.0, iou_thr)
    out: dict[str, dict[str, float]] = {}
    for cid, name in zip(class_ids, class_names, strict=True):
        cm = matches.get(cid, ClassMatches())
        n_gt = len(cm.gt_ids)
        best = {"conf": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        dets = sorted(cm.dets, key=lambda d: -d[0])
        tp = fp = 0
        i = 0
        found = False
        while i < len(dets):
            score = dets[i][0]
            while i < len(dets) and dets[i][0] == score:
                if dets[i][1]:
                    tp += 1
                else:
                    fp += 1
                i += 1
            m = _prf(tp, fp, n_gt - tp)
            # strict > keeps the earlier (higher-conf) threshold on ties
            if not found or m["f1"] > best["f1"]:
                best = {"conf": score, **m}
                found = True
        out[name] = best
    return out


def mask_iou_true_positives(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
) -> float:
    """Mean mask IoU over true-positive matches from ``greedy_match`` (segm).

    Every counted IoU is ``>= iou_thr`` by construction, so this measures mask
    quality of correct detections only and is floored at ``iou_thr``; misses
    do not lower it (use ``mask_iou_per_gt`` for that). 0.0 with no matches.
    """
    ious = [
        iou
        for cm in greedy_match(coco_gt, coco_dt, "segm", conf_thr, iou_thr).values()
        for _, ok, iou in cm.dets
        if ok and iou is not None
    ]
    return statistics.fmean(ious) if ious else 0.0


def mask_iou_per_gt(
    coco_gt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    coco_dt: Any,  # noqa: ANN401 — pycocotools.coco.COCO has no stub package
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
) -> float:
    """Mean over every non-crowd GT of its matched mask IoU, misses as 0.0.

    Uses ``greedy_match`` (segm) at ``conf_thr``/``iou_thr``. 0.0 if no GTs.
    """
    ious = [
        v
        for cm in greedy_match(coco_gt, coco_dt, "segm", conf_thr, iou_thr).values()
        for v in cm.gt_ious
    ]
    return statistics.fmean(ious) if ious else 0.0


def build_metrics_json(
    run_name: str,
    model_type: str,
    model: str,
    class_names: list[str],
    epochs_requested: int | None,
    epochs_run_this_invocation: int | None,
    batch: int | None,
    imgsz: int,
    device: str,
    patience: int | None,
    best_epoch: int | None,
    early_stopped: bool | None,
    wall_time_seconds_total: float | None,
    epoch_wall_times_seconds: list[float] | None,
    eval_result: dict[str, Any],
    eval_split: str,
    scoring_conf: float,
    inference: dict[str, float],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the schema-v2 metrics.json payload documented at module level.

    ``eval_result`` is the output of ``coco_eval.run_coco_eval``. Per-class
    precision/recall/F1/TP/FP/FN are the mask values at the operating point.
    The training fields (epochs, batch, patience, best_epoch, early_stopped,
    wall times) may be None for a standalone evaluation of weights whose
    training history is unknown.
    """
    op = eval_result["operating_point"]
    mask_op = op["mask"]["per_class"]
    empty_ap = {"ap50": 0.0, "max_recall": 0.0}
    empty_f1 = {"conf": 0.0, "f1": 0.0}
    per_class: dict[str, dict[str, Any]] = {}
    for name in class_names:
        box_pc = eval_result["box"]["per_class"].get(name, empty_ap)
        mask_pc = eval_result["mask"]["per_class"].get(name, empty_ap)
        mop = mask_op.get(
            name, {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        )
        f1o = eval_result["f1_optimal"]["mask"].get(name, empty_f1)
        per_class[name] = {
            "box_ap50": box_pc["ap50"],
            "mask_ap50": mask_pc["ap50"],
            "max_recall": mask_pc.get("max_recall", 0.0),
            "precision": mop["precision"],
            "recall": mop["recall"],
            "f1": mop["f1"],
            "tp": mop["tp"],
            "fp": mop["fp"],
            "fn": mop["fn"],
            "f1_opt_conf": f1o["conf"],
            "f1_opt": f1o["f1"],
        }

    return {
        "schema_version": 2,
        "eval_split": eval_split,
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
            "box_map50": eval_result["box"]["map50"],
            "box_map50_95": eval_result["box"]["map50_95"],
            "mask_map50": eval_result["mask"]["map50"],
            "mask_map50_95": eval_result["mask"]["map50_95"],
            "mask_iou_true_positives": eval_result["mask_iou_true_positives"],
            "mask_iou_per_gt": eval_result["mask_iou_per_gt"],
            "operating_point": {
                "conf": op["conf"],
                "iou": op["iou"],
                "scoring_conf": scoring_conf,
                "box": dict(op["box"]["macro"]),
                "mask": dict(op["mask"]["macro"]),
            },
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
