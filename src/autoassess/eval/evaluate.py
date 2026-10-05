"""Standalone evaluator — autoassess-eval.

Scores any trained checkpoint (Ultralytics YOLO-seg or torchvision Mask R-CNN)
on one split of a processed dataset with the same pipeline the trainers use,
and writes a schema-v2 ``metrics.json`` to ``<output_root>/<name>_eval_<split>/``.
Training-time fields are carried over read-only from ``<weights run dir>/metrics.json``
(or the ``training_info`` argument); peak VRAM is measured over inference only.

Usage
-----
    autoassess-eval \\
        --weights runs/yolov8_seg_v1/weights/best.pt \\
        --dataset-config configs/cardd.yaml \\
        --split test \\
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from autoassess.eval.coco_eval import load_class_names
from autoassess.eval.metrics import (
    build_metrics_json,
    count_params,
    measure_inference_latency,
    peak_vram_mb,
    reset_peak_vram,
    write_metrics_json,
)
from autoassess.eval.scoring import SCORING_CONF_THRESHOLD
from autoassess.eval.yolo_eval import (
    measure_yolo_latency,
    resolve_processed_dir,
    run_yolo_coco_eval,
)
from autoassess.utils.config import load_config
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

DEFAULT_SEED = 42
MASKRCNN_MODEL_ID = "maskrcnn_resnet50_fpn_v2"
YOLO_MODEL_TYPE = "yolov8-seg"
SPLITS = ("train", "val", "test")

# Fields copied (read-only) from a sibling training metrics.json.
_TRAINING_KEYS = (
    "epochs_requested",
    "epochs_run_this_invocation",
    "batch",
    "patience",
    "best_epoch",
    "early_stopped",
    "wall_time_seconds_total",
    "epoch_wall_times_seconds",
)


def detect_model_type(weights: Path) -> str:
    """Return ``"maskrcnn"`` for a Mask R-CNN checkpoint, else ``"yolo"``.

    A Mask R-CNN checkpoint written by ``train_maskrcnn`` is a dict holding
    ``model_state_dict``; anything else is treated as an Ultralytics checkpoint.
    """
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - Ultralytics .pt files may need their own classes
        log.info("Could not torch.load %s directly; treating it as Ultralytics.", weights)
        return "yolo"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return "maskrcnn"
    return "yolo"


def _load_yolo(weights: Path) -> Any:  # noqa: ANN401 - ultralytics.YOLO has no public type
    from ultralytics import YOLO  # type: ignore[attr-defined]

    return YOLO(str(weights))


def _load_maskrcnn(weights: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Build Mask R-CNN (no pretrained download), load the checkpoint, move to device."""
    from autoassess.train.train_maskrcnn import build_model

    model = build_model(num_classes + 1, pretrained=False)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        return torch.device("mps")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _device_name(device: torch.device, device_arg: str) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device_arg


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_run_name(weights: Path) -> str:
    # runs/<name>/weights/best.pt -> <name>
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def _read_training_fields(weights: Path) -> dict[str, Any]:
    """Read-only: training fields from ``<weights run dir>/metrics.json``, if present."""
    if weights.parent.name != "weights":
        return {}
    sibling = weights.parent.parent / "metrics.json"
    if not sibling.is_file():
        return {}
    try:
        data = json.loads(sibling.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read training fields from %s; leaving them null.", sibling)
        return {}
    fields = {k: data[k] for k in _TRAINING_KEYS if k in data}
    vram_train = data.get("model_info", {}).get("vram_peak_mb_training")
    if vram_train is not None:
        fields["vram_peak_mb_training"] = vram_train
    log.info("Training fields taken from %s", sibling)
    return fields


def _check_split(images_dir: Path, labels_dir: Path) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory for split does not exist: {images_dir}")
    if not any(images_dir.glob("*.jpg")) and not any(images_dir.glob("*.png")):
        raise FileNotFoundError(f"No .jpg/.png images found in split directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory for split does not exist: {labels_dir}")


def run_evaluation(
    weights: Path,
    model_type: str,
    dataset_config: Path,
    split: str = "test",
    run_name: str | None = None,
    output_root: Path = Path("runs"),
    device: str = "cpu",
    imgsz: int = 640,
    latency_images: int = 30,
    training_info: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Evaluate ``weights`` on ``split`` and write metrics.json; return its path.

    Output lands in ``<output_root>/<run_name>_eval_<split>/metrics.json``.
    ``run_name`` defaults to the weights' run directory name. Training fields
    come from ``training_info`` if given, else from the sibling run's
    metrics.json (read-only), else are None. ``vram_peak_mb`` covers inference
    only (reset after model load, read after scoring and latency).
    """
    if model_type not in ("yolo", "maskrcnn"):
        raise ValueError(f"model_type must be 'yolo' or 'maskrcnn', got {model_type!r}")

    name = run_name or _default_run_name(weights)
    run_dir = output_root / f"{name}_eval_{split}"
    setup_run_logging(run_dir)
    seed_everything(seed)

    class_names = load_class_names(dataset_config)
    processed_dir = resolve_processed_dir(dataset_config)
    images_dir = processed_dir / "images" / split
    labels_dir = processed_dir / "labels" / split
    _check_split(images_dir, labels_dir)
    n_images = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    torch_device = _resolve_device(device)
    log.info(
        "Evaluating %s (%s) | dataset=%s split=%s images=%d | device=%s imgsz=%d "
        "latency_images=%d seed=%d | output=%s",
        weights, model_type, dataset_config, split, n_images, device, imgsz,
        latency_images, seed, run_dir,
    )

    if model_type == "yolo":
        model_id = weights.name
        eval_model = _load_yolo(weights)
        module = eval_model.model
        assert isinstance(module, torch.nn.Module)
        # Ultralytics' checkpoint loader sets requires_grad=False on every param;
        # restore it so params_trainable is not reported as 0 (same as train_yolo).
        for p in module.parameters():
            p.requires_grad_(True)

        reset_peak_vram(torch_device)
        eval_result = run_yolo_coco_eval(
            eval_model, images_dir, labels_dir, class_names, imgsz=imgsz
        )
        latency = measure_yolo_latency(eval_model, images_dir, imgsz, n_images=latency_images)
        model_type_field = YOLO_MODEL_TYPE
    else:
        from autoassess.train.train_maskrcnn import CarDDSegmentationDataset, evaluate_split

        model_id = MASKRCNN_MODEL_ID
        module = _load_maskrcnn(weights, len(class_names), torch_device)
        dataset = CarDDSegmentationDataset(images_dir, labels_dir, imgsz=imgsz, augment=False)

        reset_peak_vram(torch_device)
        eval_result = evaluate_split(module, dataset, torch_device, class_names)
        with torch.inference_mode():
            latency = measure_inference_latency(
                lambda img: module([img.to(torch_device)]),
                [dataset[i][0] for i in range(min(len(dataset), latency_images))],
            )
        model_type_field = "maskrcnn"

    _sync_if_cuda(torch_device)
    vram = peak_vram_mb(torch_device)

    train_fields = _read_training_fields(weights)
    if training_info:
        train_fields.update(training_info)
    vram_training = train_fields.pop("vram_peak_mb_training", None)

    model_info: dict[str, Any] = {
        **count_params(module),
        "vram_peak_mb": vram,
        "vram_scope": "inference",
        "device_name": _device_name(torch_device, device),
    }
    if vram_training is not None:
        model_info["vram_peak_mb_training"] = vram_training

    payload = build_metrics_json(
        run_name=name,
        model_type=model_type_field,
        model=model_id,
        class_names=class_names,
        epochs_requested=train_fields.get("epochs_requested"),
        epochs_run_this_invocation=train_fields.get("epochs_run_this_invocation"),
        batch=train_fields.get("batch"),
        imgsz=imgsz,
        device=device,
        patience=train_fields.get("patience"),
        best_epoch=train_fields.get("best_epoch"),
        early_stopped=train_fields.get("early_stopped"),
        wall_time_seconds_total=train_fields.get("wall_time_seconds_total"),
        epoch_wall_times_seconds=train_fields.get("epoch_wall_times_seconds"),
        eval_result=eval_result,
        eval_split=split,
        scoring_conf=SCORING_CONF_THRESHOLD,
        inference=latency,
        model_info=model_info,
    )
    metrics_path = write_metrics_json(run_dir, payload)

    m = payload["metrics"]
    log.info(
        "Done | split=%s | mask mAP50=%.4f mAP50-95=%.4f | box mAP50=%.4f mAP50-95=%.4f | "
        "latency mean=%.1fms p95=%.1fms | vram_peak_mb(inference)=%s | wrote %s",
        split, m["mask_map50"], m["mask_map50_95"], m["box_map50"], m["box_map50_95"],
        latency["latency_ms_mean"], latency["latency_ms_p95"], vram, metrics_path,
    )
    return metrics_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained AutoAssess checkpoint on one dataset split and "
        "write a schema-v2 metrics.json."
    )
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to the checkpoint (.pt) to evaluate.")
    p.add_argument("--model-type", type=str, choices=["yolo", "maskrcnn"], default=None,
                   help="Model family. Default: auto-detect from the checkpoint.")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Dataset YAML (class names, processed data root).")
    p.add_argument("--split", type=str, choices=list(SPLITS), default="test",
                   help="Split to score (default: test).")
    p.add_argument("--name", type=str, default=None,
                   help="Run name; output goes to <output_root>/<name>_eval_<split>/. "
                        "Default: the weights' run directory name.")
    p.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"),
                   help="Base YAML config (output_root, seed).")
    p.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu",
                   help="Device: GPU index, 'cpu' or 'mps' (default: 0 if CUDA is available, "
                        "else cpu).")
    p.add_argument("--batch", type=int, default=8,
                   help="Accepted for CLI consistency; unused (inference is per-image).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size (default: 640).")
    p.add_argument("--latency-images", type=int, default=30,
                   help="Number of images used to measure latency (default: 30).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.base_config)
    output_root = Path(str(cfg.project.output_root))
    seed = int(cfg.project.get("seed", DEFAULT_SEED))
    model_type = args.model_type or detect_model_type(args.weights)

    run_evaluation(
        weights=args.weights,
        model_type=model_type,
        dataset_config=args.dataset_config,
        split=args.split,
        run_name=args.name,
        output_root=output_root,
        device=args.device,
        imgsz=args.imgsz,
        latency_images=args.latency_images,
        seed=seed,
    )


if __name__ == "__main__":
    main()
