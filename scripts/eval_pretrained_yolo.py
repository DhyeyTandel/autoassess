#!/usr/bin/env python3
"""scripts/eval_pretrained_yolo.py — write metrics.json for a YOLO checkpoint
that was trained outside `train_yolo.py`'s own run loop (e.g. on Colab, where
the process crashed before reaching final COCO eval / metrics.json write).

Reuses the exact same evaluation helpers `train_yolo.py` calls at the end of
a normal run, so the resulting metrics.json is directly comparable to every
other model's — this is not a different, looser evaluation path.

Usage
-----
    python scripts/eval_pretrained_yolo.py \\
        --weights runs/vehide_seg_v1/weights/best.pt \\
        --dataset-config configs/vehide.yaml \\
        --model-type yolov8-seg \\
        --model-name yolov8s-seg.pt \\
        --name vehide_seg_v1 \\
        --split val \\
        --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.eval.coco_eval import load_class_names  # noqa: E402
from autoassess.eval.metrics import (  # noqa: E402
    build_metrics_json,
    count_params,
    write_metrics_json,
)
from autoassess.eval.scoring import SCORING_CONF_THRESHOLD  # noqa: E402
from autoassess.eval.yolo_eval import (  # noqa: E402
    measure_yolo_latency,
    peak_vram_mb_for_yolo,
    reset_peak_vram_for_yolo,
    resolve_processed_dir,
    run_yolo_coco_eval,
)
from autoassess.utils.logging import get_logger, setup_run_logging  # noqa: E402

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write metrics.json for a pre-trained YOLO-seg checkpoint "
                    "that never reached train_yolo.py's own final-eval step."
    )
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to the .pt checkpoint to evaluate.")
    p.add_argument("--dataset-config", type=Path, required=True,
                   help="Dataset YAML (class names, processed data root).")
    p.add_argument("--model-type", type=str, default="yolov8-seg",
                   help="model_type field in metrics.json (default: yolov8-seg).")
    p.add_argument("--model-name", type=str, required=True,
                   help="model field in metrics.json, e.g. yolov8s-seg.pt.")
    p.add_argument("--name", type=str, required=True,
                   help="Run name; metrics.json is written to runs/<name>/metrics.json.")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"],
                   help="Which split to evaluate against (default: val).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Inference device: GPU index or 'cpu'/'mps' (default: cpu).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size used for latency measurement (default: 640).")
    p.add_argument("--epochs-completed", type=int, required=True,
                   help="How many epochs this checkpoint actually trained for — "
                        "read from the checkpoint's own 'epoch' field if unsure "
                        "(0-indexed in the .pt file, so add 1).")
    p.add_argument("--epochs-requested", type=int, required=True,
                   help="How many epochs the original run asked for (for the "
                        "epochs_requested field — this run never finished them).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.name
    setup_run_logging(run_dir)

    log.warning(
        "This checkpoint was trained outside train_yolo.py's own run loop (e.g. "
        "a Colab session that crashed before final eval) — early_stopped/patience "
        "fields below are not meaningful and are set to honest placeholders, not "
        "measured values."
    )

    from ultralytics import YOLO  # type: ignore[attr-defined]

    eval_model = YOLO(str(args.weights))
    class_names = load_class_names(args.dataset_config)
    processed_dir = resolve_processed_dir(args.dataset_config)

    device = args.device if args.device != "cpu" else "cpu"
    reset_peak_vram_for_yolo(device)

    images_dir = processed_dir / "images" / args.split
    labels_dir = processed_dir / "labels" / args.split
    log.info("Evaluating %s against %s / %s", args.weights, images_dir, labels_dir)

    final_eval = run_yolo_coco_eval(eval_model, images_dir, labels_dir, class_names)
    latency = measure_yolo_latency(eval_model, images_dir, args.imgsz)

    training_module = eval_model.model
    assert isinstance(training_module, torch.nn.Module)
    model_info = {
        **count_params(training_module),
        "vram_peak_mb": peak_vram_mb_for_yolo(device),
    }

    payload = build_metrics_json(
        run_name=args.name,
        model_type=args.model_type,
        model=args.model_name,
        class_names=class_names,
        epochs_requested=args.epochs_requested,
        epochs_run_this_invocation=args.epochs_completed,
        batch=0,  # unknown here — this script only re-evaluates, it doesn't train
        imgsz=args.imgsz,
        device=args.device,
        patience=0,
        best_epoch=args.epochs_completed,
        early_stopped=False,
        wall_time_seconds_total=0.0,
        epoch_wall_times_seconds=[],
        eval_result=final_eval,
        eval_split=args.split,
        scoring_conf=SCORING_CONF_THRESHOLD,
        inference=latency,
        model_info=model_info,
    )
    metrics_path = write_metrics_json(run_dir, payload)
    log.info("Wrote %s (evaluated on '%s' split, %d epochs completed).",
             metrics_path, args.split, args.epochs_completed)


if __name__ == "__main__":
    main()
