"""Inference entrypoint — autoassess-infer.

Usage
-----
autoassess-infer \\
    --config  configs/yolov8_seg.yaml \\
    --weights runs/yolov8_seg_v1/weights/best.pt \\
    --source  path/to/image_or_dir \\
    --device  cpu \\
    --imgsz   640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autoassess.utils.config import merge_configs, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AutoAssess inference on an image or directory."
    )
    p.add_argument("--config",      type=Path, default=Path("configs/yolov8_seg.yaml"))
    p.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--weights",     type=Path, required=True,
                   help="Path to trained model weights (.pt).")
    p.add_argument("--source",      type=Path, required=True,
                   help="Path to input image or directory of images.")
    p.add_argument("--device",  type=str, default=None)
    p.add_argument("--imgsz",   type=int, default=None)
    p.add_argument("--conf",    type=float, default=None,
                   help="Confidence threshold (overrides config).")
    p.add_argument("--iou",     type=float, default=None,
                   help="NMS IoU threshold (overrides config).")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Directory to save annotated output images.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_configs(args.base_config, args.config)

    if args.device is not None: cfg.train.device    = args.device
    if args.imgsz  is not None: cfg.data.imgsz      = args.imgsz
    if args.conf   is not None: cfg.infer.conf_threshold = args.conf
    if args.iou    is not None: cfg.infer.iou_threshold  = args.iou

    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / (cfg.project.experiment + "_infer")
    setup_run_logging(run_dir)
    log.info("Running inference: source=%s", args.source)

    seed_everything(cfg.project.seed)

    # TODO: implement inference pipeline
    raise NotImplementedError("Inference pipeline not yet implemented.")


if __name__ == "__main__":
    main()
