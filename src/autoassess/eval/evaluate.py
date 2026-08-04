"""Evaluation entrypoint — autoassess-eval.

Usage
-----
autoassess-eval \\
    --config  configs/yolov8_seg.yaml \\
    --weights runs/yolov8_seg_v1/weights/best.pt \\
    --device  0 \\
    --batch   8 \\
    --imgsz   640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autoassess.utils.config import load_config, merge_configs, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained AutoAssess model on the test split."
    )
    p.add_argument("--config",      type=Path, default=Path("configs/yolov8_seg.yaml"))
    p.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--weights",     type=Path, required=True,
                   help="Path to model weights (.pt file).")
    p.add_argument("--device",  type=str, default=None)
    p.add_argument("--batch",   type=int, default=None)
    p.add_argument("--imgsz",   type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_configs(args.base_config, args.config)

    if args.device is not None: cfg.train.device = args.device
    if args.batch  is not None: cfg.train.batch  = args.batch
    if args.imgsz  is not None: cfg.data.imgsz   = args.imgsz

    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / (cfg.project.experiment + "_eval")
    setup_run_logging(run_dir)
    log.info("Evaluating experiment: %s", cfg.project.experiment)

    seed_everything(cfg.project.seed)

    # TODO: implement evaluation loop
    raise NotImplementedError("Evaluation not yet implemented.")


if __name__ == "__main__":
    main()
