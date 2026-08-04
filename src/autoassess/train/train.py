"""Training entrypoint — autoassess-train.

Usage
-----
autoassess-train \\
    --config configs/yolov8_seg.yaml \\
    --device 0 \\
    --batch  8 \\
    --epochs 50 \\
    --imgsz  640
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
        description="Train AutoAssess damage detection / segmentation model."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolov8_seg.yaml"),
        help="Path to experiment YAML config (merged on top of base.yaml).",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to base YAML config.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: GPU index (e.g. '0') or 'cpu'.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size per training step.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size (square, pixels).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Build merged config ---
    cfg = merge_configs(args.base_config, args.config)

    # CLI overrides
    if args.device  is not None: cfg.train.device  = args.device
    if args.batch   is not None: cfg.train.batch    = args.batch
    if args.epochs  is not None: cfg.train.epochs   = args.epochs
    if args.imgsz   is not None: cfg.data.imgsz     = args.imgsz

    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    # --- Setup run directory & logging ---
    run_dir = Path(cfg.project.output_root) / cfg.project.experiment
    setup_run_logging(run_dir)
    log.info("Starting run: %s", cfg.project.experiment)

    # --- Seed ---
    seed_everything(cfg.project.seed)
    log.info("Seed set to %d", cfg.project.seed)

    # --- Save config snapshot ---
    from omegaconf import OmegaConf
    snap = run_dir / "config.yaml"
    OmegaConf.save(cfg, snap)
    log.info("Config snapshot saved → %s", snap)

    # --- TODO: instantiate model and call trainer ---
    raise NotImplementedError(
        "Model code not yet implemented. "
        "Scaffold complete — implement autoassess.models and autoassess.train.trainer next."
    )


if __name__ == "__main__":
    main()
