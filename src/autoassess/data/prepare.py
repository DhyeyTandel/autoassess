"""Data preparation entrypoint — autoassess-prep.

Orchestrates the full data pipeline:
    raw  →  interim (cleaning, annotation validation)
         →  processed (YOLO-format splits)

Usage
-----
autoassess-prep \\
    --config configs/base.yaml \\
    --stage  all          # 'clean' | 'convert' | 'split' | 'all'
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AutoAssess dataset: raw → interim → processed."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config.")
    p.add_argument(
        "--stage",
        choices=["clean", "convert", "split", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "data_prep"
    setup_run_logging(run_dir)
    seed_everything(cfg.project.seed)

    log.info("Data preparation stage: %s", args.stage)

    # TODO: implement each stage
    raise NotImplementedError(
        "Data pipeline not yet implemented. "
        "Implement clean → convert → split stages in autoassess/data/."
    )


if __name__ == "__main__":
    main()
