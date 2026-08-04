"""Centralised logging configuration for AutoAssess."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Modules should call this at the top level::

        log = get_logger(__name__)

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)


def setup_run_logging(run_dir: str | Path, level: int = logging.INFO) -> None:
    """Configure root logger to write to *stdout* and a log file inside *run_dir*.

    Should be called once at the top of every training / evaluation script.

    Args:
        run_dir: Directory for the current experiment (e.g. ``runs/yolov8_seg_v1``).
        level: Logging level. Defaults to ``logging.INFO``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers if called more than once
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
