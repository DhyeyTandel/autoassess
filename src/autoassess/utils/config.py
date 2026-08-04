"""Config loading and merging utilities using OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed OmegaConf DictConfig.
    """
    cfg: DictConfig = OmegaConf.load(Path(path))
    return cfg


def merge_configs(*configs: str | Path | DictConfig) -> DictConfig:
    """Merge multiple configs left-to-right (later values override earlier ones).

    Each argument may be a path to a YAML file or an already-loaded DictConfig.

    Args:
        *configs: Config files or DictConfig objects to merge.

    Returns:
        Merged DictConfig.
    """
    merged = OmegaConf.create({})
    for cfg in configs:
        if isinstance(cfg, (str, Path)):
            cfg = load_config(cfg)
        merged = OmegaConf.merge(merged, cfg)
    return merged


def resolve_paths(cfg: DictConfig, root: Path) -> DictConfig:
    """Resolve relative paths in *data* section against a project root.

    Args:
        cfg: OmegaConf config containing a ``data`` section.
        root: Absolute project root directory.

    Returns:
        Config with data paths converted to absolute strings.
    """
    if "data" in cfg:
        for key in ("raw_dir", "interim_dir", "processed_dir", "dataset_yaml"):
            if key in cfg.data:
                cfg.data[key] = str(root / cfg.data[key])
    if "project" in cfg and "output_root" in cfg.project:
        cfg.project.output_root = str(root / cfg.project.output_root)
    return cfg
