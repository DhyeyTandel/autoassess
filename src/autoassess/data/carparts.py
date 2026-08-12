"""Download and remap the Carparts-Seg dataset into the project's part taxonomy.

Source: the Ultralytics Carparts-Seg dataset (23 classes, YOLOv8-seg polygon
format, 3833 images) — https://docs.ultralytics.com/datasets/segment/carparts-seg/.
Downloaded via Ultralytics' own dataset-resolution machinery
(`check_det_dataset`), the same mechanism `yolo train data=carparts-seg.yaml`
uses, so no bespoke download/unzip code is needed here (contrast with
`scripts/download_data.py` for CarDD, which has no such auto-download path
and requires a manual license-gated download).

The source dataset's 23 classes don't match this project's 6-panel taxonomy
(`configs/carparts.yaml` `names`) — several raw classes merge into one panel
(front_left_door + front_right_door + front_door -> door), and several have
no corresponding panel and are dropped (mirrors, wheel, trunk, tailgate,
tail lights, the "object" catch-all, and the rear window — see
`configs/carparts.yaml` for the full mapping and the reasoning per dropped
class). This module reads the raw dataset's label files, remaps class
indices through `panel_groups`, drops unmapped instances, and writes a
second YOLO-seg dataset at `data/processed/carparts/` in the same
`images/{split}` + `labels/{split}` layout `autoassess.data.convert`
produces for CarDD — so `train_parts.py` can point at it exactly like
`train_yolo.py` points at `data/processed/cardd/`.

Known gap: no "fender" panel exists in the source dataset, so this project
cannot segment or associate damage to fenders — see `configs/carparts.yaml`.

Usage
-----
    autoassess-carparts --config configs/base.yaml --dataset-config configs/carparts.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

ULTRALYTICS_DATASET_YAML = "carparts-seg.yaml"
SPLIT_NAMES = ["train", "val", "test"]


class RawCategoryMismatchError(Exception):
    """Raised when the downloaded Carparts-Seg dataset's classes don't match
    the `raw_names` this project's remapping was built against."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Carparts-Seg and remap it to this project's 6-panel taxonomy."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (data.raw_dir / processed_dir).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/carparts.yaml"),
                   help="Path to the parts dataset YAML (raw_names, panel_groups).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed (unused today — accepted for CLI "
                        "consistency with autoassess-convert; remapping is deterministic).")
    return p.parse_args()


def load_dataset_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    raw_names: dict[int, str] = OmegaConf.to_container(  # type: ignore[assignment]
        cfg.raw_names, resolve=True
    )
    panel_names: dict[int, str] = OmegaConf.to_container(  # type: ignore[assignment]
        cfg.names, resolve=True
    )
    panel_groups: dict[str, str] = OmegaConf.to_container(  # type: ignore[assignment]
        cfg.panel_groups, resolve=True
    )
    return {
        "raw_names_by_index": {i: raw_names[i] for i in sorted(raw_names, key=int)},
        "panel_names_ordered": [panel_names[i] for i in sorted(panel_names, key=int)],
        "panel_groups": panel_groups,
    }


def download_raw_dataset() -> Path:
    """Trigger Ultralytics' own dataset resolution/download for
    carparts-seg.yaml and return the absolute root it downloaded to.

    Idempotent: if already downloaded, `check_det_dataset` verifies the
    existing files and returns immediately without re-downloading.
    """
    from ultralytics.data.utils import check_det_dataset

    log.info("Resolving Carparts-Seg dataset via Ultralytics (auto-downloads if missing)...")
    data = check_det_dataset(ULTRALYTICS_DATASET_YAML)
    root = Path(data["path"])
    log.info("Carparts-Seg dataset resolved at: %s", root)
    return root


def assert_raw_categories_match(
    raw_root: Path, expected_raw_names_by_index: dict[int, str]
) -> None:
    """Validate the downloaded dataset's own class list matches what
    `configs/carparts.yaml`'s `panel_groups` mapping was built against.

    Carparts-Seg ships its class list in the accompanying
    `carparts-seg.yaml`, not embedded per-label-file, so — unlike CarDD's
    COCO categories — there's no per-instance category id to cross-check
    against; this checks the dataset-level class list instead. Fails loudly
    on any mismatch, the same policy as `autoassess.data.convert`, since a
    silently-shifted class list would remap every instance to the wrong panel.
    """
    yaml_path = raw_root / ULTRALYTICS_DATASET_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"Expected {yaml_path} next to the downloaded dataset.")

    actual_cfg = OmegaConf.load(yaml_path)
    actual_names: dict[int, str] = OmegaConf.to_container(  # type: ignore[assignment]
        actual_cfg.names, resolve=True
    )
    actual = {int(i): name for i, name in actual_names.items()}

    if actual != expected_raw_names_by_index:
        lines = ["Carparts-Seg raw class list mismatch vs configs/carparts.yaml raw_names:", ""]
        lines.append(f"{'':>3} {'EXPECTED (id, name)':<30} {'ACTUAL (id, name)':<30}")
        all_ids = sorted(set(actual) | set(expected_raw_names_by_index))
        for i in all_ids:
            exp = (i, expected_raw_names_by_index.get(i, "-"))
            act = (i, actual.get(i, "-"))
            marker = "!=" if exp != act else "=="
            lines.append(f"{marker:>3} {str(exp):<30} {str(act):<30}")
        message = "\n".join(lines)
        log.error(message)
        raise RawCategoryMismatchError(message)


def remap_label_file(
    label_path: Path,
    raw_names_by_index: dict[int, str],
    panel_groups: dict[str, str],
    panel_name_to_index: dict[str, int],
) -> tuple[list[str], int, int]:
    """Read one raw-taxonomy label file and return (remapped_lines,
    n_kept, n_dropped). Instances whose raw class has no entry in
    `panel_groups` are dropped, not remapped to an arbitrary panel."""
    kept_lines = []
    n_kept = 0
    n_dropped = 0

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        raw_class_index = int(parts[0])
        raw_name = raw_names_by_index.get(raw_class_index)
        panel_name = panel_groups.get(raw_name) if raw_name is not None else None

        if panel_name is None:
            n_dropped += 1
            continue

        panel_index = panel_name_to_index[panel_name]
        kept_lines.append(f"{panel_index} " + " ".join(parts[1:]))
        n_kept += 1

    return kept_lines, n_kept, n_dropped


def remap_split(
    raw_root: Path,
    split: str,
    raw_names_by_index: dict[int, str],
    panel_groups: dict[str, str],
    panel_name_to_index: dict[str, int],
    dest_dir: Path,
) -> dict[str, int]:
    src_images_dir = raw_root / "images" / split
    src_labels_dir = raw_root / "labels" / split
    dest_images_dir = dest_dir / "images" / split
    dest_labels_dir = dest_dir / "labels" / split
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_images_with_kept_instances = 0
    n_instances_kept = 0
    n_instances_dropped = 0

    for label_path in sorted(src_labels_dir.glob("*.txt")):
        stem = label_path.stem
        src_image_path = _find_image(src_images_dir, stem)
        if src_image_path is None:
            log.warning("No image found for label %s, skipping.", label_path)
            continue
        n_images += 1

        kept_lines, n_kept, n_dropped = remap_label_file(
            label_path, raw_names_by_index, panel_groups, panel_name_to_index
        )
        n_instances_kept += n_kept
        n_instances_dropped += n_dropped
        if n_kept > 0:
            n_images_with_kept_instances += 1

        dest_image_path = dest_images_dir / src_image_path.name
        if not dest_image_path.exists():
            try:
                dest_image_path.symlink_to(src_image_path.resolve())
            except OSError:
                shutil.copy2(src_image_path, dest_image_path)

        dest_label_path = dest_labels_dir / (stem + ".txt")
        dest_label_path.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8"
        )

    return {
        "images": n_images,
        "images_with_panels": n_images_with_kept_instances,
        "instances_kept": n_instances_kept,
        "instances_dropped": n_instances_dropped,
    }


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "carparts_data"
    setup_run_logging(run_dir)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)

    ds_cfg = load_dataset_config(args.dataset_config)
    log.info("Panel taxonomy (from %s): %s", args.dataset_config, ds_cfg["panel_names_ordered"])

    raw_root = download_raw_dataset()
    assert_raw_categories_match(raw_root, ds_cfg["raw_names_by_index"])
    log.info("Raw category check passed: 23 classes match configs/carparts.yaml raw_names exactly.")

    # copy the raw dataset into data/raw/ for provenance/reproducibility, mirroring
    # where CarDD's raw COCO archive lives, rather than only ever reading from
    # Ultralytics' global (outside-the-repo) datasets cache
    raw_dest = Path(cfg.data.raw_dir) / "carparts"
    if not raw_dest.exists():
        log.info("Copying raw dataset into %s for provenance...", raw_dest)
        shutil.copytree(raw_root, raw_dest)
    else:
        log.info("%s already exists; not re-copying.", raw_dest)

    panel_name_to_index = {name: i for i, name in enumerate(ds_cfg["panel_names_ordered"])}
    processed_dir = Path(cfg.data.processed_dir) / "carparts"

    totals: dict[str, int] = {}
    for split in SPLIT_NAMES:
        result = remap_split(
            raw_root, split,
            ds_cfg["raw_names_by_index"], ds_cfg["panel_groups"], panel_name_to_index,
            processed_dir,
        )
        log.info("Split '%s': %s", split, result)
        for k, v in result.items():
            totals[k] = totals.get(k, 0) + v

    log.info("Remap complete. Totals across all splits: %s", totals)
    log.info("Parts dataset root: %s", processed_dir)


if __name__ == "__main__":
    main()
