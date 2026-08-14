#!/usr/bin/env python3
"""scripts/download_vehide.py — fetch and verify the VehiDE dataset via Kaggle.

VehiDE (Huynh et al., 2023) ships 13,945 images across 7 damage classes in
VIA (VGG Image Annotator) polygon JSON — see convert_vehide.py for the
Vietnamese label -> English class mapping and the COCO-format converter's
counterpart notes. Unlike CarDD, VehiDE has a stable Kaggle-hosted release,
so this script uses the `kaggle` CLI/API rather than a manual zip drop.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings
-> API -> Create New Token) with read access — this script does not create
or manage that credential.

Usage
-----
    python scripts/download_vehide.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.utils.config import load_config, resolve_paths  # noqa: E402
from autoassess.utils.logging import get_logger, setup_run_logging  # noqa: E402

log = get_logger(__name__)

KAGGLE_DATASET_REF = "hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection"
EXPECTED_IMAGE_COUNT = 13945
ANNOTATION_FILES = ["0Train_via_annos.json", "0Val_via_annos.json"]


@dataclass
class SplitSummary:
    split: str
    annotation_file: Path
    image_count: int
    region_count: int
    labels_found: set[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and/or verify the VehiDE dataset into data/raw/vehide/ via Kaggle."
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to base YAML config (used to resolve data/raw_dir).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download/unpack even if data/raw/vehide/ already looks populated.",
    )
    return p.parse_args()


def already_populated(vehide_dir: Path) -> bool:
    # Train and val images ship in two separate top-level directories
    # (image/image/ and validation/validation/) — both must be present, not
    # just one, or a re-run would wrongly skip re-unpacking a partial extract.
    has_train_dir = (vehide_dir / "image" / "image").is_dir() or (vehide_dir / "image").is_dir()
    has_val_dir = (
        (vehide_dir / "validation" / "validation").is_dir()
        or (vehide_dir / "validation").is_dir()
    )
    return all((vehide_dir / f).exists() for f in ANNOTATION_FILES) and has_train_dir and has_val_dir


def download_and_unpack(vehide_dir: Path) -> None:
    try:
        import kaggle  # noqa: F401
    except OSError as exc:
        raise RuntimeError(
            "Kaggle API credentials not found. Create an API token at "
            "kaggle.com -> Settings -> API -> Create New Token, and place the "
            "downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600)."
        ) from exc

    from kaggle.api.kaggle_api_extended import KaggleApi

    vehide_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Kaggle dataset %s -> %s", KAGGLE_DATASET_REF, vehide_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET_REF, path=str(vehide_dir), unzip=False, quiet=False)

    zip_path = vehide_dir / (KAGGLE_DATASET_REF.split("/")[-1] + ".zip")
    if not zip_path.exists():
        found = sorted(vehide_dir.glob("*.zip"))
        if not found:
            raise FileNotFoundError(f"Expected a downloaded .zip under {vehide_dir}, found none.")
        zip_path = found[0]

    log.info("Unpacking %s -> %s", zip_path, vehide_dir)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member in archive: {bad}")
        zf.extractall(vehide_dir)
    log.info("Unpack complete.")


def find_image_dirs(vehide_dir: Path) -> dict[str, Path | None]:
    """Train and val images ship in two separate directories (image/image/,
    validation/validation/) — never a single shared one, see
    convert_vehide.find_image_dir's docstring for why that distinction
    matters at conversion time."""
    train_dir = vehide_dir / "image" / "image"
    if not train_dir.is_dir():
        train_dir = vehide_dir / "image"
    val_dir = vehide_dir / "validation" / "validation"
    if not val_dir.is_dir():
        val_dir = vehide_dir / "validation"
    return {
        "train": train_dir if train_dir.is_dir() else None,
        "val": val_dir if val_dir.is_dir() else None,
    }


def summarize_split(vehide_dir: Path, annotation_file: str) -> SplitSummary:
    ann_path = vehide_dir / annotation_file
    with ann_path.open("r", encoding="utf-8") as f:
        via = json.load(f)

    region_count = 0
    labels_found: set[str] = set()
    for entry in via.values():
        for region in entry.get("regions", []):
            region_count += 1
            labels_found.add(region.get("class", "?"))

    return SplitSummary(
        split=annotation_file.removeprefix("0").removesuffix("_via_annos.json"),
        annotation_file=ann_path,
        image_count=len(via),
        region_count=region_count,
        labels_found=labels_found,
    )


def print_summary(
    vehide_dir: Path, image_dirs: dict[str, Path | None], summaries: list[SplitSummary]
) -> None:
    print("\n" + "=" * 70)
    print("VehiDE dataset summary")
    print("=" * 70)
    print(f"Dataset root:   {vehide_dir}")
    print(f"Train image dir: {image_dirs['train'] if image_dirs['train'] else '(not found)'}")
    print(f"Val image dir:   {image_dirs['val'] if image_dirs['val'] else '(not found)'}")
    print("-" * 70)

    total_images = 0
    total_regions = 0
    all_labels: set[str] = set()
    for s in summaries:
        print(f"[OK] split={s.split:<6} images={s.image_count:<6} regions={s.region_count:<6}")
        print(f"     annotations: {s.annotation_file}")
        total_images += s.image_count
        total_regions += s.region_count
        all_labels.update(s.labels_found)

    print("-" * 70)
    print(f"Total annotated images: {total_images} (expected ~{EXPECTED_IMAGE_COUNT} total incl. unannotated)")
    print(f"Total regions:          {total_regions}")
    print(f"Raw Vietnamese labels found: {sorted(all_labels)}")
    print("(cross-check these against vietnamese_labels in configs/vehide.yaml "
          "before running autoassess-convert-vehide)")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "download_vehide"
    setup_run_logging(run_dir)

    raw_dir = Path(cfg.data.raw_dir)
    vehide_dir = raw_dir / "vehide"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if already_populated(vehide_dir) and not args.force:
        log.info("data/raw/vehide already looks populated; skipping download "
                 "(use --force to redo).")
    else:
        download_and_unpack(vehide_dir)

    image_dirs = find_image_dirs(vehide_dir)
    summaries = [
        summarize_split(vehide_dir, f) for f in ANNOTATION_FILES
        if (vehide_dir / f).exists()
    ]
    print_summary(vehide_dir, image_dirs, summaries)


if __name__ == "__main__":
    main()
