"""Convert VehiDE VIA-JSON polygon annotations to YOLOv8 segmentation format.

VehiDE ships damage regions as VIA (VGG Image Annotator) JSON — a flat
{image_file_name: {"regions": [{"all_x": [...], "all_y": [...], "class": str}]}}
map — not COCO, so this is a separate converter from convert.py rather than
a shared code path. Region `class` strings are the original Vietnamese
labels (e.g. "mop_lom" for dent); every label encountered is asserted
against `vietnamese_labels` in configs/vehide.yaml before conversion, and
conversion fails loudly on anything unrecognised rather than silently
dropping instances.

For every image, writes a ``.txt`` label file with one line per instance:

    <class_index> x1 y1 x2 y2 ... xn yn

where ``class_index`` is 0-based (looked up via the Vietnamese label ->
`names` order in configs/vehide.yaml) and coordinates are normalised to
[0, 1] by image width/height, in the polygon's original point order.

VehiDE ships an official train/val split (0Train_via_annos.json /
0Val_via_annos.json) but no test split — a seeded slice is carved out of
train for that, since the fixed val set is left untouched for comparability
with the dataset's own reported numbers.

Usage
-----
    autoassess-convert-vehide --config configs/base.yaml --dataset-config configs/vehide.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

TEST_SPLIT_FRACTION = 0.15  # carved out of the official train split, seeded


@dataclass
class ViaImage:
    file_name: str
    width: int
    height: int
    # list of (class_index, flat normalised polygon coords) per instance
    instances: list[tuple[int, list[float]]] = field(default_factory=list)


class UnknownLabelError(Exception):
    """Raised when a VIA region's `class` string isn't in vietnamese_labels."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert VehiDE VIA-JSON polygons to YOLOv8-seg label format."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (data.raw_dir / processed_dir).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/vehide.yaml"),
                   help="Path to the Ultralytics dataset YAML defining canonical class order.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed for the train/test carve-out (default: config seed).")
    return p.parse_args()


def load_label_map(dataset_config_path: Path) -> dict[str, int]:
    """Return {vietnamese_label: yolo_class_index}, both taken from
    configs/vehide.yaml so the mapping lives in one auditable place."""
    ds_cfg = OmegaConf.load(dataset_config_path)
    names: dict[int, str] = OmegaConf.to_container(ds_cfg.names, resolve=True)  # type: ignore[assignment]
    vi_labels: dict[int, str] = OmegaConf.to_container(  # type: ignore[assignment]
        ds_cfg.vietnamese_labels, resolve=True
    )
    if set(names) != set(vi_labels):
        raise UnknownLabelError(
            f"configs/vehide.yaml names {sorted(names)} and vietnamese_labels "
            f"{sorted(vi_labels)} must have identical class indices."
        )
    return {vi_labels[idx]: idx for idx in names}


def load_via_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def normalise_polygon(xs: list[float], ys: list[float], width: int, height: int) -> list[float]:
    """Interleave x/y into a flat [x1, y1, ...] polygon, normalised to [0, 1]
    by image size and clamped for points that sit on/past the image border."""
    normed = []
    for x, y in zip(xs, ys, strict=True):
        normed.append(min(max(x / width, 0.0), 1.0))
        normed.append(min(max(y / height, 0.0), 1.0))
    return normed


def build_images(
    via: dict[str, Any], image_dir: Path, label_to_class_index: dict[str, int]
) -> list[ViaImage]:
    images: list[ViaImage] = []
    dropped_missing_file = 0
    dropped_degenerate = 0
    unknown_labels: set[str] = set()

    for file_name, entry in via.items():
        img_path = image_dir / file_name
        if not img_path.exists():
            dropped_missing_file += 1
            continue
        with Image.open(img_path) as im:
            width, height = im.size

        via_image = ViaImage(file_name=file_name, width=width, height=height)
        for region in entry.get("regions", []):
            label = region.get("class", "")
            if label not in label_to_class_index:
                unknown_labels.add(label)
                continue
            xs, ys = region.get("all_x", []), region.get("all_y", [])
            if len(xs) < 3 or len(xs) != len(ys):
                dropped_degenerate += 1
                continue
            class_index = label_to_class_index[label]
            normed = normalise_polygon(xs, ys, width, height)
            via_image.instances.append((class_index, normed))
        images.append(via_image)

    if unknown_labels:
        raise UnknownLabelError(
            f"Found VIA region class(es) not in configs/vehide.yaml vietnamese_labels: "
            f"{sorted(unknown_labels)}. Add them to the mapping (with a translated "
            "English name in `names`) before converting."
        )
    if dropped_missing_file:
        log.warning("Skipped %d image(s) referenced in annotations but not found on disk.",
                    dropped_missing_file)
    if dropped_degenerate:
        log.warning("Dropped %d degenerate region(s) (<3 points or mismatched x/y).",
                    dropped_degenerate)

    return images


def write_yolo_labels(
    images: list[ViaImage], src_image_dir: Path, dest_images_dir: Path, dest_labels_dir: Path
) -> None:
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        src_img_path = src_image_dir / image.file_name
        dest_img_path = dest_images_dir / image.file_name
        if not dest_img_path.exists():
            try:
                dest_img_path.symlink_to(src_img_path.resolve())
            except OSError:
                import shutil as _shutil
                _shutil.copy2(src_img_path, dest_img_path)

        label_path = dest_labels_dir / (Path(image.file_name).stem + ".txt")
        lines = []
        for class_index, poly in image.instances:
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{class_index} {coords}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def find_image_dir(vehide_dir: Path, annotation_split: str) -> Path:
    """VehiDE's archive layout keeps train and val images in two entirely
    separate top-level directories (image/image/ for the 0Train annotations,
    validation/validation/ for 0Val) — they are not interchangeable, and a
    single shared image_dir would silently resolve every val image to a
    missing path (build_images logs and skips missing files rather than
    erroring, so this would quietly drop the whole val split instead of
    failing loudly)."""
    candidates = {
        "train": [vehide_dir / "image" / "image", vehide_dir / "image"],
        "val": [vehide_dir / "validation" / "validation", vehide_dir / "validation"],
    }[annotation_split]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No VehiDE '{annotation_split}' image directory found under {vehide_dir} "
        f"(checked {candidates}). Run scripts/download_vehide.py first."
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "convert_vehide_data"
    setup_run_logging(run_dir)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)

    label_to_class_index = load_label_map(args.dataset_config)
    log.info("Label map (vietnamese -> class index) from %s: %s",
              args.dataset_config, label_to_class_index)

    raw_dir = Path(cfg.data.raw_dir)
    vehide_dir = raw_dir / "vehide"
    train_image_dir = find_image_dir(vehide_dir, "train")
    val_image_dir = find_image_dir(vehide_dir, "val")
    processed_dir = Path(cfg.data.processed_dir) / "vehide"

    train_via = load_via_json(vehide_dir / "0Train_via_annos.json")
    val_via = load_via_json(vehide_dir / "0Val_via_annos.json")

    train_images = build_images(train_via, train_image_dir, label_to_class_index)
    val_images = build_images(val_via, val_image_dir, label_to_class_index)

    # VehiDE ships no test split — carve a seeded slice out of train so the
    # official val set stays untouched for comparability with published numbers.
    # The carved-out test images still physically live under train_image_dir.
    rng = random.Random(seed)
    rng.shuffle(train_images)
    n_test = int(len(train_images) * TEST_SPLIT_FRACTION)
    test_images, train_images = train_images[:n_test], train_images[n_test:]

    counts = {}
    for split, images, src_image_dir in [
        ("train", train_images, train_image_dir),
        ("val", val_images, val_image_dir),
        ("test", test_images, train_image_dir),
    ]:
        write_yolo_labels(
            images, src_image_dir,
            processed_dir / "images" / split, processed_dir / "labels" / split,
        )
        counts[split] = len(images)

    log.info("Conversion complete. Images per split: %s", counts)
    log.info("YOLO dataset root: %s", processed_dir)


if __name__ == "__main__":
    main()
