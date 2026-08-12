"""Convert CarDD COCO polygon annotations to YOLOv8 segmentation format.

For every image, writes a ``.txt`` label file with one line per instance:

    <class_index> x1 y1 x2 y2 ... xn yn

where ``class_index`` is 0-based and coordinates are normalised to [0, 1]
by image width/height, in the polygon's original point order. Class index
is always looked up from the ordered ``names`` list in ``configs/cardd.yaml``
— never from the order categories happen to appear in the source COCO JSON,
since COCO does not guarantee that ordering matches category id order.

If the source COCO annotations already define train/val/test splits (i.e.
``instances_{train,val,test}2017.json`` all exist), that official split is
preserved as-is. Otherwise a single combined annotation set is split
70/15/15 using a seeded, stratified-on-multi-label-presence assignment, so
that rare classes are not left out of val/test by chance.

Usage
-----
    autoassess-convert --config configs/base.yaml --dataset-config configs/cardd.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

SPLIT_NAMES = ["train", "val", "test"]
OFFICIAL_SPLIT_FILES = {"train": "train2017", "val": "val2017", "test": "test2017"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int
    # list of (class_index, flat normalised polygon coords) per instance
    instances: list[tuple[int, list[float]]] = field(default_factory=list)


class CategoryMismatchError(Exception):
    """Raised when the source COCO categories don't match configs/cardd.yaml."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CarDD COCO polygons to YOLOv8-seg label format."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (data.raw_dir / processed_dir).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Path to the Ultralytics dataset YAML defining canonical class order.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed for the stratified split (default: config seed).")
    return p.parse_args()


def load_expected_categories(dataset_config_path: Path) -> list[tuple[int, str]]:
    """Return the expected [(coco_category_id, name), ...] in canonical order.

    Order is taken from ``names`` (YOLO class index order), with COCO ids
    resolved via ``coco_category_ids`` in the same file.
    """
    ds_cfg = OmegaConf.load(dataset_config_path)
    names: dict[int, str] = OmegaConf.to_container(ds_cfg.names, resolve=True)  # type: ignore[assignment]
    coco_ids: dict[str, int] = OmegaConf.to_container(  # type: ignore[assignment]
        ds_cfg.coco_category_ids, resolve=True
    )

    ordered_names = [names[i] for i in sorted(names, key=int)]
    return [(int(coco_ids[name]), name) for name in ordered_names]


def assert_categories_match(
    coco_categories: list[dict[str, Any]], expected: list[tuple[int, str]]
) -> dict[int, int]:
    """Validate COCO categories against the expected ordered list.

    Checks name, count, and id ordering all at once (not just set equality),
    since a silently-reordered category id would misassign every class in
    the converted labels. Fails loudly with a side-by-side listing on any
    mismatch.

    Returns:
        Mapping from COCO category id -> YOLO class index (0-based, in
        expected-list order) if validation passes.
    """
    actual = sorted(
        ((c["id"], c["name"]) for c in coco_categories), key=lambda t: t[0]
    )
    expected_sorted = sorted(expected, key=lambda t: t[0])

    if actual != expected_sorted:
        lines = ["Category mismatch between COCO annotations and configs/cardd.yaml:", ""]
        lines.append(f"{'':>3} {'EXPECTED (id, name)':<35} {'ACTUAL (id, name)':<35}")
        max_len = max(len(actual), len(expected_sorted))
        for i in range(max_len):
            exp = expected_sorted[i] if i < len(expected_sorted) else ("-", "-")
            act = actual[i] if i < len(actual) else ("-", "-")
            marker = "!=" if exp != act else "=="
            lines.append(f"{marker:>3} {str(exp):<35} {str(act):<35}")
        lines.append("")
        lines.append(f"Expected {len(expected_sorted)} categories, found {len(actual)}.")
        message = "\n".join(lines)
        log.error(message)
        raise CategoryMismatchError(message)

    # expected is already in YOLO class-index order (index 0 == class 0, ...)
    coco_id_to_class_index = {coco_id: idx for idx, (coco_id, _name) in enumerate(expected)}
    return coco_id_to_class_index


def polygon_from_segmentation(seg: list[list[float]] | dict[str, Any] | None) -> list[float] | None:
    """Extract a single flat [x1, y1, x2, y2, ...] polygon from a COCO
    `segmentation` field, taking the largest ring if multiple are present
    (holes / disjoint parts are dropped — YOLO-seg labels support only one
    polygon per instance line).
    """
    if seg is None or isinstance(seg, dict):
        # RLE mask (or missing), not polygon — cannot be converted to a YOLO-seg polygon line.
        return None
    if not seg:
        return None
    ring = max(seg, key=len) if len(seg) > 1 else seg[0]
    if len(ring) < 6:  # fewer than 3 points
        return None
    return list(ring)


def normalise_polygon(poly: list[float], width: int, height: int) -> list[float]:
    """Normalise a flat [x1, y1, x2, y2, ...] polygon to [0, 1] by image
    size, clamping to handle annotation coordinates that sit exactly on or
    fractionally past the image border. Point order is preserved exactly."""
    normed = []
    for i, v in enumerate(poly):
        size = width if i % 2 == 0 else height
        normed.append(min(max(v / size, 0.0), 1.0))
    return normed


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def find_coco_root(cardd_dir: Path) -> Path:
    if (cardd_dir / "annotations").is_dir():
        return cardd_dir
    for candidate in cardd_dir.rglob("annotations"):
        if candidate.is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"No COCO 'annotations/' directory found under {cardd_dir}. "
        "Run scripts/download_data.py first."
    )


def find_image_dir(coco_root: Path, split_file_stem: str) -> Path:
    flat = coco_root / split_file_stem
    nested = coco_root / "images" / split_file_stem
    if flat.is_dir():
        return flat
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"No image directory found for split '{split_file_stem}' under {coco_root}"
    )


def build_images(
    coco: dict[str, Any], coco_id_to_class_index: dict[int, int]
) -> dict[int, CocoImage]:
    images: dict[int, CocoImage] = {}
    for img in coco["images"]:
        images[img["id"]] = CocoImage(
            image_id=img["id"], file_name=img["file_name"], width=img["width"], height=img["height"]
        )

    dropped_rle = 0
    dropped_degenerate = 0
    for ann in coco["annotations"]:
        image = images.get(ann["image_id"])
        if image is None:
            continue
        poly = polygon_from_segmentation(ann.get("segmentation"))
        if poly is None:
            if isinstance(ann.get("segmentation"), dict):
                dropped_rle += 1
            else:
                dropped_degenerate += 1
            continue
        class_index = coco_id_to_class_index[ann["category_id"]]
        normed = normalise_polygon(poly, image.width, image.height)
        image.instances.append((class_index, normed))

    if dropped_rle:
        log.warning("Dropped %d instance(s) with RLE (non-polygon) segmentation.", dropped_rle)
    if dropped_degenerate:
        log.warning(
            "Dropped %d instance(s) with degenerate polygons (<3 points).", dropped_degenerate
        )

    return images


def official_splits_available(coco_root: Path) -> bool:
    ann_dir = coco_root / "annotations"
    return all(
        (ann_dir / f"instances_{file_stem}.json").exists()
        for file_stem in OFFICIAL_SPLIT_FILES.values()
    )


def stratified_split(
    images: list[CocoImage], num_classes: int, seed: int
) -> dict[str, list[CocoImage]]:
    """Greedy multi-label stratified split, seeded.

    Standard iterative-stratification approach (Sechidis et al.): repeatedly
    pick the (remaining) sample with the rarest label in its label set, and
    assign it to whichever split most under-represents that label relative
    to its target ratio. Falls back to random assignment by target ratio for
    images with no instances at all.
    """
    rng = random.Random(seed)

    label_sets: dict[int, set[int]] = {}
    with_labels: list[CocoImage] = []
    without_labels: list[CocoImage] = []
    for image in images:
        labels = {cls for cls, _poly in image.instances}
        if labels:
            label_sets[image.image_id] = labels
            with_labels.append(image)
        else:
            without_labels.append(image)

    # per-class total counts, used to find "rarest label in this sample"
    class_totals = [0] * num_classes
    for labels in label_sets.values():
        for c in labels:
            class_totals[c] += 1

    target_counts = {
        split: {c: SPLIT_RATIOS[split] * class_totals[c] for c in range(num_classes)}
        for split in SPLIT_NAMES
    }
    current_counts = {split: [0] * num_classes for split in SPLIT_NAMES}
    split_totals = dict.fromkeys(SPLIT_NAMES, 0)
    assigned: dict[str, list[CocoImage]] = {split: [] for split in SPLIT_NAMES}

    remaining = list(with_labels)
    rng.shuffle(remaining)

    while remaining:
        # pick the sample containing the globally rarest remaining label
        remaining_class_counts = [0] * num_classes
        for image in remaining:
            for c in label_sets[image.image_id]:
                remaining_class_counts[c] += 1
        nonzero = [c for c in range(num_classes) if remaining_class_counts[c] > 0]
        rarest_class = min(nonzero, key=lambda c: remaining_class_counts[c])

        candidates = [
            image for image in remaining if rarest_class in label_sets[image.image_id]
        ]
        image = rng.choice(candidates)
        remaining.remove(image)

        labels = label_sets[image.image_id]
        # assign to the split with the largest positive deficit (target - current)
        # for the rarest label, tie-broken by overall split size deficit
        def deficit(split: str) -> tuple[float, float]:
            label_deficit = target_counts[split][rarest_class] - current_counts[split][rarest_class]
            size_deficit = SPLIT_RATIOS[split] * len(images) - split_totals[split]
            return (label_deficit, size_deficit)

        best_split = max(SPLIT_NAMES, key=deficit)
        assigned[best_split].append(image)
        split_totals[best_split] += 1
        for c in labels:
            current_counts[best_split][c] += 1

    # distribute unlabelled images purely by target split size
    rng.shuffle(without_labels)
    for image in without_labels:
        def size_deficit(split: str) -> float:
            return SPLIT_RATIOS[split] * len(images) - split_totals[split]

        best_split = max(SPLIT_NAMES, key=size_deficit)
        assigned[best_split].append(image)
        split_totals[best_split] += 1

    for split in SPLIT_NAMES:
        rng.shuffle(assigned[split])

    return assigned


def write_yolo_labels(
    images: list[CocoImage],
    src_image_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
) -> None:
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        src_img_path = src_image_dir / image.file_name
        if not src_img_path.exists():
            log.warning("Image file missing, skipping: %s", src_img_path)
            continue

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


def convert_official_splits(
    coco_root: Path, coco_id_to_class_index: dict[int, int], processed_dir: Path
) -> dict[str, int]:
    counts = {}
    for split, file_stem in OFFICIAL_SPLIT_FILES.items():
        coco = load_coco(coco_root / "annotations" / f"instances_{file_stem}.json")
        images_by_id = build_images(coco, coco_id_to_class_index)
        src_image_dir = find_image_dir(coco_root, file_stem)
        write_yolo_labels(
            list(images_by_id.values()),
            src_image_dir,
            processed_dir / "images" / split,
            processed_dir / "labels" / split,
        )
        counts[split] = len(images_by_id)
    return counts


def convert_with_stratified_split(
    coco_root: Path,
    coco_id_to_class_index: dict[int, int],
    processed_dir: Path,
    num_classes: int,
    seed: int,
) -> dict[str, int]:
    # combine every split file found, since there is no official split to preserve
    all_images: dict[int, CocoImage] = {}
    file_stem_by_image_id: dict[int, str] = {}
    ann_dir = coco_root / "annotations"
    for ann_file in sorted(ann_dir.glob("*.json")):
        coco = load_coco(ann_file)
        images_by_id = build_images(coco, coco_id_to_class_index)
        file_stem = ann_file.stem.removeprefix("instances_")
        for image_id, image in images_by_id.items():
            all_images[image_id] = image
            file_stem_by_image_id[image_id] = file_stem

    assigned = stratified_split(list(all_images.values()), num_classes, seed)

    counts = {}
    for split, split_images in assigned.items():
        by_stem: dict[str, list[CocoImage]] = defaultdict(list)
        for image in split_images:
            by_stem[file_stem_by_image_id[image.image_id]].append(image)
        for file_stem, images_in_stem in by_stem.items():
            src_image_dir = find_image_dir(coco_root, file_stem)
            write_yolo_labels(
                images_in_stem,
                src_image_dir,
                processed_dir / "images" / split,
                processed_dir / "labels" / split,
            )
        counts[split] = len(split_images)
    return counts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "convert_data"
    setup_run_logging(run_dir)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)

    expected_categories = load_expected_categories(args.dataset_config)
    log.info("Expected categories (from %s): %s", args.dataset_config, expected_categories)

    raw_dir = Path(cfg.data.raw_dir)
    cardd_dir = raw_dir / "cardd"
    coco_root = find_coco_root(cardd_dir)

    # validate against whichever annotation file is available first
    ann_dir = coco_root / "annotations"
    probe_file = next(iter(sorted(ann_dir.glob("*.json"))), None)
    if probe_file is None:
        raise FileNotFoundError(f"No COCO annotation files found under {ann_dir}")
    probe_coco = load_coco(probe_file)
    coco_id_to_class_index = assert_categories_match(probe_coco["categories"], expected_categories)
    log.info("Category check passed: %d classes match configs/cardd.yaml exactly.",
              len(expected_categories))

    processed_dir = Path(cfg.data.processed_dir) / "cardd"

    if official_splits_available(coco_root):
        log.info("Official train/val/test split found — preserving it.")
        counts = convert_official_splits(coco_root, coco_id_to_class_index, processed_dir)
    else:
        log.info("No official split found — creating seeded 70/15/15 stratified split (seed=%d).",
                  seed)
        counts = convert_with_stratified_split(
            coco_root, coco_id_to_class_index, processed_dir, len(expected_categories), seed
        )

    log.info("Conversion complete. Images per split: %s", counts)
    log.info("YOLO dataset root: %s", processed_dir)


if __name__ == "__main__":
    main()
