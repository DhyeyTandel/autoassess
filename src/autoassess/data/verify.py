"""Reverse-render converted YOLO-seg labels back onto their images.

Sanity check for src/autoassess/data/convert.py: draws each label's
polygons (denormalised back to pixel space) over the source image so
mistakes in polygon ordering or coordinate normalisation are visible by
eye, not just by passing shape checks.

Sampling is NOT uniform over the label set. Rare classes (as listed under
`rare_classes` in configs/cardd.yaml — tire flat, lamp broken, glass
shatter) are heavily under-represented, so a uniform random sample of a
handful of images would almost certainly show nothing but dent/scratch and
say nothing about whether small polygons survived conversion. Instead this
script guarantees at least `--min-per-rare-class` images containing each
rare class, then fills the remainder of the sample uniformly at random.

Usage
-----
    autoassess-verify --config configs/base.yaml --dataset-config configs/cardd.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

SAMPLE_SIZE = 20
MIN_PER_RARE_CLASS = 2
BOX_COLOR_PALETTE = [
    (66, 135, 245),   # blue
    (245, 130, 66),   # orange
    (66, 245, 141),   # green
    (245, 66, 218),   # magenta
    (245, 222, 66),   # yellow
    (147, 66, 245),   # purple
]


@dataclass
class LabelInstance:
    class_index: int
    polygon: list[float]  # flat normalised [x1, y1, x2, y2, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reverse-render converted YOLO-seg labels onto images for visual QA."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (data.processed_dir).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Path to the Ultralytics dataset YAML (class names, rare_classes).")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Which converted split to sample from.")
    p.add_argument("--n", type=int, default=SAMPLE_SIZE,
                   help="Total number of images to reverse-render.")
    p.add_argument("--min-per-rare-class", type=int, default=MIN_PER_RARE_CLASS,
                   help="Minimum images containing each rare class in the sample.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed for sampling (default: config seed).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (default: reports/figures/label_check/).")
    return p.parse_args()


def load_class_names(dataset_config_path: Path) -> list[str]:
    ds_cfg = OmegaConf.load(dataset_config_path)
    names: dict[int, str] = OmegaConf.to_container(ds_cfg.names, resolve=True)  # type: ignore[assignment]
    return [names[i] for i in sorted(names, key=int)]


def load_rare_class_indices(dataset_config_path: Path, class_names: list[str]) -> list[int]:
    ds_cfg = OmegaConf.load(dataset_config_path)
    rare_names: list[str] = OmegaConf.to_container(  # type: ignore[assignment]
        ds_cfg.rare_classes, resolve=True
    )
    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in rare_names if name not in name_to_index]
    if missing:
        raise ValueError(f"rare_classes in {dataset_config_path} not found in names: {missing}")
    return [name_to_index[name] for name in rare_names]


def parse_label_file(label_path: Path) -> list[LabelInstance]:
    instances: list[LabelInstance] = []
    if not label_path.exists():
        return instances
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_index = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        instances.append(LabelInstance(class_index=class_index, polygon=coords))
    return instances


def index_labels_by_class(
    images_dir: Path, labels_dir: Path
) -> tuple[dict[str, list[LabelInstance]], dict[int, list[str]]]:
    """Return (stem -> instances) and (class_index -> [stems containing that class])."""
    labels_by_stem: dict[str, list[LabelInstance]] = {}
    stems_by_class: dict[int, list[str]] = {}

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        instances = parse_label_file(label_path)
        if not instances:
            continue
        labels_by_stem[stem] = instances
        for cls in {inst.class_index for inst in instances}:
            stems_by_class.setdefault(cls, []).append(stem)

    return labels_by_stem, stems_by_class


def build_sample(
    labels_by_stem: dict[str, list[LabelInstance]],
    stems_by_class: dict[int, list[str]],
    rare_class_indices: list[int],
    n_total: int,
    min_per_rare_class: int,
    rng: random.Random,
) -> list[str]:
    """Guarantee >= min_per_rare_class images per rare class, then fill the
    rest of the sample uniformly at random from whatever remains."""
    selected: list[str] = []
    selected_set: set[str] = set()

    for cls in rare_class_indices:
        candidates = stems_by_class.get(cls, [])
        if not candidates:
            log.warning("No converted labels contain rare class index %d — "
                        "cannot include it in the sample.", cls)
            continue
        rng.shuffle(candidates)
        picked = 0
        for stem in candidates:
            if stem in selected_set:
                continue
            selected.append(stem)
            selected_set.add(stem)
            picked += 1
            if picked >= min_per_rare_class:
                break
        if picked < min_per_rare_class:
            log.warning("Only found %d/%d images for rare class index %d.",
                        picked, min_per_rare_class, cls)

    remaining_pool = [stem for stem in labels_by_stem if stem not in selected_set]
    rng.shuffle(remaining_pool)
    n_fill = max(n_total - len(selected), 0)
    selected.extend(remaining_pool[:n_fill])

    rng.shuffle(selected)
    return selected[:n_total] if len(selected) > n_total else selected


def denormalise_polygon(poly: list[float], width: int, height: int) -> np.ndarray:
    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts.astype(np.int32)


def render_label(
    image_path: Path,
    instances: list[LabelInstance],
    class_names: list[str],
    out_path: Path,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read image, skipping render: %s", image_path)
        return
    height, width = img.shape[:2]

    for inst in instances:
        color = BOX_COLOR_PALETTE[inst.class_index % len(BOX_COLOR_PALETTE)]
        pts = denormalise_polygon(inst.polygon, width, height)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        label_text = class_names[inst.class_index] if inst.class_index < len(class_names) \
            else f"cls{inst.class_index}"
        anchor = tuple(pts[0])
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img, (anchor[0], anchor[1] - th - 6), (anchor[0] + tw + 4, anchor[1]), color, -1
        )
        cv2.putText(
            img, label_text, (anchor[0] + 2, anchor[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
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

    run_dir = Path(cfg.project.output_root) / "verify_labels"
    setup_run_logging(run_dir)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)
    rng = random.Random(seed)

    class_names = load_class_names(args.dataset_config)
    rare_class_indices = load_rare_class_indices(args.dataset_config, class_names)
    log.info("Rare classes: %s (indices %s)",
             [class_names[i] for i in rare_class_indices], rare_class_indices)

    processed_dir = Path(cfg.data.processed_dir) / "cardd"
    images_dir = processed_dir / "images" / args.split
    labels_dir = processed_dir / "labels" / args.split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Converted split not found at {images_dir} / {labels_dir}. "
            "Run autoassess-convert first."
        )

    labels_by_stem, stems_by_class = index_labels_by_class(images_dir, labels_dir)
    log.info("Indexed %d labelled images in split '%s'.", len(labels_by_stem), args.split)

    sample_stems = build_sample(
        labels_by_stem, stems_by_class, rare_class_indices, args.n, args.min_per_rare_class, rng
    )
    log.info("Selected %d images for reverse-render (n=%d requested).", len(sample_stems), args.n)

    out_dir = args.out_dir or (project_root / "reports" / "figures" / "label_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for stem in sample_stems:
        image_path = find_image_for_stem(images_dir, stem)
        if image_path is None:
            log.warning("No image file found for label stem '%s', skipping.", stem)
            continue
        render_label(image_path, labels_by_stem[stem], class_names, out_dir / f"{stem}.jpg")
        rendered += 1

    log.info("Reverse-rendered %d/%d sampled images -> %s", rendered, len(sample_stems), out_dir)


if __name__ == "__main__":
    main()
