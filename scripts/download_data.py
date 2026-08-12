#!/usr/bin/env python3
"""scripts/download_data.py — fetch and verify the CarDD dataset.

CarDD (Car Damage Detection) ships 4,000 images across 6 damage classes
(dent, scratch, crack, glass shatter, lamp broken, tire flat) in COCO
format, released as a single ``CarDD_release.zip`` gated behind a license
form on the official project page (no stable direct-download URL exists).

Two supported workflows:

1. **Automatic**: if ``--url`` is given (e.g. a personal/institutional
   mirror you are entitled to use), the archive is downloaded and unpacked.
2. **Manual**: download ``CarDD_release.zip`` yourself from
   https://cardd-ustc.github.io/ and drop it in ``data/raw/`` (or pass
   ``--zip-path``). Re-running this script will detect it, unpack it, and
   verify it — no re-download.

Either way, the script always ends by printing a checksum/count summary
of what landed in ``data/raw/cardd/``.

Usage
-----
    python scripts/download_data.py --config configs/base.yaml
    python scripts/download_data.py --config configs/base.yaml --zip-path data/raw/CarDD_release.zip
    python scripts/download_data.py --config configs/base.yaml --url https://example.com/CarDD_release.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.utils.config import load_config, resolve_paths  # noqa: E402
from autoassess.utils.logging import get_logger, setup_run_logging  # noqa: E402

log = get_logger(__name__)

CARDD_HOMEPAGE = "https://cardd-ustc.github.io/"
DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
EXPECTED_IMAGE_COUNT = 4000
EXPECTED_SPLITS = ["train2017", "val2017", "test2017"]
CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass
class SplitSummary:
    split: str
    annotation_file: Path | None
    image_dir: Path | None
    image_count: int
    annotated_image_count: int
    instance_count: int
    category_names: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and/or verify the CarDD dataset into data/raw/cardd/."
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to base YAML config (used to resolve data/raw_dir).",
    )
    p.add_argument(
        "--url", type=str, default=None,
        help="Direct URL to CarDD_release.zip. CarDD has no stable public "
             "direct-download link (Google Drive, gated by a license form) — "
             "only pass this if you have a mirror URL you are entitled to use.",
    )
    p.add_argument(
        "--zip-path", type=Path, default=None,
        help="Path to an already-downloaded CarDD_release.zip "
             "(default: <raw_dir>/CarDD_release.zip).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-unpack even if data/raw/cardd/ already looks populated.",
    )
    return p.parse_args()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100 * done / total_size
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB / {total_size / 1e6:8.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)  # noqa: S310
    sys.stdout.write("\n")
    log.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def find_existing_zip(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("CarDD*release*.zip")) + sorted(raw_dir.glob("*.zip"))
    return candidates[0] if candidates else None


def unpack(zip_path: Path, dest_dir: Path) -> None:
    log.info("Unpacking %s -> %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member in archive: {bad}")
        zf.extractall(dest_dir)
    log.info("Unpack complete.")


def find_coco_root(cardd_dir: Path) -> Path | None:
    """Locate the directory containing an `annotations/` subfolder, wherever
    the zip happened to nest it (CarDD_release/CarDD_COCO/, CarDD_COCO/, or
    directly at cardd_dir)."""
    if (cardd_dir / "annotations").is_dir():
        return cardd_dir
    for candidate in cardd_dir.rglob("annotations"):
        if candidate.is_dir():
            return candidate.parent
    return None


def summarize_split(coco_root: Path, split: str) -> SplitSummary:
    ann_candidates = [
        coco_root / "annotations" / f"instances_{split}.json",
        coco_root / "annotations" / f"{split}.json",
    ]
    ann_file = next((p for p in ann_candidates if p.exists()), None)
    flat_img_dir = coco_root / split
    nested_img_dir = coco_root / "images" / split
    img_dir: Path | None
    if flat_img_dir.is_dir():
        img_dir = flat_img_dir
    elif nested_img_dir.is_dir():
        img_dir = nested_img_dir
    else:
        img_dir = None

    image_count = 0
    if img_dir is not None:
        image_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))

    annotated_image_count = 0
    instance_count = 0
    category_names: list[str] = []
    if ann_file is not None:
        with ann_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        annotated_image_count = len(coco.get("images", []))
        instance_count = len(coco.get("annotations", []))
        category_names = [c["name"] for c in coco.get("categories", [])]

    return SplitSummary(
        split=split,
        annotation_file=ann_file,
        image_dir=img_dir,
        image_count=image_count,
        annotated_image_count=annotated_image_count,
        instance_count=instance_count,
        category_names=category_names,
    )


def print_summary(cardd_dir: Path, zip_path: Path | None, summaries: list[SplitSummary]) -> None:
    print("\n" + "=" * 70)
    print("CarDD dataset summary")
    print("=" * 70)

    if zip_path is not None and zip_path.exists():
        size_mb = zip_path.stat().st_size / 1e6
        print(f"Archive:        {zip_path}")
        print(f"Archive size:   {size_mb:.1f} MB")
        print(f"SHA-256:        {sha256sum(zip_path)}")
    else:
        print("Archive:        (none found — verifying unpacked directory only)")

    print(f"Dataset root:   {cardd_dir}")
    print("-" * 70)

    total_images = 0
    total_instances = 0
    all_categories: set[str] = set()
    for s in summaries:
        status = "OK" if s.annotation_file and s.image_count else "MISSING"
        print(f"[{status}] split={s.split:<10} images={s.image_count:<6} "
              f"annotated={s.annotated_image_count:<6} instances={s.instance_count:<6}")
        if s.annotation_file:
            print(f"        annotations: {s.annotation_file}")
        if s.image_dir:
            print(f"        images dir:  {s.image_dir}")
        total_images += s.image_count
        total_instances += s.instance_count
        all_categories.update(s.category_names)

    print("-" * 70)
    print(f"Total images:      {total_images} (expected ~{EXPECTED_IMAGE_COUNT})")
    print(f"Total instances:   {total_instances}")
    print(f"Categories found:  {sorted(all_categories) if all_categories else '(none)'}")
    print(f"Expected classes:  {DAMAGE_CLASSES}")

    if total_images == 0:
        print("\nNo images found. To proceed:")
        print(f"  1. Fill in the license form at {CARDD_HOMEPAGE} to get the Google Drive link.")
        print("  2. Download CarDD_release.zip.")
        print("  3. Place it at data/raw/CarDD_release.zip (or pass --zip-path).")
        print("  4. Re-run this script.")
    elif total_images != EXPECTED_IMAGE_COUNT:
        print(f"\nWARNING: image count ({total_images}) does not match the expected "
              f"{EXPECTED_IMAGE_COUNT}. The archive may be incomplete or a different release.")
    else:
        missing_classes = set(DAMAGE_CLASSES) - all_categories
        if missing_classes:
            print(f"\nWARNING: expected classes not found in annotations: "
                  f"{sorted(missing_classes)}")
        else:
            print("\nDataset looks complete and matches the expected CarDD layout.")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / "download_data"
    setup_run_logging(run_dir)

    raw_dir = Path(cfg.data.raw_dir)
    cardd_dir = raw_dir / "cardd"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.zip_path or (raw_dir / "CarDD_release.zip")

    already_populated = find_coco_root(cardd_dir) is not None
    if already_populated and not args.force:
        log.info("data/raw/cardd already looks populated; skipping download/unpack "
                 "(use --force to redo).")
    else:
        if args.url:
            download(args.url, zip_path)
        elif not zip_path.exists():
            found = find_existing_zip(raw_dir)
            if found is not None:
                log.info("Found manually-downloaded archive: %s", found)
                zip_path = found
            else:
                log.warning(
                    "No --url given and no zip found at %s. CarDD requires filling in a "
                    "license form at %s to obtain the Google Drive download link — this "
                    "cannot be automated. Download CarDD_release.zip manually and place it "
                    "in %s, then re-run this script.",
                    zip_path, CARDD_HOMEPAGE, raw_dir,
                )
                zip_path = None

        if zip_path is not None and zip_path.exists():
            unpack(zip_path, cardd_dir)
        elif cardd_dir.exists() and not already_populated:
            log.warning("No archive available; nothing to unpack.")

    coco_root = find_coco_root(cardd_dir)
    if coco_root is None:
        log.warning("Could not locate a COCO-style 'annotations/' directory under %s", cardd_dir)
        print_summary(cardd_dir, zip_path if zip_path and zip_path.exists() else None, [])
        return

    summaries = [summarize_split(coco_root, split) for split in EXPECTED_SPLITS]
    print_summary(cardd_dir, zip_path if zip_path and zip_path.exists() else None, summaries)


if __name__ == "__main__":
    main()
