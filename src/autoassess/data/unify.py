"""Merge the processed CarDD and VehiDE datasets into one unified YOLO-seg dataset.

Reads configs/unified.yaml (an Ultralytics dataset YAML plus ``sources``,
``oversample`` and ``labelled_classes`` keys) and writes, under the config's
``path`` (resolved relative to the config file):

    images/{train,val}            both sources, filenames prefixed ``<source>__``
    images/test_<source>          per-source test split
    images/test                   combined test split
    labels/...                    matching YOLO-seg labels, class ids remapped
    train.txt                     training list; each image repeated oversample[source] times
    manifest.csv                  unified_name,source,source_name,split
    skipped.csv                   source,split,source_name,error (unreadable images, dropped)
    build_info.json               max_side, jpeg_quality and the source dataset configs

Oversampling mechanism: Ultralytics keeps duplicate lines of a list file
(``BaseDataset.get_img_files`` builds a plain list and only sorts it; each
entry is verified independently in ``cache_labels``), so repeats are plain
repeated lines in train.txt, with no extra copies on disk.

Source class ids are remapped through an explicit per-source ``class_map``; any
source class that is unmapped or maps to a name not in ``names`` is an error.

Usage
-----
    autoassess-unify --config configs/base.yaml --unified-config configs/unified.yaml
    autoassess-unify --max-side 1280 --overwrite
    autoassess-unify --max-side 800 --jpeg-quality 85 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image, ImageOps, UnidentifiedImageError

from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 92
SOURCE_SPLITS = ("train", "val", "test")


class UnifyError(Exception):
    """Raised on inconsistent configs, unknown classes or an unsafe output dir."""


@dataclass(frozen=True)
class Source:
    name: str
    root: Path
    class_ids: dict[int, int]  # source class id -> unified class id
    source_names: dict[int, str]
    oversample: int
    config: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge processed CarDD + VehiDE into one unified YOLOv8-seg dataset."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (project.output_root for run logs).")
    p.add_argument("--unified-config", type=Path, default=Path("configs/unified.yaml"),
                   help="Path to the unified dataset YAML (names, sources, oversample).")
    p.add_argument("--project-root", type=Path, default=Path("."),
                   help="Project root that relative source dataset configs resolve against.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed (default: config seed).")
    p.add_argument("--max-side", type=int, default=None,
                   help="If set, write copies downscaled so the long side is <= N "
                        "(never upscaled). Default: symlink the original images.")
    p.add_argument("--jpeg-quality", type=int, default=None,
                   help="JPEG quality 1-95. Requires --max-side. Re-encodes EVERY image "
                        "(EXIF-transposed, RGB, downscaled only if long side > max-side) "
                        "as JPEG; PNG inputs become .jpg. Default: unset (resized q92, "
                        "small images copied unchanged).")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete and rebuild a non-empty output directory.")
    return p.parse_args(argv)


def _names_dict(raw: dict[Any, Any] | list[Any]) -> dict[int, str]:
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {i: str(v) for i, v in enumerate(raw)}


def load_sources(
    unified: dict[str, Any], project_root: Path
) -> tuple[dict[int, str], list[Source]]:
    """Validate the unified config against each source's dataset YAML."""
    unified_names = _names_dict(unified["names"])
    if len(set(unified_names.values())) != len(unified_names):
        raise UnifyError(f"Duplicate names in unified `names`: {unified_names}")
    name_to_uid = {n: i for i, n in unified_names.items()}
    oversample = unified.get("oversample", {})
    labelled = unified.get("labelled_classes", {})

    sources: list[Source] = []
    for sname, scfg in unified["sources"].items():
        cfg_path = Path(scfg["config"])
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        ds_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert isinstance(ds_cfg, dict)
        src_names = _names_dict(ds_cfg["names"])
        class_map: dict[str, str] = dict(scfg["class_map"])

        if set(class_map) != set(src_names.values()):
            raise UnifyError(
                f"[{sname}] class_map keys {sorted(class_map)} != source class names "
                f"{sorted(src_names.values())} (missing: "
                f"{sorted(set(src_names.values()) - set(class_map))}, extra: "
                f"{sorted(set(class_map) - set(src_names.values()))})."
            )
        bad = sorted(v for v in class_map.values() if v not in name_to_uid)
        if bad:
            raise UnifyError(f"[{sname}] class_map targets not in unified names: {bad}")
        if sname in labelled and set(labelled[sname]) != set(class_map.values()):
            raise UnifyError(
                f"[{sname}] labelled_classes {sorted(labelled[sname])} != mapped classes "
                f"{sorted(set(class_map.values()))}"
            )
        factor = int(oversample.get(sname, 1))
        if factor < 1:
            raise UnifyError(f"[{sname}] oversample must be >= 1, got {factor}")

        root = (cfg_path.parent / str(ds_cfg["path"])).resolve()
        sources.append(Source(
            name=sname, root=root, source_names=src_names, oversample=factor, config=cfg_path,
            class_ids={i: name_to_uid[class_map[n]] for i, n in src_names.items()},
        ))
    return unified_names, sources


def remap_label_file(label_path: Path, source: Source) -> tuple[list[str], list[int]]:
    """Return (remapped lines, unified class ids); a missing file means background."""
    if not label_path.is_file():
        log.warning("[%s] no label file for %s; treating as background.", source.name, label_path)
        return [], []
    lines: list[str] = []
    classes: list[int] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError as e:
            raise UnifyError(f"[{source.name}] bad class token in {label_path}: {raw!r}") from e
        if cid not in source.class_ids:
            raise UnifyError(
                f"[{source.name}] unknown source class id {cid} in {label_path} "
                f"(known: {sorted(source.class_ids)})."
            )
        uid = source.class_ids[cid]
        classes.append(uid)
        lines.append(" ".join([str(uid), *parts[1:]]))
    return lines, classes


def _validate_image(src: Path) -> None:
    """Fully decode *src* (EXIF-transposed); raise OSError if it is unreadable.

    ``LOAD_TRUNCATED_IMAGES`` is deliberately left off so partial pixels are rejected.
    """
    with Image.open(src) as im:
        im.load()
        ImageOps.exif_transpose(im).load()


def _write_image(
    src: Path, dest: Path, max_side: int | None, jpeg_quality: int | None = None
) -> None:
    """Symlink (max_side None) or write a downscaled/copied/re-encoded image at *dest*."""
    if max_side is None:
        dest.symlink_to(src.resolve())
        return
    try:
        if jpeg_quality is not None:
            _write_reencoded(src, dest, max_side, jpeg_quality)
        else:
            _write_resized(src, dest, max_side)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _write_reencoded(src: Path, dest: Path, max_side: int, quality: int) -> None:
    """Decode, EXIF-transpose, RGB-convert, downscale if needed, save as JPEG."""
    with Image.open(src) as im:
        fixed = ImageOps.exif_transpose(im).convert("RGB")
        w, h = fixed.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            fixed = fixed.resize(size, Image.Resampling.LANCZOS)
        fixed.save(dest, format="JPEG", quality=quality)


def _write_resized(src: Path, dest: Path, max_side: int) -> None:
    with Image.open(src) as im:
        # Ultralytics reads images EXIF-corrected; bake that in before resizing.
        fixed = ImageOps.exif_transpose(im)
        w, h = fixed.size
        if max(w, h) <= max_side:
            shutil.copy2(src, dest)
            return
        scale = max_side / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        resized = fixed.resize(new_size, Image.Resampling.LANCZOS)
        if src.suffix.lower() == ".png":
            resized.save(dest, format="PNG")
        else:
            resized.convert("RGB").save(dest, format="JPEG", quality=JPEG_QUALITY)


def _link_or_copy(src: Path, dest: Path) -> None:
    """Make *dest* the same file as *src* (symlink stays a symlink; else hardlink/copy)."""
    if src.is_symlink():
        dest.symlink_to(src.resolve())
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _prepare_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.is_symlink():
        raise UnifyError(f"Refusing to use a symlink as output dir: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise UnifyError(
                f"Output dir {out_dir} is not empty; pass --overwrite to rebuild it."
            )
        log.warning("Removing existing output dir %s (--overwrite).", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def unify(
    unified_config: Path,
    project_root: Path,
    max_side: int | None = None,
    overwrite: bool = False,
    jpeg_quality: int | None = None,
) -> dict[str, Any]:
    """Build the unified dataset; return summary counts."""
    if max_side is not None and max_side < 1:
        raise UnifyError("--max-side must be a positive integer")
    if jpeg_quality is not None:
        if max_side is None:
            raise UnifyError("--jpeg-quality requires --max-side to be set")
        if not 1 <= jpeg_quality <= 95:
            raise UnifyError(f"--jpeg-quality must be between 1 and 95, got {jpeg_quality}")
    raw_cfg = OmegaConf.to_container(OmegaConf.load(unified_config), resolve=True)
    assert isinstance(raw_cfg, dict)
    unified: dict[str, Any] = {str(k): v for k, v in raw_cfg.items()}
    unified_names, sources = load_sources(unified, project_root)
    out_dir = (unified_config.parent / str(unified["path"])).resolve()
    log.info("Unified classes: %s", unified_names)
    log.info("Output dir: %s (max_side=%s, jpeg_quality=%s)", out_dir, max_side, jpeg_quality)

    _prepare_output(out_dir, overwrite)

    manifest: list[tuple[str, str, str, str]] = []
    skipped: list[tuple[str, str, str, str]] = []
    train_lines: list[str] = []
    per_split_source: Counter[tuple[str, str]] = Counter()
    per_split_class: Counter[tuple[str, int]] = Counter()

    for src in sources:
        for split in SOURCE_SPLITS:
            img_dir = src.root / "images" / split
            if not img_dir.is_dir():
                raise UnifyError(f"[{src.name}] missing image dir {img_dir}")
            # split -> dirs it is written to (test also goes to the combined dir)
            dests = ["test_" + src.name, "test"] if split == "test" else [split]
            for d in dests:
                (out_dir / "images" / d).mkdir(parents=True, exist_ok=True)
                (out_dir / "labels" / d).mkdir(parents=True, exist_ok=True)

            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_SUFFIXES:
                    continue
                out_name = img.with_suffix(".jpg").name if jpeg_quality is not None else img.name
                new_name = f"{src.name}__{out_name}"
                lines, classes = remap_label_file(
                    src.root / "labels" / split / (img.stem + ".txt"), src
                )
                label_text = "\n".join(lines) + ("\n" if lines else "")
                first_img = out_dir / "images" / dests[0] / new_name
                try:
                    _validate_image(img)
                    _write_image(img, first_img, max_side, jpeg_quality)
                except (OSError, UnidentifiedImageError) as e:
                    first_img.unlink(missing_ok=True)
                    err = f"{type(e).__name__}: {e}"
                    log.warning(
                        "[%s] skipping unreadable image split=%s file=%s (%s)",
                        src.name, split, img.name, err,
                    )
                    skipped.append((src.name, split, img.name, err))
                    continue
                for d in dests:
                    (out_dir / "labels" / d / f"{src.name}__{img.stem}.txt").write_text(
                        label_text, encoding="utf-8"
                    )
                    if d != dests[0]:
                        _link_or_copy(first_img, out_dir / "images" / d / new_name)

                manifest.append((new_name, src.name, img.name, split))
                per_split_source[(split, src.name)] += 1
                for c in classes:
                    per_split_class[(split, c)] += 1
                if split == "train":
                    train_lines.extend([f"./images/train/{new_name}"] * src.oversample)

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unified_name", "source", "source_name", "split"])
        w.writerows(manifest)
    with (out_dir / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "source_name", "error"])
        w.writerows(skipped)
    build_info = {
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "sources": {s.name: {"config": str(s.config)} for s in sources},
    }
    (out_dir / "build_info.json").write_text(
        json.dumps(build_info, indent=2) + "\n", encoding="utf-8"
    )
    if skipped:
        log.warning("Skipped %d unreadable image(s); see %s", len(skipped), out_dir / "skipped.csv")
    else:
        log.info("Skipped 0 unreadable images.")

    for (split, sname), n in sorted(per_split_source.items()):
        log.info("images  split=%-5s source=%-8s %d", split, sname, n)
    for (split, c), n in sorted(per_split_class.items()):
        log.info("instances split=%-5s class=%-14s %d", split, unified_names[c], n)
    log.info("train.txt entries (after oversampling): %d", len(train_lines))
    log.info("Unified dataset written to %s", out_dir)
    return {
        "out_dir": out_dir,
        "images": dict(per_split_source),
        "instances": dict(per_split_class),
        "train_entries": len(train_lines),
        "skipped": len(skipped),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    cfg = resolve_paths(load_config(args.config), project_root)
    setup_run_logging(Path(cfg.project.output_root) / "unify_data")
    seed_everything(args.seed if args.seed is not None else cfg.project.seed)

    unify(args.unified_config, project_root, args.max_side, args.overwrite, args.jpeg_quality)


if __name__ == "__main__":
    main()
