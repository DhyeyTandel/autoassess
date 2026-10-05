#!/usr/bin/env python3
"""scripts/split_audit.py — report-only near-duplicate audit between train and test.

Hashes every train and test image with a 64-bit DCT perceptual hash, finds each
test image's nearest train image by Hamming distance, and writes a markdown
report plus side-by-side figures of the closest pairs. It never re-splits
anything: it only reads the processed dataset and only writes under --reports-dir.

Usage
-----
    python scripts/split_audit.py \\
        --dataset-config configs/cardd.yaml --dataset-config configs/carparts.yaml \\
        --threshold 6 --top-k 30 --reports-dir reports
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.eval.yolo_eval import resolve_processed_dir  # noqa: E402
from autoassess.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HIST_BINS: list[tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-10", 7, 10),
    ("11-20", 11, 20),
    (">20", 21, 64),
]
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class Pair:
    """A test image with its nearest train image."""

    test: Path
    train: Path
    distance: int


@dataclass(frozen=True)
class AuditResult:
    """Outcome of auditing one dataset."""

    n_train: int
    n_test: int
    threshold: int
    n_flagged: int
    pct_flagged: float
    histogram: list[int]  # nearest-neighbour distance counts, index 0..64
    pairs: list[Pair]  # top_k closest, sorted by (distance, test name, train name)


def phash(image_path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Standard DCT perceptual hash packed into an int of hash_size**2 bits.

    Grayscale, resize to (hash_size*highfreq_factor)^2 with LANCZOS, 2D DCT,
    keep the top-left hash_size x hash_size block, threshold at the median of
    those coefficients excluding the DC term. The first coefficient is the MSB.
    """
    side = hash_size * highfreq_factor
    with Image.open(image_path) as im:
        gray = im.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    dct = cv2.dct(np.asarray(gray, dtype=np.float32))
    low = dct[:hash_size, :hash_size].flatten()
    median = np.median(low[1:])
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Pairwise Hamming distances between uint64 hash arrays, shape (len(a), len(b))."""
    a = np.ascontiguousarray(a, dtype=np.uint64)
    b = np.ascontiguousarray(b, dtype=np.uint64)
    out = np.empty((len(a), len(b)), dtype=np.int32)
    for start in range(0, len(a), chunk):
        xor = a[start : start + chunk, None] ^ b[None, :]
        bytes_ = xor.view(np.uint8).reshape(xor.shape[0], xor.shape[1], 8)
        out[start : start + chunk] = _POPCOUNT[bytes_].sum(axis=2, dtype=np.int32)
    return out


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _hash_all(paths: list[Path], label: str) -> npt.NDArray[np.uint64]:
    hashes = np.empty(len(paths), dtype=np.uint64)
    step = max(1, len(paths) // 10)
    for i, p in enumerate(paths):
        hashes[i] = np.uint64(phash(p))
        if (i + 1) % step == 0 or i + 1 == len(paths):
            log.info("hashed %s %d/%d", label, i + 1, len(paths))
    return hashes


def audit_split(processed_dir: Path, threshold: int = 6, top_k: int = 30) -> AuditResult:
    """Find each test image's nearest train image by phash Hamming distance.

    The top_k list holds one pair per test image (its nearest train image),
    ranked closest first, so one test image cannot fill the list on its own.
    """
    train_paths = _list_images(processed_dir / "images" / "train")
    test_paths = _list_images(processed_dir / "images" / "test")
    if not train_paths or not test_paths:
        raise ValueError(f"empty train or test images dir under {processed_dir}")
    dist = hamming_matrix(_hash_all(test_paths, "test"), _hash_all(train_paths, "train"))
    nearest = dist.argmin(axis=1)
    nn_dist = dist[np.arange(len(test_paths)), nearest]
    histogram = np.bincount(nn_dist, minlength=65).tolist()
    n_flagged = int((nn_dist <= threshold).sum())
    pairs = sorted(
        (
            Pair(test_paths[i], train_paths[int(nearest[i])], int(nn_dist[i]))
            for i in range(len(test_paths))
        ),
        key=lambda p: (p.distance, p.test.name, p.train.name),
    )[:top_k]
    return AuditResult(
        n_train=len(train_paths),
        n_test=len(test_paths),
        threshold=threshold,
        n_flagged=n_flagged,
        pct_flagged=100.0 * n_flagged / len(test_paths),
        histogram=histogram,
        pairs=pairs,
    )


def _side_by_side(pair: Pair, out_path: Path, height: int = 320) -> None:
    panels: list[Image.Image] = []
    for path in (pair.test, pair.train):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        w = max(1, round(rgb.width * height / rgb.height))
        panels.append(rgb.resize((w, height), Image.Resampling.LANCZOS))
    label_h = 36
    canvas = Image.new("RGB", (panels[0].width + panels[1].width, height + label_h), "white")
    canvas.paste(panels[0], (0, label_h))
    canvas.paste(panels[1], (panels[0].width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"TEST  {pair.test.name}", fill="black")
    draw.text((panels[0].width + 4, 2), f"TRAIN {pair.train.name}", fill="black")
    draw.text((4, 18), f"hamming distance = {pair.distance}", fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=90)


def write_figures(name: str, result: AuditResult, reports_dir: Path) -> list[Path]:
    """Save one side-by-side JPEG per top-k pair; returns paths in rank order."""
    fig_dir = reports_dir / "figures" / "split_audit" / name
    paths: list[Path] = []
    for rank, pair in enumerate(result.pairs, start=1):
        out = fig_dir / f"pair_{rank:02d}_d{pair.distance}.jpg"
        _side_by_side(pair, out)
        paths.append(out)
    return paths


def _section(name: str, result: AuditResult, figures: list[Path], reports_dir: Path) -> str:
    lines = [
        f"## {name}",
        "",
        f"**Method:** phash (64-bit DCT), nearest train image per test image, "
        f"flagged when Hamming distance <= {result.threshold}.",
        "",
        f"- Train images: {result.n_train}",
        f"- Test images: {result.n_test}",
        f"- Test images with a train neighbour at distance <= {result.threshold}: "
        f"{result.n_flagged} ({result.pct_flagged:.1f}%)",
        "",
        "Nearest-neighbour distance histogram (test images per bin):",
        "",
        "| Distance | Test images |",
        "|---|---|",
    ]
    for label, lo, hi in HIST_BINS:
        lines.append(f"| {label} | {sum(result.histogram[lo : hi + 1])} |")
    lines += [
        "",
        f"Top {len(result.pairs)} closest pairs:",
        "",
        "| Rank | Test | Train | Distance | Figure |",
        "|---|---|---|---|---|",
    ]
    for rank, (pair, fig) in enumerate(zip(result.pairs, figures, strict=True), start=1):
        rel = fig.relative_to(reports_dir).as_posix()
        lines.append(
            f"| {rank} | {pair.test.name} | {pair.train.name} | {pair.distance} | "
            f"[{fig.name}]({rel}) |"
        )
    lines += [
        "",
        "**How to read this.** phash flags visually similar photos, not proven leakage: "
        "open the figures and review the closest pairs by eye to decide whether they are "
        "the same photo or vehicle. Same-vehicle shots taken from a different angle or "
        "distance will NOT be caught, so a low flagged count is not proof of a clean split.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--dataset-config",
        action="append",
        type=Path,
        help="Dataset YAML to audit (repeatable). Default: configs/cardd.yaml and "
        "configs/carparts.yaml.",
    )
    parser.add_argument(
        "--threshold", type=int, default=6, help="Flag test images at Hamming distance <= this."
    )
    parser.add_argument(
        "--top-k", type=int, default=30, help="Number of closest pairs to list and render."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory the report and figures are written under.",
    )
    args = parser.parse_args(argv)
    configs: list[Path] = args.dataset_config or [
        Path("configs/cardd.yaml"),
        Path("configs/carparts.yaml"),
    ]
    reports_dir: Path = args.reports_dir

    sections: list[str] = []
    for cfg in configs:
        name = cfg.stem
        processed = resolve_processed_dir(cfg)
        if not _list_images(processed / "images" / "train") or not _list_images(
            processed / "images" / "test"
        ):
            log.warning("skipping %s: train or test images missing/empty under %s", name, processed)
            continue
        log.info("auditing %s (%s)", name, processed)
        result = audit_split(processed, threshold=args.threshold, top_k=args.top_k)
        figures = write_figures(name, result, reports_dir)
        sections.append(_section(name, result, figures, reports_dir))

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_md = reports_dir / "split_audit.md"
    out_md.write_text("# Train/test near-duplicate audit\n\n" + "\n".join(sections))
    log.info("wrote %s (%d dataset sections)", out_md, len(sections))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sys.exit(main())
