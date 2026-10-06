"""Tests for the CarDD + VehiDE unifier, using two tiny synthetic sources.

Source "alpha" labels [dent, scratch]; source "beta" labels [scrape, hole] where
scrape -> scratch and hole -> hole in the unified space [dent, scratch, hole].
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from autoassess.data.unify import UnifyError, main

UNIFIED_NAMES = ["dent", "scratch", "hole"]


def _write_source(
    root: Path,
    name: str,
    names: list[str],
    splits: dict[str, list[tuple[str, tuple[int, int], list[int]]]],
) -> Path:
    """Write a processed-style dataset + its dataset YAML; return the YAML path."""
    ds_root = root / "processed" / name
    for split, items in splits.items():
        (ds_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for fname, size, classes in items:
            img = Image.new("RGB", size, (120, 40, 40))
            img.save(ds_root / "images" / split / fname)
            lines = [f"{c} 0.1 0.1 0.5 0.1 0.5 0.5" for c in classes]
            (ds_root / "labels" / split / (Path(fname).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
    cfg_path = root / f"{name}.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "path": f"./processed/{name}",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(names)),
            }
        )
    )
    return cfg_path


class Env:
    def __init__(self, root: Path, unified_cfg: Path, base_cfg: Path) -> None:
        self.root = root
        self.unified_cfg = unified_cfg
        self.base_cfg = base_cfg
        self.out = root / "data" / "processed" / "unified"

    def run(self, *extra: str) -> None:
        main(
            [
                "--config", str(self.base_cfg),
                "--unified-config", str(self.unified_cfg),
                "--project-root", str(self.root),
                *extra,
            ]
        )


def _build_env(
    tmp_path: Path,
    beta_class_map: dict[str, str] | None = None,
    beta_labels: list[int] | None = None,
) -> Env:
    alpha = _write_source(
        tmp_path, "alpha", ["dent", "scratch"],
        {
            "train": [("a1.jpg", (200, 100), [0, 1]), ("a2.jpg", (50, 40), [1])],
            "val": [("a3.jpg", (200, 100), [0])],
            "test": [("a4.jpg", (100, 200), [1])],
        },
    )
    beta = _write_source(
        tmp_path, "beta", ["scrape", "hole"],
        {
            "train": [("b1.png", (120, 60), beta_labels or [0, 1])],
            "val": [("b2.jpg", (60, 60), [1])],
            "test": [("b3.jpg", (60, 60), [0])],
        },
    )
    unified = {
        "path": "../data/processed/unified",
        "train": "train.txt",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(UNIFIED_NAMES)),
        "sources": {
            "alpha": {
                "config": str(alpha),
                "class_map": {"dent": "dent", "scratch": "scratch"},
            },
            "beta": {
                "config": str(beta),
                "class_map": beta_class_map or {"scrape": "scratch", "hole": "hole"},
            },
        },
        "oversample": {"alpha": 3, "beta": 1},
        "labelled_classes": {"alpha": ["dent", "scratch"], "beta": ["scratch", "hole"]},
    }
    (tmp_path / "configs").mkdir()
    unified_cfg = tmp_path / "configs" / "unified.yaml"
    unified_cfg.write_text(yaml.safe_dump(unified))
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        yaml.safe_dump({"project": {"seed": 42, "output_root": "runs"},
                        "data": {"processed_dir": "data/processed"}})
    )
    return Env(tmp_path, unified_cfg, base_cfg)


def _labels(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines() if ln.strip()]


def test_class_remapping(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    # alpha is identity
    assert [r[0] for r in _labels(env.out / "labels/train/alpha__a1.txt")] == ["0", "1"]
    # beta scrape(0) -> scratch(1); hole(1) -> hole(2)
    assert [r[0] for r in _labels(env.out / "labels/train/beta__b1.txt")] == ["1", "2"]
    # coordinates untouched
    coords = "0.1 0.1 0.5 0.1 0.5 0.5".split()
    assert _labels(env.out / "labels/train/beta__b1.txt")[0][1:] == coords


def test_unknown_source_class_id_fails(tmp_path: Path) -> None:
    env = _build_env(tmp_path, beta_labels=[0, 5])
    with pytest.raises(UnifyError, match="5"):
        env.run()


def test_class_map_must_cover_source_names(tmp_path: Path) -> None:
    env = _build_env(tmp_path, beta_class_map={"scrape": "scratch"})
    with pytest.raises(UnifyError, match="hole"):
        env.run()


def test_class_map_target_must_be_unified_name(tmp_path: Path) -> None:
    env = _build_env(tmp_path, beta_class_map={"scrape": "scratch", "hole": "nonsense"})
    with pytest.raises(UnifyError, match="nonsense"):
        env.run()


def test_filename_prefixing_and_manifest(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    names = sorted(p.name for p in (env.out / "images/train").iterdir())
    assert names == ["alpha__a1.jpg", "alpha__a2.jpg", "beta__b1.png"]
    assert [p.name for p in (env.out / "images/val").iterdir() if p.name.startswith("alpha")]
    df = pd.read_csv(env.out / "manifest.csv")
    assert list(df.columns) == ["unified_name", "source", "source_name", "split"]
    row = df[df.unified_name == "beta__b1.png"].iloc[0]
    assert (row.source, row.source_name, row.split) == ("beta", "b1.png", "train")
    assert len(df) == 3 + 2 + 2  # train + val + test across both sources


def test_per_source_and_combined_test_splits(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    assert [p.name for p in (env.out / "images/test_alpha").iterdir()] == ["alpha__a4.jpg"]
    assert [p.name for p in (env.out / "images/test_beta").iterdir()] == ["beta__b3.jpg"]
    assert sorted(p.name for p in (env.out / "images/test").iterdir()) == [
        "alpha__a4.jpg", "beta__b3.jpg"
    ]
    for d in ("test_alpha", "test_beta", "test"):
        for img in (env.out / "images" / d).iterdir():
            assert (env.out / "labels" / d / (img.stem + ".txt")).is_file()
    # beta test label remapped: scrape(0) -> scratch(1)
    assert _labels(env.out / "labels/test_beta/beta__b3.txt")[0][0] == "1"


def test_max_side_resizes_without_upscaling(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run("--max-side", "100")
    with Image.open(env.out / "images/train/alpha__a1.jpg") as im:
        assert im.size == (100, 50)  # 200x100 -> long side 100, aspect kept
    with Image.open(env.out / "images/train/alpha__a2.jpg") as im:
        assert im.size == (50, 40)  # already small: not upscaled
    with Image.open(env.out / "images/test/alpha__a4.jpg") as im:
        assert im.size == (50, 100)
    with Image.open(env.out / "images/train/beta__b1.png") as im:
        assert im.format == "PNG"
        assert im.size == (100, 50)
    assert not (env.out / "images/train/alpha__a1.jpg").is_symlink()
    # labels identical to the non-resized run (normalised coords)
    assert _labels(env.out / "labels/train/alpha__a1.txt") == [
        ["0", "0.1", "0.1", "0.5", "0.1", "0.5", "0.5"],
        ["1", "0.1", "0.1", "0.5", "0.1", "0.5", "0.5"],
    ]


def test_symlink_mode_without_max_side(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    link = env.out / "images/train/alpha__a1.jpg"
    assert link.is_symlink()
    assert link.resolve() == (tmp_path / "processed/alpha/images/train/a1.jpg").resolve()
    with Image.open(link) as im:
        assert im.size == (200, 100)


def test_oversampling_list(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    lines = (env.out / "train.txt").read_text().splitlines()
    assert lines.count("./images/train/alpha__a1.jpg") == 3
    assert lines.count("./images/train/alpha__a2.jpg") == 3
    assert lines.count("./images/train/beta__b1.png") == 1
    assert len(lines) == 7
    # every listed path exists relative to the dataset root
    for ln in set(lines):
        assert (env.out / ln).exists()
    # no extra physical copies were created for repeats
    assert len(list((env.out / "images/train").iterdir())) == 3


@pytest.mark.slow
def test_ultralytics_preserves_duplicate_lines(tmp_path: Path) -> None:
    from ultralytics.data.dataset import YOLODataset

    env = _build_env(tmp_path)
    env.run()
    ds = YOLODataset(
        img_path=str(env.out / "train.txt"),
        data={"names": dict(enumerate(UNIFIED_NAMES)), "channels": 3},
        task="segment",
        imgsz=64,
        augment=False,
    )
    assert len(ds) == 7
    counts: dict[str, int] = {}
    for lb in ds.labels:
        counts[Path(lb["im_file"]).name] = counts.get(Path(lb["im_file"]).name, 0) + 1
    assert counts == {"alpha__a1.jpg": 3, "alpha__a2.jpg": 3, "beta__b1.png": 1}


def test_overwrite_guard(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    marker = env.out / "stale.txt"
    marker.write_text("x")
    with pytest.raises(UnifyError, match="overwrite"):
        env.run()
    assert marker.exists()  # refused: nothing touched
    env.run("--overwrite")
    assert not marker.exists()
    assert (env.out / "train.txt").is_file()


def test_empty_existing_output_dir_is_ok(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.out.mkdir(parents=True)
    env.run()
    assert (env.out / "manifest.csv").is_file()



def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _corrupt_sources(root: Path) -> None:
    alpha = root / "processed" / "alpha" / "images"
    beta = root / "processed" / "beta" / "images"
    _truncate(alpha / "train" / "a1.jpg")
    (alpha / "train" / "a2.jpg").write_bytes(b"this is not an image")
    _truncate(alpha / "test" / "a4.jpg")
    (beta / "test" / "b3.jpg").write_bytes(b"garbage garbage garbage")


@pytest.mark.parametrize("extra", [[], ["--max-side", "100"]], ids=["symlink", "resize"])
def test_unreadable_images_are_skipped(tmp_path: Path, extra: list[str]) -> None:
    env = _build_env(tmp_path)
    _corrupt_sources(tmp_path)
    env.run(*extra)

    bad = ["alpha__a1.jpg", "alpha__a2.jpg", "alpha__a4.jpg", "beta__b3.jpg"]
    for d in ("train", "val", "test", "test_alpha", "test_beta"):
        for name in bad:
            assert not (env.out / "images" / d / name).exists()
            assert not (env.out / "images" / d / name).is_symlink()
            stem = Path(name).stem
            assert not (env.out / "labels" / d / f"{stem}.txt").exists()

    # good images survive, no leftovers
    assert sorted(p.name for p in (env.out / "images/train").iterdir()) == ["beta__b1.png"]
    assert list((env.out / "images/test_alpha").iterdir()) == []
    assert list((env.out / "images/test").iterdir()) == []
    assert (env.out / "images/val/alpha__a3.jpg").exists()

    assert (env.out / "train.txt").read_text().splitlines() == ["./images/train/beta__b1.png"]
    df = pd.read_csv(env.out / "manifest.csv")
    assert sorted(df.unified_name) == ["alpha__a3.jpg", "beta__b1.png", "beta__b2.jpg"]

    sk = pd.read_csv(env.out / "skipped.csv")
    assert list(sk.columns) == ["source", "split", "source_name", "error"]
    assert sorted(sk.source_name) == ["a1.jpg", "a2.jpg", "a4.jpg", "b3.jpg"]
    assert set(zip(sk.source, sk.split, strict=True)) == {
        ("alpha", "train"), ("alpha", "test"), ("beta", "test")
    }
    assert sk.error.notna().all() and (sk.error.str.len() > 0).all()


def test_skipped_csv_written_when_nothing_skipped(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    env.run()
    sk = pd.read_csv(env.out / "skipped.csv")
    assert list(sk.columns) == ["source", "split", "source_name", "error"]
    assert len(sk) == 0


def _jpeg_env(tmp_path: Path) -> Env:
    """Env plus: a high-quality noisy EXIF-rotated small JPEG, a big JPEG, a test PNG."""
    import os

    env = _build_env(tmp_path)
    alpha = tmp_path / "processed" / "alpha"
    noisy = Image.frombytes("RGB", (300, 200), os.urandom(300 * 200 * 3))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display -> 200x300
    noisy.save(alpha / "images/train/a2.jpg", quality=100, exif=exif)
    Image.new("RGB", (1600, 1000), (10, 90, 10)).save(alpha / "images/train/a1.jpg")
    Image.new("RGB", (80, 60), (5, 5, 200)).save(alpha / "images/test/a5.png")
    (alpha / "labels/test/a5.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
    return env


def test_jpeg_quality_reencodes_everything(tmp_path: Path) -> None:
    env = _jpeg_env(tmp_path)
    src = tmp_path / "processed/alpha/images/train/a2.jpg"
    env.run("--max-side", "800", "--jpeg-quality", "85")

    small = env.out / "images/train/alpha__a2.jpg"
    assert small.read_bytes() != src.read_bytes()
    assert small.stat().st_size != src.stat().st_size
    with Image.open(small) as im:
        assert im.format == "JPEG"
        assert im.size == (200, 300)  # EXIF rotation baked in, not upscaled
        assert im.getexif().get(0x0112, 1) == 1

    with Image.open(env.out / "images/train/alpha__a1.jpg") as im:
        assert max(im.size) == 800
        assert im.size == (800, 500)


def test_jpeg_quality_png_becomes_jpg_everywhere(tmp_path: Path) -> None:
    env = _jpeg_env(tmp_path)
    env.run("--max-side", "800", "--jpeg-quality", "85")

    assert not list(env.out.rglob("*.png"))
    with Image.open(env.out / "images/train/beta__b1.jpg") as im:
        assert im.format == "JPEG"
    assert "./images/train/beta__b1.jpg" in (env.out / "train.txt").read_text().splitlines()
    for ln in set((env.out / "train.txt").read_text().splitlines()):
        assert (env.out / ln).is_file()
    df = pd.read_csv(env.out / "manifest.csv")
    assert "beta__b1.jpg" in set(df.unified_name)
    assert "alpha__a5.jpg" in set(df.unified_name)
    row = df[df.unified_name == "beta__b1.jpg"].iloc[0]
    assert row.source_name == "b1.png"  # original name kept for provenance
    for d in ("test", "test_alpha"):
        assert (env.out / "images" / d / "alpha__a5.jpg").is_file()
        assert (env.out / "labels" / d / "alpha__a5.txt").is_file()
    assert _labels(env.out / "labels/train/beta__b1.txt")[0][0] == "1"


def test_jpeg_quality_requires_max_side(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    with pytest.raises(UnifyError, match="--max-side"):
        env.run("--jpeg-quality", "85")
    assert not (env.out / "train.txt").exists()


@pytest.mark.parametrize("q", ["0", "96"])
def test_jpeg_quality_range(tmp_path: Path, q: str) -> None:
    env = _build_env(tmp_path)
    with pytest.raises(UnifyError, match="1"):
        env.run("--max-side", "800", "--jpeg-quality", q)


def test_build_info_json(tmp_path: Path) -> None:
    import json

    env = _build_env(tmp_path)
    env.run("--max-side", "800", "--jpeg-quality", "85")
    info = json.loads((env.out / "build_info.json").read_text())
    assert info["max_side"] == 800
    assert info["jpeg_quality"] == 85
    assert set(info["sources"]) == {"alpha", "beta"}
    assert info["sources"]["alpha"]["config"].endswith("alpha.yaml")

    env.run("--overwrite")
    info = json.loads((env.out / "build_info.json").read_text())
    assert info["max_side"] is None and info["jpeg_quality"] is None


def test_jpeg_quality_skips_unreadable(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    _corrupt_sources(tmp_path)
    env.run("--max-side", "100", "--jpeg-quality", "85")
    assert sorted(p.name for p in (env.out / "images/train").iterdir()) == ["beta__b1.jpg"]
    assert list((env.out / "images/test").iterdir()) == []
    sk = pd.read_csv(env.out / "skipped.csv")
    assert sorted(sk.source_name) == ["a1.jpg", "a2.jpg", "a4.jpg", "b3.jpg"]
