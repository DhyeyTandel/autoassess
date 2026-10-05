"""Tests for scripts/split_audit.py (near-duplicate audit across splits)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "split_audit.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("split_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["split_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sa() -> ModuleType:
    return _load()


def _synthetic(seed: int, size: int = 128) -> Image.Image:
    """Smooth, structured RGB image: low-res noise upscaled (stable under recompression)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC)


def _recompressed(img: Image.Image, path: Path, quality: int = 40) -> None:
    img.save(path, format="JPEG", quality=quality)


def test_phash_deterministic_and_robust(sa: ModuleType, tmp_path: Path) -> None:
    base = _synthetic(1)
    p0 = tmp_path / "base.png"
    base.save(p0)
    h0 = sa.phash(p0)
    assert h0 == sa.phash(p0)
    assert 0 <= h0 < 2**64

    p1 = tmp_path / "recomp.jpg"
    _recompressed(base, p1)
    bright = Image.fromarray(np.clip(np.asarray(base).astype(int) + 12, 0, 255).astype(np.uint8))
    p2 = tmp_path / "bright.png"
    bright.save(p2)
    p3 = tmp_path / "other.png"
    _synthetic(2).save(p3)

    assert bin(h0 ^ sa.phash(p1)).count("1") <= 6
    assert bin(h0 ^ sa.phash(p2)).count("1") <= 6
    assert bin(h0 ^ sa.phash(p3)).count("1") > 10


def test_hamming_matrix_matches_naive(sa: ModuleType) -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**64, size=7, dtype=np.uint64)
    b = rng.integers(0, 2**64, size=11, dtype=np.uint64)
    a[0] = np.uint64(2**63)
    b[0] = np.uint64(2**64 - 1)
    got = sa.hamming_matrix(a, b)
    assert got.shape == (7, 11)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            assert got[i, j] == bin(int(x) ^ int(y)).count("1")


def _fake_processed(root: Path) -> Path:
    train = root / "images" / "train"
    test = root / "images" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    for i in range(5):
        _synthetic(100 + i).save(train / f"tr_{i}.png")
    _recompressed(_synthetic(102), test / "te_dup.jpg")
    _synthetic(200).save(test / "te_a.png")
    _synthetic(201).save(test / "te_b.png")
    return root


def test_audit_split_flags_only_the_duplicate(sa: ModuleType, tmp_path: Path) -> None:
    res = sa.audit_split(_fake_processed(tmp_path / "ds"), threshold=6, top_k=30)
    assert res.n_train == 5
    assert res.n_test == 3
    assert res.n_flagged == 1
    assert res.pct_flagged == pytest.approx(100 / 3)
    assert sum(res.histogram) == 3
    assert len(res.histogram) == 65
    first = res.pairs[0]
    assert first.test.name == "te_dup.jpg"
    assert first.train.name == "tr_2.png"
    assert first.distance <= 6
    dists = [p.distance for p in res.pairs]
    assert dists == sorted(dists)
    assert len(res.pairs) == 3
    assert all(p.distance > 6 for p in res.pairs[1:])
    assert len(sa.audit_split(tmp_path / "ds", top_k=2).pairs) == 2


def test_main_writes_only_under_reports_dir(
    sa: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_processed(tmp_path / "data" / "processed" / "fake")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "fake.yaml"
    cfg.write_text(f"path: {proc}\ntrain: images/train\ntest: images/test\nnames:\n  0: x\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    reports = tmp_path / "out" / "reports"

    rc = sa.main(["--dataset-config", str(cfg), "--reports-dir", str(reports), "--top-k", "2"])
    assert rc == 0
    md = (reports / "split_audit.md").read_text()
    assert "## fake" in md
    assert "figures/split_audit/fake/pair_01_d" in md
    figs = sorted((reports / "figures" / "split_audit" / "fake").glob("pair_*_d*.jpg"))
    assert len(figs) == 2
    assert figs[0].name.startswith("pair_01_d")
    assert list(cwd.iterdir()) == []
    # nothing written into the data dir
    assert not list(proc.rglob("*.md")) and not list(proc.rglob("pair_*"))
