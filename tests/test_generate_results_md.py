"""generate_results_md renders v2 and legacy v1 runs from a tmp runs dir."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoassess.eval.compare import LEGACY_NOTICE
from autoassess.eval.metrics import write_metrics_json

from .conftest import SyntheticCase
from .test_compare import V1_PRECISION, make_v1, make_v2

SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_results_md.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_results_md", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _section(md: str, title: str) -> str:
    start = md.index(f"## {title}")
    nxt = md.find("\n## ", start + 1)
    return md[start:] if nxt < 0 else md[start:nxt]


def test_generate_results_md_v2_and_v1(
    synthetic_case: SyntheticCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    write_metrics_json(runs / "yolov8_seg_v1", make_v2(synthetic_case, "yolov8_seg_v1"))
    v1 = make_v1("maskrcnn_v1")
    (runs / "maskrcnn_v1").mkdir(parents=True)
    (runs / "maskrcnn_v1" / "metrics.json").write_text(json.dumps(v1), encoding="utf-8")
    out = tmp_path / "out" / "results.md"

    mod = _load_script()
    monkeypatch.setattr(
        sys, "argv",
        ["generate_results_md.py", "--runs-dir", str(runs), "--out", str(out),
         "--figures-dir", str(tmp_path / "figs")],
    )
    mod.main()
    md = out.read_text(encoding="utf-8")

    v2_sec = _section(md, "Damage detection — YOLOv8-seg")
    assert "Precision@0.25" in v2_sec and "Best-F1 conf" in v2_sec and "Max recall" in v2_sec
    assert "| Evaluated on | test |" in v2_sec
    assert "Mask IoU (per GT, misses = 0)" in v2_sec
    assert LEGACY_NOTICE not in v2_sec
    exp = synthetic_case.expected["conf_0.25"]["scratch"]
    row = next(ln for ln in v2_sec.splitlines() if ln.startswith("| scratch |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:4] == [f"{exp['precision']:.4f}", f"{exp['recall']:.4f}", f"{exp['f1']:.4f}"]

    v1_sec = _section(md, "Damage detection — Mask R-CNN")
    assert LEGACY_NOTICE in v1_sec
    assert "val (legacy)" in v1_sec
    assert "Mask IoU (legacy, matched pairs only) | 0.8123" in v1_sec
    row = next(ln for ln in v1_sec.splitlines() if ln.startswith("| dent |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:5] == ["n/a"] * 4
    assert cells[6] == f"{V1_PRECISION:.4f}"  # only as Mask AP50
    assert "0.9911" not in v1_sec
