"""Trainers report on the test split through the evaluator (T6). CPU, offline."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from autoassess.train import train_maskrcnn, train_parts, train_yolo

YOLO_TRAINERS = [train_yolo, train_parts]
EXPECTED_LABEL = {
    "autoassess.train.train_yolo": "yolov8-seg",
    "autoassess.train.train_parts": "yolov8-seg-parts",
}
TRAINING_INFO_KEYS = {
    "epochs_requested",
    "epochs_run_this_invocation",
    "batch",
    "patience",
    "best_epoch",
    "early_stopped",
    "wall_time_seconds_total",
    "epoch_wall_times_seconds",
    "vram_peak_mb_training",
}


def _recorder(calls: list[dict[str, Any]], result: Path) -> Callable[..., Path]:
    def fake(**kwargs: Any) -> Path:  # noqa: ANN401
        calls.append(kwargs)
        return result

    return fake


def _yolo_report_kwargs(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "dataset_config": Path("configs/cardd.yaml"),
        "imgsz": 512,
        "device": "cpu",
        "epochs": 50,
        "batch": 8,
        "patience": 20,
        "best_epoch": 11,
        "early_stopped": True,
        "wall_time_seconds_total": 99.5,
        "epoch_wall_times_seconds": [1.0, 2.0],
        "vram_peak_mb_training": 1234.5,
    }


def _yolo_kwargs(run_dir: Path) -> dict[str, Any]:
    return {**_yolo_report_kwargs(run_dir), "model": "yolov8s-seg.pt"}


@pytest.mark.parametrize("trainer", YOLO_TRAINERS, ids=lambda m: m.__name__)
def test_yolo_report_on_test_uses_test_split(
    trainer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "weights" / "last.pt").write_bytes(b"y")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(trainer, "run_evaluation", _recorder(calls, run_dir / "metrics.json"))

    out = trainer.report_on_test(**_yolo_kwargs(run_dir))

    assert out == run_dir / "metrics.json"
    (kw,) = calls
    assert kw["split"] == "test"
    assert kw["output_dir"] == run_dir
    assert kw["configure_logging"] is False
    assert kw["imgsz"] == 512
    assert kw["device"] == "cpu"
    assert kw["model_type"] == "yolo"
    assert kw["model_id"] == "yolov8s-seg.pt"
    assert kw["model_type_label"] == EXPECTED_LABEL[trainer.__name__]
    assert kw["weights"] == run_dir / "weights" / "best.pt"
    assert kw["dataset_config"] == Path("configs/cardd.yaml")
    info = kw["training_info"]
    assert TRAINING_INFO_KEYS <= set(info)
    assert info["vram_peak_mb_training"] == 1234.5
    assert info["best_epoch"] == 11
    assert info["epochs_requested"] == 50
    assert "params_total" not in info  # the evaluator counts params


@pytest.mark.parametrize("trainer", YOLO_TRAINERS, ids=lambda m: m.__name__)
def test_yolo_report_falls_back_to_last_pt(
    trainer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"y")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(trainer, "run_evaluation", _recorder(calls, run_dir / "metrics.json"))

    with caplog.at_level(logging.WARNING):
        trainer.report_on_test(**_yolo_kwargs(run_dir))

    assert calls[0]["weights"] == run_dir / "weights" / "last.pt"
    assert any("last.pt" in r.getMessage() for r in caplog.records)


def test_maskrcnn_report_on_test_uses_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"x")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        train_maskrcnn, "run_evaluation", _recorder(calls, run_dir / "metrics.json")
    )

    out = train_maskrcnn.report_on_test(
        model=torch.nn.Linear(2, 2),
        **_yolo_report_kwargs(run_dir),
    )

    assert out == run_dir / "metrics.json"
    (kw,) = calls
    assert kw["split"] == "test"
    assert kw["output_dir"] == run_dir
    assert kw["configure_logging"] is False
    assert kw["imgsz"] == 512
    assert kw["model_type"] == "maskrcnn"
    assert kw["weights"] == run_dir / "weights" / "best.pt"
    info = kw["training_info"]
    assert TRAINING_INFO_KEYS <= set(info)
    assert info["vram_peak_mb_training"] == 1234.5
    assert info["best_epoch"] == 11
    assert "model_id" not in kw and "model_type_label" not in kw
    assert (run_dir / "weights" / "best.pt").read_bytes() == b"x"  # not overwritten


def test_maskrcnn_report_saves_in_memory_state_when_best_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        train_maskrcnn, "run_evaluation", _recorder(calls, run_dir / "metrics.json")
    )
    model = torch.nn.Linear(2, 2)

    train_maskrcnn.report_on_test(model=model, **_yolo_report_kwargs(run_dir))

    best = run_dir / "weights" / "best.pt"
    assert calls[0]["weights"] == best
    state = torch.load(best, weights_only=False)
    assert set(state["model_state_dict"]) == set(model.state_dict())


@pytest.mark.parametrize(
    "trainer", [train_yolo, train_parts, train_maskrcnn], ids=lambda m: m.__name__
)
def test_main_reports_via_evaluator_and_brackets_training_vram(trainer: ModuleType) -> None:
    src = inspect.getsource(trainer.main)
    assert "report_on_test(" in src
    assert "run_yolo_coco_eval" not in src
    assert "write_metrics_json" not in src
    assert 'eval_split="val"' not in src
    reset = src.index("reset_peak_vram")
    train = src.index("model.train(") if "model.train(" in src else src.index("train_one_epoch(")
    peak = src.index("peak_vram_mb")
    report = src.index("report_on_test(")
    assert reset < train < peak < report


def test_maskrcnn_main_resolves_processed_dir_from_dataset_config() -> None:
    src = inspect.getsource(train_maskrcnn.main)
    assert "resolve_processed_dir(args.dataset_config)" in src
    assert '"cardd"' not in src
