"""Tests for the standalone evaluator (T5). CPU, offline, tmp_path only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from autoassess.eval import evaluate
from autoassess.eval.evaluate import detect_model_type, run_evaluation
from autoassess.eval.scoring import OPERATING_CONF, SCORING_CONF_THRESHOLD

CLASS_NAMES = ["dent", "scratch"]


class _Arr:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeYolo:
    """Records every predict call; returns one box-shaped mask per call."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.events = events if events is not None else []
        self.model = torch.nn.Linear(3, 2)
        for p in self.model.parameters():
            p.requires_grad_(False)  # mimic Ultralytics' reloaded checkpoint

    def predict(self, source: str, **kwargs: Any) -> list[Any]:  # noqa: ANN401
        self.events.append("predict")
        self.calls.append((source, kwargs))
        poly = np.array([[5, 5], [30, 5], [30, 30], [5, 30]], dtype=np.float32)
        return [
            SimpleNamespace(
                orig_shape=(64, 64),
                masks=SimpleNamespace(xy=[poly]),
                boxes=SimpleNamespace(cls=_Arr([0.0]), conf=_Arr([0.9])),
            )
        ]


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Tiny processed dataset (val + test, 2 images each) and its YAML."""
    root = tmp_path / "processed"
    for split in ("val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
                root / "images" / split / f"{split}{i}.jpg"
            )
            (root / "labels" / split / f"{split}{i}.txt").write_text(
                "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n"
            )
    yaml_path = tmp_path / "ds.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: dent\n  1: scratch\n"
    )
    return yaml_path


def _weights(tmp_path: Path, name: str = "exp1") -> Path:
    w = tmp_path / "runs" / name / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"not a real checkpoint")
    return w


def _patch_yolo(monkeypatch: pytest.MonkeyPatch, fake: _FakeYolo) -> None:
    monkeypatch.setattr(evaluate, "_load_yolo", lambda weights: fake)


def _run(tmp_path: Path, dataset: Path, **kw: Any) -> Path:  # noqa: ANN401
    return run_evaluation(
        weights=_weights(tmp_path) if "weights" not in kw else kw.pop("weights"),
        model_type="yolo",
        dataset_config=dataset,
        output_root=tmp_path / "out",
        latency_images=2,
        **kw,
    )


def test_yolo_writes_v2_metrics_for_test_split(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeYolo()
    _patch_yolo(monkeypatch, fake)

    path = _run(tmp_path, dataset)

    assert path == tmp_path / "out" / "exp1_eval_test" / "metrics.json"
    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    assert data["eval_split"] == "test"
    assert data["run_name"] == "exp1"
    assert data["model_type"] == "yolov8-seg"
    assert data["class_names"] == CLASS_NAMES
    assert data["metrics"]["operating_point"]["scoring_conf"] == SCORING_CONF_THRESHOLD
    assert data["model_info"]["vram_scope"] == "inference"
    assert data["model_info"]["vram_peak_mb"] is None  # CPU
    assert data["model_info"]["device_name"] == "cpu"
    assert data["model_info"]["params_trainable"] == data["model_info"]["params_total"] > 0
    assert data["inference"]["n_images"] == 2

    sources = [Path(s) for s, _ in fake.calls]
    assert sources
    assert all(s.parent.name == "test" and s.parent.parent.name == "images" for s in sources)
    assert not any("val" in s.name for s in sources)
    scoring = [kw for _, kw in fake.calls if kw["conf"] == SCORING_CONF_THRESHOLD]
    assert len(scoring) == 2  # one scoring predict per test image
    assert {kw["conf"] for _, kw in fake.calls} == {SCORING_CONF_THRESHOLD, OPERATING_CONF}


def test_vram_reset_happens_before_first_predict(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_yolo(monkeypatch, _FakeYolo(events))
    monkeypatch.setattr(evaluate, "reset_peak_vram", lambda device: events.append("reset"))

    _run(tmp_path, dataset)

    assert events[0] == "reset"
    assert events.count("reset") == 1
    assert "predict" in events


def test_split_val_writes_val_dir(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeYolo()
    _patch_yolo(monkeypatch, fake)

    path = _run(tmp_path, dataset, split="val")

    assert path == tmp_path / "out" / "exp1_eval_val" / "metrics.json"
    assert json.loads(path.read_text())["eval_split"] == "val"
    assert all(Path(s).parent.name == "val" for s, _ in fake.calls)


def test_explicit_run_name(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_yolo(monkeypatch, _FakeYolo())
    path = _run(tmp_path, dataset, run_name="custom")
    assert path.parent.name == "custom_eval_test"
    assert json.loads(path.read_text())["run_name"] == "custom"


TRAIN_FIELDS = {
    "epochs_requested": 50,
    "epochs_run_this_invocation": 31,
    "batch": 8,
    "patience": 20,
    "best_epoch": 11,
    "early_stopped": True,
    "wall_time_seconds_total": 123.5,
    "epoch_wall_times_seconds": [2.0, 4.0],
}


def test_training_fields_none_when_no_sibling(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_yolo(monkeypatch, _FakeYolo())
    data = json.loads(_run(tmp_path, dataset).read_text())
    for key in TRAIN_FIELDS:
        if key != "epoch_wall_times_seconds":
            assert data[key] is None
    assert data["wall_time_seconds_per_epoch_mean"] is None
    assert data["epoch_wall_times_seconds"] in (None, [])


def test_training_fields_copied_from_sibling_metrics(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_yolo(monkeypatch, _FakeYolo())
    weights = _weights(tmp_path)
    sibling = weights.parent.parent / "metrics.json"
    sibling.write_text(json.dumps({**TRAIN_FIELDS, "metrics": {"box_map50": 0.99}}))
    before = sibling.read_text()

    data = json.loads(_run(tmp_path, dataset, weights=weights).read_text())

    for key, value in TRAIN_FIELDS.items():
        assert data[key] == value
    assert data["wall_time_seconds_per_epoch_mean"] == pytest.approx(3.0)
    assert data["metrics"]["box_map50"] != 0.99  # metrics are re-scored, not copied
    assert sibling.read_text() == before  # read-only


def test_training_info_overrides_sibling(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_yolo(monkeypatch, _FakeYolo())
    weights = _weights(tmp_path)
    (weights.parent.parent / "metrics.json").write_text(json.dumps(TRAIN_FIELDS))

    data = json.loads(
        _run(
            tmp_path,
            dataset,
            weights=weights,
            training_info={
                "epochs_requested": 7,
                "best_epoch": 3,
                "vram_peak_mb_training": 1234.5,
            },
        ).read_text()
    )

    assert data["epochs_requested"] == 7
    assert data["best_epoch"] == 3
    assert data["model_info"]["vram_peak_mb_training"] == 1234.5


def test_detect_model_type(tmp_path: Path) -> None:
    mrcnn = tmp_path / "m.pt"
    torch.save({"model_state_dict": {}}, mrcnn)
    other = tmp_path / "y.pt"
    torch.save({"model": "whatever"}, other)
    assert detect_model_type(mrcnn) == "maskrcnn"
    assert detect_model_type(other) == "yolo"


def test_missing_split_dir_names_path(tmp_path: Path, dataset: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        run_evaluation(
            weights=_weights(tmp_path),
            model_type="yolo",
            dataset_config=dataset,
            split="train",
            output_root=tmp_path / "out",
        )
    assert str(tmp_path / "processed" / "images" / "train") in str(exc.value)


def test_empty_split_dir_raises(tmp_path: Path, dataset: Path) -> None:
    empty = tmp_path / "processed" / "images" / "train"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="train"):
        run_evaluation(
            weights=_weights(tmp_path),
            model_type="yolo",
            dataset_config=dataset,
            split="train",
            output_root=tmp_path / "out",
        )


def test_maskrcnn_end_to_end(tmp_path: Path, dataset: Path) -> None:
    from autoassess.train.train_maskrcnn import build_model

    torch.manual_seed(0)
    model = build_model(len(CLASS_NAMES) + 1, pretrained=False)
    weights = tmp_path / "runs" / "mrcnn" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict()}, weights)

    path = run_evaluation(
        weights=weights,
        model_type="maskrcnn",
        dataset_config=dataset,
        split="test",
        output_root=tmp_path / "out",
        device="cpu",
        imgsz=64,
        latency_images=1,
    )

    assert path == tmp_path / "out" / "mrcnn_eval_test" / "metrics.json"
    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    assert data["model_type"] == "maskrcnn"
    assert data["eval_split"] == "test"
    assert set(data["metrics"]["per_class"]) == set(CLASS_NAMES)
    assert data["inference"]["n_images"] == 1
    assert data["model_info"]["vram_scope"] == "inference"
    assert data["model_info"]["params_total"] > 0
