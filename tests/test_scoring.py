"""Tests for full-PR-curve scoring of both models (T2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from autoassess.eval.coco_eval import build_coco_ground_truth
from autoassess.eval.scoring import (
    OPERATING_CONF,
    SCORING_CONF_THRESHOLD,
    SCORING_MAX_DETS,
)
from autoassess.eval.yolo_eval import collect_yolo_detections, measure_yolo_latency
from autoassess.train.train_maskrcnn import build_model, run_predictions

CLASS_NAMES = ["dent", "scratch"]


class _Arr:
    """Minimal stand-in for a torch tensor exposing `.tolist()`."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeYolo:
    def __init__(self, confs: list[float]) -> None:
        self.confs = confs
        self.calls: list[dict[str, Any]] = []

    def predict(self, source: str, **kwargs: Any) -> list[Any]:  # noqa: ANN401
        self.calls.append(kwargs)
        polys = [
            np.array([[5, 5], [30, 5], [30, 30], [5, 30]], dtype=np.float32) + 10 * i
            for i in range(len(self.confs))
        ]
        result = SimpleNamespace(
            orig_shape=(64, 64),
            masks=SimpleNamespace(xy=polys),
            boxes=SimpleNamespace(
                cls=_Arr([0.0] * len(self.confs)), conf=_Arr(self.confs)
            ),
        )
        return [result]


def _make_split(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    labels = root / "labels"
    images.mkdir()
    labels.mkdir()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(images / "a.jpg")
    (labels / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    return images, labels


def test_collect_yolo_detections_uses_scoring_threshold(tmp_path: Path) -> None:
    images, labels = _make_split(tmp_path)
    gt = build_coco_ground_truth(images, labels, CLASS_NAMES)
    model = _FakeYolo([0.9, 0.1, 0.002])

    dets = collect_yolo_detections(model, images, gt, imgsz=320)

    assert len(model.calls) == 1
    kwargs = model.calls[0]
    assert kwargs["conf"] == SCORING_CONF_THRESHOLD == 0.001
    assert kwargs["max_det"] == SCORING_MAX_DETS == 100
    assert kwargs["imgsz"] == 320
    scores = sorted(d["score"] for d in dets)
    assert scores == pytest.approx([0.002, 0.1, 0.9])
    assert any(s < OPERATING_CONF for s in scores)


def test_latency_uses_operating_conf(tmp_path: Path) -> None:
    images, _ = _make_split(tmp_path)
    model = _FakeYolo([0.9])
    measure_yolo_latency(model, images, imgsz=320, n_images=1)
    assert model.calls
    assert all(c["conf"] == OPERATING_CONF == 0.25 for c in model.calls)


def test_build_model_scoring_thresholds() -> None:
    model = build_model(num_classes_with_background=3, pretrained=False)
    assert model.roi_heads.score_thresh == 0.001  # type: ignore[union-attr]
    assert model.roi_heads.detections_per_img == 100  # type: ignore[union-attr]


class _StubDataset:
    imgsz = 32

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.zeros(3, 32, 32), {"file_name": "a.jpg"}


class _FakeMaskRCNN(torch.nn.Module):
    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.scores = scores
        self.inference_flags: list[bool] = []

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        self.inference_flags.append(torch.is_inference_mode_enabled())
        n = len(self.scores)
        masks = torch.ones(n, 1, 32, 32)
        return [{
            "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]] * n),
            "scores": torch.tensor(self.scores),
            "labels": torch.ones(n, dtype=torch.int64),
            "masks": masks,
        }]


def test_run_predictions_keeps_low_scores_and_uses_inference_mode() -> None:
    model = _FakeMaskRCNN([0.9, 0.1, 0.0005])
    dets, _ = run_predictions(model, _StubDataset(), torch.device("cpu"))  # type: ignore[arg-type]
    scores = sorted(d["score"] for d in dets)
    assert scores == pytest.approx([0.1, 0.9])
    assert model.inference_flags == [True]


def test_run_predictions_caps_detections_per_image() -> None:
    n = SCORING_MAX_DETS + 20
    model = _FakeMaskRCNN([0.5 + i * 0.001 for i in range(n)])
    dets, _ = run_predictions(model, _StubDataset(), torch.device("cpu"))  # type: ignore[arg-type]
    assert len(dets) == SCORING_MAX_DETS
    assert min(d["score"] for d in dets) == pytest.approx(0.5 + 20 * 0.001)


@pytest.mark.slow
def test_real_yolo_emits_sub_operating_conf_detections(tmp_path: Path) -> None:
    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[1] / "yolov8n-seg.pt"
    model = YOLO(str(weights))
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    Image.fromarray(noise).save(images / "noise.jpg")
    (labels / "noise.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    gt = build_coco_ground_truth(images, labels, CLASS_NAMES)

    dets = collect_yolo_detections(model, images, gt, imgsz=640)

    assert any(d["score"] < OPERATING_CONF for d in dets), dets[:3]
