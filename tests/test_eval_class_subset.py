"""Class-subset and custom-split evaluation (CPU, offline, tmp_path only)."""

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
from autoassess.eval.coco_eval import build_coco_ground_truth, resolve_class_subset
from autoassess.eval.evaluate import load_eval_classes_from_yaml, run_evaluation
from autoassess.eval.yolo_eval import run_yolo_coco_eval

from .conftest import SyntheticCase

NAMES = ["dent", "scratch", "crack"]
POLY = np.array([[5, 5], [30, 5], [30, 30], [5, 30]], dtype=np.float32)


class _Arr:
    def __init__(self, v: list[float]) -> None:
        self._v = v

    def tolist(self) -> list[float]:
        return self._v


class _Yolo:
    """Per image: a perfect dent plus a high-confidence 'crack' false positive."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.model = torch.nn.Linear(3, 2)

    def predict(self, source: str, **kwargs: Any) -> list[Any]:  # noqa: ANN401
        self.calls.append(source)
        far = POLY + 30
        return [
            SimpleNamespace(
                orig_shape=(64, 64),
                masks=SimpleNamespace(xy=[POLY, far]),
                boxes=SimpleNamespace(cls=_Arr([0.0, 2.0]), conf=_Arr([0.9, 0.95])),
            )
        ]


def _make_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    for i in range(2):
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
            root / "images" / split / f"{split}{i}.jpg"
        )
        # dent GT matching the prediction exactly (5/64..30/64), plus a scratch GT
        (root / "labels" / split / f"{split}{i}.txt").write_text(
            "0 0.078125 0.078125 0.46875 0.078125 0.46875 0.46875 0.078125 0.46875\n"
            "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
        )


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    _make_split(root, "test")
    _make_split(root, "test_cardd")
    yaml_path = tmp_path / "ds.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: dent\n  1: scratch\n  2: crack\n"
    )
    return yaml_path


def _run(tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> Path:  # noqa: ANN401
    monkeypatch.setattr(evaluate, "_load_yolo", lambda weights: _Yolo())
    w = tmp_path / "runs" / "exp1" / "weights" / "best.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"x")
    return run_evaluation(
        weights=w, model_type="yolo", dataset_config=dataset,
        output_root=tmp_path / "out", latency_images=1, configure_logging=False, **kw,
    )


def test_custom_split_name(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _run(tmp_path, dataset, monkeypatch, split="test_cardd")
    assert path.parent.name == "exp1_eval_test_cardd"
    assert json.loads(path.read_text())["eval_split"] == "test_cardd"


def test_missing_custom_split_errors(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="test_nope"):
        _run(tmp_path, dataset, monkeypatch, split="test_nope")


def test_cli_split_not_restricted_to_train_val_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["autoassess-eval", "--weights", "w.pt", "--split", "test_x"])
    assert evaluate.parse_args().split == "test_x"


def test_subset_ignores_excluded_class_fp_and_gt(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = json.loads(_run(tmp_path, dataset, monkeypatch).read_text())
    sub_path = _run(tmp_path, dataset, monkeypatch, eval_classes=["dent"])
    sub = json.loads(sub_path.read_text())

    # crack FPs and missed scratch GT lower the full-set mAP; dent is perfect
    assert full["metrics"]["mask_map50"] < 1.0
    assert sub["metrics"]["mask_map50"] == pytest.approx(1.0)
    assert sub["metrics"]["box_map50"] == pytest.approx(1.0)
    assert sub["metrics"]["operating_point"]["mask"]["precision"] == pytest.approx(1.0)
    assert sub["metrics"]["operating_point"]["mask"]["recall"] == pytest.approx(1.0)
    assert sub["metrics"]["operating_point"]["mask"]["f1"] == pytest.approx(1.0)
    assert sub["eval_classes"] == ["dent"]
    assert full["eval_classes"] is None
    assert sub_path.parent.name == "exp1_eval_test_cls1"
    assert full["metrics"]["per_class"].keys() == set(NAMES)


def test_subset_per_class_lists_only_subset(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = json.loads(
        _run(tmp_path, dataset, monkeypatch, eval_classes=["scratch", "dent"]).read_text()
    )
    assert set(data["metrics"]["per_class"]) == {"dent", "scratch"}
    assert data["class_names"] == ["dent", "scratch"]  # dataset order, not request order
    # the crack prediction must not appear as a scratch/dent FP
    assert data["metrics"]["per_class"]["dent"]["fp"] == 0


def test_unknown_class_name_errors(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="wheel"):
        _run(tmp_path, dataset, monkeypatch, eval_classes=["dent", "wheel"])
    with pytest.raises(ValueError):
        resolve_class_subset(NAMES, [])


def test_subset_maps_by_name_not_index(tmp_path: Path) -> None:
    root = tmp_path / "p"
    _make_split(root, "test")
    gt = build_coco_ground_truth(
        root / "images" / "test", root / "labels" / "test", NAMES, class_subset=["scratch"]
    )
    assert [c["name"] for c in gt["categories"]] == ["scratch"]
    assert {a["category_id"] for a in gt["annotations"]} == {2}  # original id kept
    assert len(gt["images"]) == 2  # images stay, so their FPs still count
    res = run_yolo_coco_eval(
        _Yolo(), root / "images" / "test", root / "labels" / "test", NAMES,
        class_subset=["scratch"],
    )
    assert list(res["mask"]["per_class"]) == ["scratch"]


def test_eval_classes_from_yaml(tmp_path: Path) -> None:
    y = tmp_path / "unified.yaml"
    y.write_text("labelled_classes:\n  cardd: [dent, scratch]\n  vehide: [crack]\n")
    assert load_eval_classes_from_yaml(y, "cardd") == ["dent", "scratch"]
    with pytest.raises(ValueError, match="nope"):
        load_eval_classes_from_yaml(y, "nope")


def test_cli_eval_classes_from_end_to_end_and_mutual_exclusion(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    y = tmp_path / "unified.yaml"
    y.write_text("labelled_classes:\n  cardd: [dent, scratch]\n")
    base = ["autoassess-eval", "--weights", "w.pt"]
    monkeypatch.setattr(
        "sys.argv", [*base, "--eval-classes-from", str(y), "--eval-source", "cardd"]
    )
    assert evaluate.resolve_eval_classes(evaluate.parse_args()) == ["dent", "scratch"]
    monkeypatch.setattr(
        "sys.argv",
        [*base, "--eval-classes", "dent", "--eval-classes-from", str(y), "--eval-source", "cardd"],
    )
    with pytest.raises(SystemExit):
        evaluate.parse_args()
    monkeypatch.setattr("sys.argv", [*base, "--eval-classes", "dent, scratch"])
    assert evaluate.resolve_eval_classes(evaluate.parse_args()) == ["dent", "scratch"]
    monkeypatch.setattr("sys.argv", base)
    assert evaluate.resolve_eval_classes(evaluate.parse_args()) is None


def test_no_subset_matches_existing_expected_values(synthetic_case: SyntheticCase) -> None:
    from autoassess.eval.coco_eval import run_coco_eval

    a = run_coco_eval(synthetic_case.gt_dict, synthetic_case.detections, synthetic_case.class_names)
    assert a["operating_point"]["mask"]["per_class"]["dent"]["tp"] == 2
    assert a["operating_point"]["mask"]["per_class"]["scratch"]["fn"] == 2


def test_maskrcnn_with_subset_not_implemented(tmp_path: Path, dataset: Path) -> None:
    with pytest.raises(NotImplementedError, match="eval-classes"):
        run_evaluation(
            weights=tmp_path / "w.pt", model_type="maskrcnn", dataset_config=dataset,
            output_root=tmp_path / "out", eval_classes=["dent"], configure_logging=False,
        )
