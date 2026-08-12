"""Train a learned severity classifier — autoassess-train-severity.

A two-branch ResNet-18 classifier: one branch sees the full image (global
context — how much of the frame the vehicle occupies, overall scene), the
other sees a crop of the damaged region (the union of all detected damage
boxes, padded — same region the heuristic in `autoassess.models.severity`
uses as its "vehicle region" proxy). Both branches' 512-d ResNet-18 features
are concatenated and fed to a 3-way minor/moderate/severe head.

Labels are **not** ground truth — CarDD has no severity annotation. Every
label used here is produced by the heuristic in `autoassess.models.severity`
from the same converted damage-instance labels. This is weak supervision,
and it is circular in a specific, important way — see "On circularity" below
and the fuller discussion in `autoassess.eval.severity_eval`.

On circularity
---------------
Training a classifier on labels derived from a fixed formula and then asking
"how accurate is the classifier" is answering the wrong question. The
classifier cannot learn anything about real damage severity that isn't
already encoded in the heuristic's class weights and thresholds — at best it
learns a smoothed, image-feature-based *approximation* of the same formula,
plus whatever noise the crop/resize/backbone pipeline adds on top. A perfect
learned classifier, by construction, would just reproduce the heuristic
exactly; any measured "accuracy" against heuristic labels is really measuring
how well a CNN can approximate a closed-form function of mask areas it never
gets to see directly (it only sees pixels), which is a strange and roundabout
thing to optimize for.

What this training run is actually useful for, honestly: sanity-checking
that a two-branch CNN *can* learn to approximate the heuristic's output from
pixels alone (a necessary but not sufficient condition for the architecture
to eventually work), and producing an artifact to demonstrate the evaluation
and comparison tooling. It is not evidence that either the heuristic or the
learned model grades real-world damage severity correctly.

The fix is real labels: a sample of claim photos graded by human adjusters
(or actual claim payout bands, if available) with class balance checked
across minor/moderate/severe, used as the held-out evaluation set at minimum
and ideally as training labels too. Until that exists, treat every number
this script or `severity_eval.py` produces as "agreement with the heuristic,"
never as "severity grading accuracy."

Usage
-----
    autoassess-train-severity \\
        --epochs 30 \\
        --batch  16 \\
        --imgsz  224 \\
        --device 0 \\
        --name   severity_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autoassess.eval.coco_eval import load_class_names
from autoassess.eval.metrics import count_params, peak_vram_mb, reset_peak_vram
from autoassess.models.severity import (
    SEVERITY_LABELS,
    DamageInstance,
    SeverityConfig,
    grade_image,
)
from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

DEFAULT_PATIENCE = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a two-branch ResNet-18 severity classifier on heuristic labels "
                    "(weak supervision — see module docstring for the circularity caveat)."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (project/output paths, seed).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Path to the dataset YAML (class names, processed data root).")
    p.add_argument("--severity-config", type=Path, default=Path("configs/severity.yaml"),
                   help="Path to the heuristic severity config (class weights, thresholds).")
    p.add_argument("--model", type=str, default="resnet18-two-branch",
                   help="Model architecture id (default: resnet18-two-branch).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of training epochs (default: 30).")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16).")
    p.add_argument("--imgsz", type=int, default=224,
                   help="Input size per branch, square (default: 224, ResNet's native size).")
    p.add_argument("--device", type=str, default="0",
                   help="Training device: GPU index (e.g. '0'), 'cpu', or 'mps' (default: 0).")
    p.add_argument("--name", type=str, default="severity_v1",
                   help="Run name; outputs land in runs/<name>/ (default: severity_v1).")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help=f"Early-stopping patience in epochs, measured on val accuracy "
                        f"against heuristic labels (default: {DEFAULT_PATIENCE}).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed (default: config seed).")
    p.add_argument("--workers", type=int, default=None,
                   help="Dataloader workers (default: config data.num_workers).")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate (default: 1e-4).")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze both ResNet-18 backbones and only train the fusion head "
                        "(faster, fewer params to overfit on a small dataset).")
    return p.parse_args()


def _polygon_mask_area(poly_norm: list[float], width: int, height: int) -> float:
    """Shoelace-formula area of a normalised flat polygon, in pixels^2.
    Avoids rasterising a mask just to sum pixels — exact for the polygon
    label format, and matches what `autoassess.data.convert` wrote."""
    pts = np.array(poly_norm, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _polygon_bbox(
    poly_norm: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    pts = np.array(poly_norm, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def load_instances_from_label_file(
    label_path: Path, width: int, height: int, class_names: list[str]
) -> list[DamageInstance]:
    """Parse a converted YOLO-seg `.txt` label file into DamageInstance
    objects with exact polygon-derived area and bbox (no rasterisation)."""
    instances = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_idx = int(parts[0])
        poly = [float(v) for v in parts[1:]]
        if len(poly) < 6:
            continue
        area = _polygon_mask_area(poly, width, height)
        bbox = _polygon_bbox(poly, width, height)
        instances.append(
            DamageInstance(class_name=class_names[class_idx], mask_area=area, bbox=bbox)
        )
    return instances


def union_bbox(
    instances: list[DamageInstance], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Same padded union-box construction as the heuristic's vehicle-region
    proxy (`autoassess.models.severity.vehicle_region_area`), returned as
    integer pixel coordinates for cropping. Falls back to the full image
    when there are no instances."""
    if not instances:
        return 0, 0, width, height

    x0 = min(inst.bbox[0] for inst in instances)
    y0 = min(inst.bbox[1] for inst in instances)
    x1 = max(inst.bbox[2] for inst in instances)
    y1 = max(inst.bbox[3] for inst in instances)
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding

    px0 = max(0, int(x0 - pad_x))
    py0 = max(0, int(y0 - pad_y))
    px1 = min(width, int(x1 + pad_x))
    py1 = min(height, int(y1 + pad_y))
    return px0, py0, max(px1, px0 + 1), max(py1, py0 + 1)


class SeverityDataset(Dataset):  # type: ignore[type-arg]
    """Yields (full_image_tensor, crop_tensor, label_index) triples.

    The label is computed on the fly from the heuristic (`grade_image`) —
    there is no separate label file. See module docstring: this is the
    weak-supervision signal, not ground truth.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_names: list[str],
        severity_config: SeverityConfig,
        imgsz: int,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.severity_config = severity_config
        self.imgsz = imgsz
        self.label_files = sorted(labels_dir.glob("*.txt"))
        self.label_to_index = {name: i for i, name in enumerate(SEVERITY_LABELS)}

    def __len__(self) -> int:
        return len(self.label_files)

    def _find_image(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for label stem '{stem}' in {self.images_dir}")

    def severity_label_for_index(self, idx: int) -> str:
        """Compute just the heuristic label (no image I/O) — used by the
        stratified splitter and by severity_eval.py without paying for a
        full image load."""
        label_path = self.label_files[idx]
        image_path = self._find_image(label_path.stem)
        with Image.open(image_path) as im:
            width, height = im.size
        instances = load_instances_from_label_file(label_path, width, height, self.class_names)
        return grade_image(instances, width, height, self.severity_config)["label"]  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        label_path = self.label_files[idx]
        image_path = self._find_image(label_path.stem)

        with Image.open(image_path) as im:
            image = np.array(im.convert("RGB"))
        height, width = image.shape[:2]

        instances = load_instances_from_label_file(label_path, width, height, self.class_names)
        result = grade_image(instances, width, height, self.severity_config)
        label_name = result["label"]
        label_idx = self.label_to_index[label_name]

        x0, y0, x1, y1 = union_bbox(
            instances, width, height, self.severity_config.vehicle_region_padding
        )
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            crop = image

        full_tensor = _to_tensor(image, self.imgsz)
        crop_tensor = _to_tensor(crop, self.imgsz)
        return full_tensor, crop_tensor, label_idx, image_path.name


def _to_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    import cv2

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class TwoBranchSeverityNet(nn.Module):
    """Two ResNet-18 backbones (full image, damaged-region crop) with their
    512-d penultimate features concatenated into a 3-way severity head."""

    def __init__(self, freeze_backbone: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        full_backbone = resnet18(weights=weights)
        crop_backbone = resnet18(weights=weights)
        self.full_branch = nn.Sequential(*list(full_backbone.children())[:-1])  # drop fc
        self.crop_branch = nn.Sequential(*list(crop_backbone.children())[:-1])

        if freeze_backbone:
            for param in self.full_branch.parameters():
                param.requires_grad_(False)
            for param in self.crop_branch.parameters():
                param.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, len(SEVERITY_LABELS)),
        )

    def forward(self, full_image: torch.Tensor, crop: torch.Tensor) -> torch.Tensor:
        full_feat = torch.flatten(self.full_branch(full_image), 1)
        crop_feat = torch.flatten(self.crop_branch(crop), 1)
        combined = torch.cat([full_feat, crop_feat], dim=1)
        return self.head(combined)  # type: ignore[no-any-return]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps")
    return torch.device(f"cuda:{device_arg}")


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for full_img, crop, labels, _ in loader:
        full_img, crop, labels = full_img.to(device), crop.to(device), labels.to(device)
        logits = model(full_img, crop)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device  # type: ignore[type-arg]
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_file_names: list[str] = []

    for full_img, crop, labels, file_names in loader:
        full_img, crop = full_img.to(device), crop.to(device)
        logits = model(full_img, crop)
        preds = logits.argmax(dim=1).cpu()

        correct += int((preds == labels).sum())
        total += labels.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_file_names.extend(file_names)

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "file_names": all_file_names,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / args.name
    setup_run_logging(run_dir)
    log.info("Starting severity classifier training run: %s", args.name)
    log.warning(
        "Labels are heuristic-derived weak supervision, not ground truth — "
        "see train_severity.py module docstring for the circularity caveat."
    )

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)
    log.info("Seed set to %d", seed)

    class_names = load_class_names(args.dataset_config)
    severity_config = SeverityConfig.load(args.severity_config)

    processed_dir = Path(cfg.data.processed_dir) / "cardd"
    workers = args.workers if args.workers is not None else int(cfg.data.get("num_workers", 4))

    train_dataset = SeverityDataset(
        processed_dir / "images" / "train", processed_dir / "labels" / "train",
        class_names, severity_config, imgsz=args.imgsz,
    )
    val_dataset = SeverityDataset(
        processed_dir / "images" / "val", processed_dir / "labels" / "val",
        class_names, severity_config, imgsz=args.imgsz,
    )
    log.info("Train images: %d, Val images: %d", len(train_dataset), len(val_dataset))

    label_counts = {label: 0 for label in SEVERITY_LABELS}
    for i in range(len(train_dataset)):
        label_counts[train_dataset.severity_label_for_index(i)] += 1
    log.info("Train heuristic label distribution: %s", label_counts)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=workers,
    )

    device = resolve_device(args.device)
    model = TwoBranchSeverityNet(freeze_backbone=args.freeze_backbone).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = float("-inf")
    best_epoch = 0
    no_improve = 0
    early_stopped = False
    epoch_times: list[float] = []

    reset_peak_vram(device)
    train_start = time.monotonic()

    final_eval: dict[str, Any] = {
        "accuracy": 0.0, "predictions": [], "labels": [], "file_names": [],
    }
    for epoch in range(args.epochs):
        epoch_start = time.monotonic()
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, criterion)
        epoch_time = time.monotonic() - epoch_start
        epoch_times.append(epoch_time)

        final_eval = evaluate(model, val_loader, device)
        accuracy = final_eval["accuracy"]
        log.info(
            "Epoch %d/%d | loss=%.4f | val_accuracy_vs_heuristic=%.4f | %.1fs",
            epoch + 1, args.epochs, mean_loss, accuracy, epoch_time,
        )

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict()},
            weights_dir / "last.pt",
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, weights_dir / "best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            log.info(
                "Early stopping: val accuracy vs heuristic has not improved for %d epochs "
                "(best=%.4f @ epoch %d).",
                args.patience, best_accuracy, best_epoch,
            )
            early_stopped = True
            break

    total_wall_time = time.monotonic() - train_start

    best_ckpt = weights_dir / "best.pt"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        final_eval = evaluate(model, val_loader, device)

    model_info = {**count_params(model), "vram_peak_mb": peak_vram_mb(device)}

    payload = {
        "run_name": args.name,
        "model_type": "severity-learned",
        "model": args.model,
        "labels": SEVERITY_LABELS,
        "epochs_requested": args.epochs,
        "epochs_run_this_invocation": len(epoch_times),
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "wall_time_seconds_total": total_wall_time,
        "wall_time_seconds_per_epoch_mean": (
            sum(epoch_times) / len(epoch_times) if epoch_times else None
        ),
        "epoch_wall_times_seconds": epoch_times,
        "train_label_distribution": label_counts,
        "val_accuracy_vs_heuristic": final_eval["accuracy"],
        "val_predictions": final_eval["predictions"],
        "val_labels": final_eval["labels"],
        "val_file_names": final_eval["file_names"],
        "model_info": model_info,
        "circularity_warning": (
            "val_accuracy_vs_heuristic measures agreement with this project's own "
            "hand-authored severity heuristic, not real-world grading accuracy — "
            "see train_severity.py module docstring."
        ),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Training complete. Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
