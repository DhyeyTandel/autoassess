"""Fine-tune torchvision Mask R-CNN (ResNet50-FPN-v2) on CarDD — autoassess-train-maskrcnn.

Trained on the same converted splits (`data/processed/cardd/{images,labels}/<split>`)
and the same phone-photo augmentation policy (`autoassess.data.augmentations`) as
`train_yolo.py`, and writes `runs/<name>/metrics.json` in the same normalised
schema (`autoassess.eval.metrics`), so the two models are directly comparable
via `autoassess.eval.compare`.

Starts from COCO-pretrained weights (`MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`)
with the box/mask predictor heads replaced for CarDD's class count.

Usage
-----
    autoassess-train-maskrcnn \\
        --epochs 50 \\
        --batch  8 \\
        --imgsz  640 \\
        --device 0 \\
        --name   maskrcnn_v1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from autoassess.data.augmentations import (
    DEFAULT_AUGMENT_P,
    DEFAULT_PERSPECTIVE_SCALE,
    build_full_augmentation_pipeline,
)
from autoassess.eval.coco_eval import (
    load_class_names,
    mask_to_rle,
    run_coco_eval,
)
from autoassess.eval.metrics import (
    build_metrics_json,
    count_params,
    measure_inference_latency,
    peak_vram_mb,
    reset_peak_vram,
    write_metrics_json,
)
from autoassess.eval.scoring import SCORING_CONF_THRESHOLD, SCORING_MAX_DETS
from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

DEFAULT_PATIENCE = 20
MASK_MAP50_95_KEY = "mask_map50_95"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Mask R-CNN on CarDD with the same augmentation policy as YOLO."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (project/output paths, seed).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Path to the dataset YAML (class names, processed data root).")
    p.add_argument("--model", type=str, default="maskrcnn_resnet50_fpn_v2",
                   help="Model architecture id (default: maskrcnn_resnet50_fpn_v2). "
                        "Reserved for future variants; only this one is implemented.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs (default: 50).")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size, square (default: 640).")
    p.add_argument("--device", type=str, default="0",
                   help="Training device: GPU index (e.g. '0'), 'cpu', or 'mps' (default: 0).")
    p.add_argument("--name", type=str, default="maskrcnn_v1",
                   help="Run name; outputs land in runs/<name>/ (default: maskrcnn_v1).")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help=f"Early-stopping patience in epochs, measured on mask mAP50-95 "
                        f"(default: {DEFAULT_PATIENCE}).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from runs/<name>/weights/last.pt, if a resumable "
                        "(mid-training) checkpoint exists there.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed (default: config seed).")
    p.add_argument("--workers", type=int, default=None,
                   help="Dataloader workers (default: config data.num_workers).")
    p.add_argument("--augment-p", type=float, default=DEFAULT_AUGMENT_P,
                   help="Per-image probability of applying each phone-photo augmentation "
                        f"(default: {DEFAULT_AUGMENT_P}).")
    p.add_argument("--lr", type=float, default=0.005,
                   help="SGD learning rate for the fine-tuned heads (default: 0.005).")
    return p.parse_args()


class CarDDSegmentationDataset(Dataset):  # type: ignore[type-arg]
    """Reads the converted YOLO-seg label format (normalised polygons, one
    `.txt` per image) and yields (image, target) pairs in torchvision's
    Mask R-CNN training format: target = {boxes, labels, masks, image_id,
    area, iscrowd}. Category labels are 1-indexed (0 is background, matching
    torchvision's convention), i.e. YOLO class index + 1.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        imgsz: int,
        augment: bool,
        augment_p: float = DEFAULT_AUGMENT_P,
        perspective_scale: float = DEFAULT_PERSPECTIVE_SCALE,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.imgsz = imgsz
        self.label_files = sorted(labels_dir.glob("*.txt"))
        self.augment = augment
        self.transform = (
            build_full_augmentation_pipeline(augment_p, perspective_scale) if augment else None
        )

    def __len__(self) -> int:
        return len(self.label_files)

    def _find_image(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for label stem '{stem}' in {self.images_dir}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        label_path = self.label_files[idx]
        stem = label_path.stem
        image_path = self._find_image(stem)

        with Image.open(image_path) as im:
            image = np.array(im.convert("RGB"))
        height, width = image.shape[:2]

        masks: list[np.ndarray] = []
        class_indices: list[int] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_idx = int(parts[0])
            poly = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            poly[:, 0] *= width
            poly[:, 1] *= height

            mask = _polygon_to_binary_mask(poly, width, height)
            if mask.sum() == 0:
                continue
            masks.append(mask)
            class_indices.append(class_idx)

        bboxes = [_mask_to_bbox(m) for m in masks]

        if self.transform is not None and masks:
            augmented = self.transform(
                image=image, masks=masks, bboxes=bboxes, labels=class_indices
            )
            image = augmented["image"]
            masks = list(augmented["masks"])
            bboxes = list(augmented["bboxes"])
            class_indices = list(augmented["labels"])

        image = _resize_image(image, self.imgsz)
        scale_x = self.imgsz / width
        scale_y = self.imgsz / height
        masks = [_resize_mask(m, self.imgsz) for m in masks]

        valid_boxes, valid_masks, valid_labels = [], [], []
        for box, mask, cls in zip(bboxes, masks, class_indices, strict=True):
            x0, y0, x1, y1 = box
            x0, x1 = x0 * scale_x, x1 * scale_x
            y0, y1 = y0 * scale_y, y1 * scale_y
            if x1 - x0 < 1 or y1 - y0 < 1:
                # degenerate box after augmentation/resize — torchvision requires positive area
                continue
            valid_boxes.append([x0, y0, x1, y1])
            valid_masks.append(mask)
            valid_labels.append(cls + 1)  # 0 is background in torchvision's convention

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if valid_boxes:
            boxes_t = torch.tensor(valid_boxes, dtype=torch.float32)
            masks_t = torch.tensor(np.stack(valid_masks), dtype=torch.uint8)
            labels_t = torch.tensor(valid_labels, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, self.imgsz, self.imgsz), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": idx,
            "area": area_t,
            "iscrowd": torch.zeros((len(valid_boxes),), dtype=torch.int64),
            "file_name": image_path.name,
        }
        return image_tensor, target


def _polygon_to_binary_mask(poly_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_xy.astype(np.int32)], 1)
    return mask


def _mask_to_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_model(num_classes_with_background: int, pretrained: bool = True) -> torch.nn.Module:
    """Build Mask R-CNN with heads sized for `num_classes_with_background`.

    The model's own box_score_thresh / box_detections_per_img are set to the
    scoring values so torchvision's internal 0.05 filter does not truncate
    the PR curve. `pretrained=False` skips weight downloads (offline tests).
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if pretrained:
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_score_thresh=SCORING_CONF_THRESHOLD,
            box_detections_per_img=SCORING_MAX_DETS,
        )
    else:
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            box_score_thresh=SCORING_CONF_THRESHOLD,
            box_detections_per_img=SCORING_MAX_DETS,
        )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes_with_background)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes_with_background
    )
    return model  # type: ignore[no-any-return]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps")
    return torch.device(f"cuda:{device_arg}")


class ResumeError(Exception):
    """Raised when --resume is requested but no resumable checkpoint exists."""


def _resolve_resume_state(run_dir: Path) -> dict[str, Any] | None:
    """Return the saved training state dict if a mid-training checkpoint
    exists for this run, else None (nothing to resume — start fresh)."""
    last_ckpt = run_dir / "weights" / "last.pt"
    if not last_ckpt.exists():
        return None
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    if "optimizer_state_dict" not in state or state.get("epoch", -1) < 0:
        raise ResumeError(
            f"{last_ckpt} is not resumable: missing optimizer state or epoch marker. "
            "Only checkpoints saved mid-training (via periodic saves) can be resumed."
        )
    return state  # type: ignore[no-any-return]


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                   for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_predictions(
    model: torch.nn.Module,
    dataset: CarDDSegmentationDataset,
    device: torch.device,
    score_threshold: float = SCORING_CONF_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run the model over every image in `dataset` (eval mode, no augmentation)
    and return COCO-format detections plus a file_name -> dataset image_id map.

    Runs under `torch.inference_mode()` and keeps at most `SCORING_MAX_DETS`
    detections per image, by score."""
    model.eval()
    detections: list[dict[str, Any]] = []
    file_name_to_local_id: dict[str, int] = {}

    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        file_name_to_local_id[target["file_name"]] = idx

        with torch.inference_mode():
            output = model([image_tensor.to(device)])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # [N, 1, H, W] soft masks

        keep = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= score_threshold]
        for i in keep[:SCORING_MAX_DETS]:
            box, score, label, mask = boxes[i], scores[i], labels[i], masks[i]
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            rle = mask_to_rle(binary_mask)
            x0, y0, x1, y1 = box
            detections.append({
                "image_id": idx,
                "category_id": int(label),
                "segmentation": rle,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "score": float(score),
            })

    return detections, file_name_to_local_id


def evaluate_split(
    model: torch.nn.Module,
    dataset: CarDDSegmentationDataset,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Build COCO ground truth directly from the eval dataset (no augmentation,
    so it matches the images actually scored) and run pycocotools eval against
    the model's predictions on the same image_id numbering."""
    detections, _ = run_predictions(model, dataset, device)

    images_meta = []
    annotations = []
    ann_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        images_meta.append({
            "id": idx,
            "file_name": target["file_name"],
            "width": dataset.imgsz,
            "height": dataset.imgsz,
        })
        boxes = target["boxes"].numpy()
        masks = target["masks"].numpy()
        labels = target["labels"].numpy()
        for box, mask, label in zip(boxes, masks, labels, strict=True):
            rle = mask_to_rle(mask)
            x0, y0, x1, y1 = box
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(label),
                "segmentation": rle,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float((x1 - x0) * (y1 - y0)),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name, "supercategory": "damage"}
                  for i, name in enumerate(class_names)]
    coco_gt_dict = {"images": images_meta, "annotations": annotations, "categories": categories}

    return run_coco_eval(coco_gt_dict, detections, class_names)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / args.name
    setup_run_logging(run_dir)
    log.info("Starting Mask R-CNN training run: %s", args.name)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)
    log.info("Seed set to %d", seed)

    class_names = load_class_names(args.dataset_config)
    num_classes_with_background = len(class_names) + 1

    processed_dir = Path(cfg.data.processed_dir) / "cardd"
    workers = args.workers if args.workers is not None else int(cfg.data.get("num_workers", 4))

    train_dataset = CarDDSegmentationDataset(
        processed_dir / "images" / "train", processed_dir / "labels" / "train",
        imgsz=args.imgsz, augment=True, augment_p=args.augment_p,
    )
    val_dataset = CarDDSegmentationDataset(
        processed_dir / "images" / "val", processed_dir / "labels" / "val",
        imgsz=args.imgsz, augment=False,
    )
    log.info("Train images: %d, Val images: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=workers, collate_fn=collate_fn,
    )

    device = resolve_device(args.device)
    model = build_model(num_classes_with_background).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    start_epoch = 0
    if args.resume:
        state = _resolve_resume_state(run_dir)
        if state is not None:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state["epoch"] + 1
            log.info("Resuming from checkpoint: epoch %d", start_epoch)
        else:
            log.warning("--resume was passed but no checkpoint found; starting fresh.")

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_mask_map50_95 = float("-inf")
    best_epoch = 0
    no_improve = 0
    early_stopped = False
    epoch_times: list[float] = []

    reset_peak_vram(device)
    train_start = time.monotonic()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        mean_loss = train_one_epoch(model, optimizer, train_loader, device)
        epoch_time = time.monotonic() - epoch_start
        epoch_times.append(epoch_time)

        eval_result = evaluate_split(model, val_dataset, device, class_names)
        current_mask_map50_95 = eval_result["mask"]["map50_95"]
        log.info(
            "Epoch %d/%d | loss=%.4f | mask_mAP50-95=%.4f | %.1fs",
            epoch + 1, args.epochs, mean_loss, current_mask_map50_95, epoch_time,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            weights_dir / "last.pt",
        )

        if current_mask_map50_95 > best_mask_map50_95:
            best_mask_map50_95 = current_mask_map50_95
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, weights_dir / "best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            log.info(
                "Early stopping: mask mAP50-95 has not improved for %d epochs "
                "(best=%.4f @ epoch %d).",
                args.patience, best_mask_map50_95, best_epoch,
            )
            early_stopped = True
            break

    total_wall_time = time.monotonic() - train_start

    # final evaluation with the best checkpoint, mirroring YOLO's own final re-validation
    best_ckpt = weights_dir / "best.pt"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
    final_eval = evaluate_split(model, val_dataset, device, class_names)

    latency = measure_inference_latency(
        lambda img: model([img.to(device)]),
        [val_dataset[i][0] for i in range(min(len(val_dataset), 30))],
    )
    model_info = {
        **count_params(model),
        "vram_peak_mb": peak_vram_mb(device),
    }

    payload = build_metrics_json(
        run_name=args.name,
        model_type="maskrcnn",
        model=args.model,
        class_names=class_names,
        epochs_requested=args.epochs,
        epochs_run_this_invocation=len(epoch_times),
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        wall_time_seconds_total=total_wall_time,
        epoch_wall_times_seconds=epoch_times,
        box_metrics=final_eval["box"],
        mask_metrics=final_eval["mask"],
        mask_iou=final_eval["mask_iou_mean"],
        inference=latency,
        model_info=model_info,
    )
    metrics_path = write_metrics_json(run_dir, payload)
    log.info("Training complete. Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
