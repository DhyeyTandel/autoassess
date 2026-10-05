"""Train YOLOv8-seg on CarDD via Ultralytics — autoassess-train-yolo.

Wraps ``ultralytics.YOLO.train`` with:
  - a phone-photo Albumentations pipeline (illumination shift, motion blur,
    JPEG compression, random shadow) injected via Ultralytics' documented
    ``hyp.augmentations`` hook, plus native perspective warp
  - resume support (``--resume``) for genuinely interrupted runs (crash,
    Ctrl-C, preemption). Ultralytics strips optimizer state once a run
    completes normally, so a *finished* run cannot be resumed further —
    ``--resume`` against a completed checkpoint fails loudly rather than
    silently retraining from scratch.
  - early stopping on mask mAP50-95 specifically, since Ultralytics' own
    ``patience``/stopper uses a combined box+mask fitness score, not mask
    mAP50-95 alone
  - a ``runs/<name>/metrics.json`` written at the end of training by scoring the
    selected checkpoint on the *test* split via ``autoassess.eval.evaluate``
    (val is used only for checkpoint selection and early stopping)

Usage
-----
    autoassess-train-yolo \\
        --model  yolov8s-seg.pt \\
        --epochs 50 \\
        --batch  8 \\
        --imgsz  640 \\
        --device 0 \\
        --name   yolov8_seg_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from autoassess.data.augmentations import (
    DEFAULT_AUGMENT_P,
    DEFAULT_PERSPECTIVE_SCALE,
    build_phone_photo_augmentations,
)
from autoassess.eval.evaluate import run_evaluation
from autoassess.eval.yolo_eval import (
    peak_vram_mb_for_yolo,
    reset_peak_vram_for_yolo,
    resolve_dataset_yaml_for_ultralytics,
)
from autoassess.utils.config import load_config, resolve_paths
from autoassess.utils.logging import get_logger, setup_run_logging
from autoassess.utils.seed import seed_everything

log = get_logger(__name__)

MASK_MAP50_95_KEY = "metrics/mAP50-95(M)"
DEFAULT_PATIENCE = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8-seg on CarDD with a phone-photo augmentation pipeline."
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                   help="Path to base YAML config (project/output paths, seed).")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/cardd.yaml"),
                   help="Path to the Ultralytics dataset YAML (images/labels/class names).")
    p.add_argument("--model", type=str, default="yolov8s-seg.pt",
                   help="Model checkpoint or config to start from (default: yolov8s-seg.pt).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs (default: 50).")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size, square (default: 640).")
    p.add_argument("--device", type=str, default="0",
                   help="Training device: GPU index (e.g. '0') or 'cpu' (default: 0).")
    p.add_argument("--name", type=str, default="yolov8_seg_v1",
                   help="Run name; outputs land in runs/<name>/ (default: yolov8_seg_v1).")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help=f"Early-stopping patience in epochs, measured on mask mAP50-95 "
                        f"(default: {DEFAULT_PATIENCE}).")
    p.add_argument("--resume", action="store_true",
                   help="Resume the most recent checkpoint for --name, if one exists.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override project.seed (default: config seed).")
    p.add_argument("--workers", type=int, default=None,
                   help="Dataloader workers (default: config data.num_workers).")
    p.add_argument("--augment-p", type=float, default=DEFAULT_AUGMENT_P,
                   help="Per-image probability of applying each phone-photo augmentation "
                        f"(default: {DEFAULT_AUGMENT_P}).")
    return p.parse_args()


def build_early_stopping_callback(patience: int, total_epochs: int) -> tuple[Any, dict[str, Any]]:
    """Return an `on_fit_epoch_end` callback that stops training when mask
    mAP50-95 hasn't improved for `patience` epochs, overriding Ultralytics'
    own combined box+mask fitness stopper (accessible via the closure's
    `state` dict for logging after training ends).

    Ultralytics fires `on_fit_epoch_end` one extra time after training ends
    (in `final_eval`, with `trainer.epoch` bumped to `epochs + 1` as a
    sentinel) to re-log metrics from the re-validated best checkpoint. That
    call is excluded from `no_improve` bookkeeping — it doesn't represent an
    additional real epoch — but its (generally more accurate) score can
    still update `best`, just labelled as the final re-validation rather
    than a fabricated out-of-range epoch number.
    """
    state: dict[str, Any] = {"best": float("-inf"), "best_epoch": 0, "no_improve": 0}

    def _callback(trainer: Any) -> None:  # noqa: ANN401 — Ultralytics BaseTrainer has no public type
        metrics = getattr(trainer, "metrics", None) or {}
        current = metrics.get(MASK_MAP50_95_KEY)
        if current is None:
            return  # no validation ran this epoch (e.g. val=False mid-run)

        epoch_1_indexed = trainer.epoch + 1
        is_final_revalidation = epoch_1_indexed > total_epochs

        improved = current > state["best"]
        if improved:
            state["best"] = current
            state["best_epoch"] = min(epoch_1_indexed, total_epochs)
        if is_final_revalidation:
            return  # sentinel call — don't count towards patience

        if improved:
            state["no_improve"] = 0
        else:
            state["no_improve"] += 1

        if state["no_improve"] >= patience:
            log.info(
                "Early stopping: mask mAP50-95 has not improved for %d epochs "
                "(best=%.4f @ epoch %d).",
                patience, state["best"], state["best_epoch"],
            )
            trainer.stop = True

    return _callback, state


def report_on_test(
    *,
    model: str,
    run_dir: Path,
    dataset_config: Path,
    imgsz: int,
    device: str,
    epochs: int,
    batch: int,
    patience: int,
    best_epoch: int,
    early_stopped: bool,
    wall_time_seconds_total: float,
    epoch_wall_times_seconds: list[float],
    vram_peak_mb_training: float | None,
) -> Path:
    """Score the selected checkpoint on the **test** split and write
    ``run_dir/metrics.json`` (schema v2) through the standalone evaluator.

    Val is used only for checkpoint selection / early stopping during training;
    it never feeds the reported numbers. Uses ``weights/best.pt``; if that is
    missing (e.g. no validation ran) it falls back to ``weights/last.pt`` with
    a warning. ``params_total`` / ``params_trainable`` come from the evaluator
    (which restores grad flags on the reloaded Ultralytics checkpoint), so they
    are not passed here. ``model`` is the starting model name (``args.model``)
    recorded as ``model`` in metrics.json.
    """
    weights_dir = run_dir / "weights"
    weights = weights_dir / "best.pt"
    if not weights.exists():
        weights = weights_dir / "last.pt"
        log.warning("best.pt not found in %s; reporting on last.pt instead.", weights_dir)

    training_info: dict[str, Any] = {
        "epochs_requested": epochs,
        "epochs_run_this_invocation": len(epoch_wall_times_seconds),
        "batch": batch,
        "patience": patience,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "wall_time_seconds_total": wall_time_seconds_total,
        "epoch_wall_times_seconds": epoch_wall_times_seconds,
        "vram_peak_mb_training": vram_peak_mb_training,
    }
    metrics_path = run_evaluation(
        weights=weights,
        model_type="yolo",
        dataset_config=dataset_config,
        split="test",
        output_dir=run_dir,
        device=device,
        imgsz=imgsz,
        training_info=training_info,
        configure_logging=False,
        model_type_label="yolov8-seg",
        model_id=model,
    )
    _log_headline(metrics_path)
    return metrics_path


def _log_headline(metrics_path: Path) -> None:
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    except (OSError, ValueError, KeyError):
        log.warning("Could not read headline metrics back from %s", metrics_path)
        return
    log.info(
        "Test-split metrics written to %s | mask mAP50=%.4f mAP50-95=%.4f | "
        "box mAP50=%.4f mAP50-95=%.4f",
        metrics_path, m["mask_map50"], m["mask_map50_95"], m["box_map50"], m["box_map50_95"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    cfg = resolve_paths(cfg, project_root)

    run_dir = Path(cfg.project.output_root) / args.name
    setup_run_logging(run_dir)
    log.info("Starting YOLOv8-seg training run: %s", args.name)

    seed = args.seed if args.seed is not None else cfg.project.seed
    seed_everything(seed)
    log.info("Seed set to %d", seed)

    from ultralytics import YOLO  # type: ignore[attr-defined]

    dataset_yaml = resolve_dataset_yaml_for_ultralytics(args.dataset_config)
    workers = args.workers if args.workers is not None else int(cfg.data.get("num_workers", 4))

    augmentations = build_phone_photo_augmentations(args.augment_p)
    early_stop_callback, early_stop_state = build_early_stopping_callback(
        args.patience, args.epochs
    )

    model_source = (
        args.model if not args.resume else _resolve_resume_checkpoint(run_dir, args.model)
    )
    model = YOLO(model_source)
    model.add_callback("on_fit_epoch_end", early_stop_callback)

    epoch_times: list[float] = []
    epoch_start_time: dict[str, float] = {}

    def _on_epoch_start(trainer: Any) -> None:  # noqa: ANN401
        epoch_start_time["t"] = time.monotonic()

    def _on_epoch_end(trainer: Any) -> None:  # noqa: ANN401
        if "t" in epoch_start_time:
            epoch_times.append(time.monotonic() - epoch_start_time["t"])

    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_train_epoch_end", _on_epoch_end)

    # Training-time VRAM: reset right before training, read right after it ends
    # (before the test-split evaluation, which measures its own inference peak).
    reset_peak_vram_for_yolo(args.device)
    train_start = time.monotonic()
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=workers,
        project=str(Path(cfg.project.output_root).resolve()),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        seed=seed,
        amp=True,
        patience=0,  # disable Ultralytics' own combined-fitness stopper; we handle it above
        augmentations=augmentations,
        perspective=DEFAULT_PERSPECTIVE_SCALE,  # viewpoint warp — simulates off-angle phone photos
    )
    total_wall_time = time.monotonic() - train_start
    vram_peak_mb_training = peak_vram_mb_for_yolo(args.device)
    del results  # Ultralytics' own results_dict isn't used; the evaluator re-scores best.pt

    metrics_path = report_on_test(
        model=args.model,
        run_dir=run_dir,
        dataset_config=args.dataset_config,
        imgsz=args.imgsz,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        best_epoch=early_stop_state["best_epoch"],
        early_stopped=early_stop_state["no_improve"] >= args.patience,
        wall_time_seconds_total=total_wall_time,
        epoch_wall_times_seconds=epoch_times,
        vram_peak_mb_training=vram_peak_mb_training,
    )
    log.info("Training complete. Test-split metrics written to %s", metrics_path)


class ResumeError(Exception):
    """Raised when --resume is requested but the checkpoint can't actually be resumed."""


def _resolve_resume_checkpoint(run_dir: Path, fallback_model: str) -> str:
    """Return the last.pt checkpoint for this run if it exists and is
    genuinely resumable, else fall back to the requested starting model
    (first run, nothing to resume).

    Ultralytics strips optimizer state from `last.pt`/`best.pt` once a run
    finishes (`epoch` reset to -1, `optimizer` set to None) to shrink the
    saved file — at that point the checkpoint can no longer be resumed, and
    passing it to `.train(resume=True, ...)` silently retrains from scratch
    instead of raising, which would make `metrics.json` report misleading
    epoch counts. We check for that here and fail loudly instead, since a
    silent full retrain defeats the purpose of --resume (only genuinely
    interrupted/incomplete runs — crash, Ctrl-C, preemption — are resumable).
    """
    last_ckpt = run_dir / "weights" / "last.pt"
    if not last_ckpt.exists():
        log.warning(
            "--resume was passed but no checkpoint found at %s; starting fresh from %s",
            last_ckpt, fallback_model,
        )
        return fallback_model

    ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", -1)
    has_optimizer = ckpt.get("optimizer") is not None
    if epoch < 0 or not has_optimizer:
        raise ResumeError(
            f"{last_ckpt} is not resumable: it was saved after training completed "
            f"(epoch={epoch}, optimizer={'present' if has_optimizer else 'stripped'}). "
            "Ultralytics strips optimizer state from checkpoints once a run finishes, "
            "so a completed run cannot be resumed further — start a new run (different "
            "--name, or load this checkpoint via --model) instead of --resume."
        )

    log.info("Resuming from checkpoint: %s (epoch=%d, optimizer state present)", last_ckpt, epoch)
    return str(last_ckpt)


if __name__ == "__main__":
    main()
