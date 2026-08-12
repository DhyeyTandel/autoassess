"""Shared augmentation policy — phone-photo simulation for CarDD training.

Single source of truth for the augmentation pipeline used by both
``train_yolo.py`` and ``train_maskrcnn.py``, so the YOLOv8-seg vs Mask R-CNN
comparison is trained under the same policy rather than two independently
hand-matched copies drifting apart.

Simulates user-submitted phone photos: illumination shift, motion blur,
JPEG compression, random shadow, and viewpoint (perspective) warp.
"""

from __future__ import annotations

from typing import Any

DEFAULT_AUGMENT_P = 0.5
DEFAULT_PERSPECTIVE_SCALE = 0.0005


def build_phone_photo_augmentations(p: float = DEFAULT_AUGMENT_P) -> list[Any]:
    """Pixel-only Albumentations pipeline: illumination shift, motion blur,
    JPEG compression, and random shadow.

    Kept pixel-only (no geometry change) so it composes safely regardless of
    the downstream framework's mask/bbox handling; viewpoint warp is a
    separate geometric transform (see `build_perspective_transform`) since
    it must be applied consistently to image, boxes, and masks together.
    """
    import albumentations as A

    return [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=p),
        A.RandomGamma(gamma_limit=(70, 130), p=p),
        A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_limit=(1, 3), p=p * 0.6),
        A.MotionBlur(blur_limit=(3, 9), p=p * 0.5),
        A.ImageCompression(quality_range=(40, 90), p=p),
    ]


def build_perspective_transform(
    scale: float = DEFAULT_PERSPECTIVE_SCALE, p: float = DEFAULT_AUGMENT_P
) -> Any:  # noqa: ANN401 — Albumentations transform objects have no shared public base type
    """Mask/bbox-aware perspective warp simulating off-angle phone photos.

    Returned as a standalone Albumentations transform (rather than folded
    into `build_phone_photo_augmentations`) so callers that need it wired
    through a bbox/mask-aware `A.Compose` (Mask R-CNN) and callers that use
    a framework-native perspective hyperparameter (YOLO's `RandomPerspective`
    via Ultralytics' `perspective` hyp) can both consume the same `scale`.
    """
    import albumentations as A

    return A.Perspective(scale=(0.0, scale), keep_size=True, p=p)


def build_full_augmentation_pipeline(
    p: float = DEFAULT_AUGMENT_P, perspective_scale: float = DEFAULT_PERSPECTIVE_SCALE
) -> Any:  # noqa: ANN401 — Albumentations Compose object has no shared public base type
    """Full mask/bbox-aware Albumentations pipeline (pixel transforms +
    perspective warp) for frameworks that consume a single `A.Compose`,
    e.g. the Mask R-CNN training dataset. Bbox/mask coordinates are passed
    through Albumentations' `bboxes`/`masks` targets by the caller.
    """
    import albumentations as A

    transforms = [
        *build_phone_photo_augmentations(p),
        build_perspective_transform(perspective_scale, p),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
