"""Shared scoring-time inference settings for every model in the project.

Average precision integrates the whole precision-recall curve, so the
detector must emit low-confidence detections when it is being scored. A
0.25 confidence cutoff truncates the curve at the high-precision end and
penalises whichever model gets cut harder (YOLO and Mask R-CNN have
differently calibrated scores). Scoring therefore uses a near-zero
threshold for both models, while latency is measured at the deployment
operating point.
"""

from __future__ import annotations

SCORING_CONF_THRESHOLD: float = 0.001
"""Minimum confidence kept when scoring (AP/mAP). Near zero so the PR curve
is not truncated."""

SCORING_MAX_DETS: int = 100
"""Maximum detections per image when scoring. Matches COCOeval's largest
`maxDets`, so keeping more would be ignored anyway."""

OPERATING_CONF: float = 0.25
"""Confidence threshold of the deployment operating point. Used for latency
measurement and any operating-point (precision/recall/F1) reporting, never
for AP."""
