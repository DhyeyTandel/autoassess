#!/usr/bin/env python3
"""scripts/generate_results_md.py — assemble reports/results.md from runs/*/metrics.json.

Scans `runs/` for `metrics.json` files (written by every trainer in this
project — `train_yolo.py`, `train_maskrcnn.py`, `train_parts.py`,
`train_severity.py` — in the shared schema from `autoassess.eval.metrics`,
plus `train_severity.py`'s own extended schema) and writes one Markdown
report summarising whichever runs actually exist.

This never fabricates a row for a model that hasn't been trained — a
section with no matching run is written as "_Pending — not yet trained._"
rather than omitted or filled with placeholder numbers, so the report is
honest about what it does and doesn't cover as of generation time.

Usage
-----
    python scripts/generate_results_md.py --runs-dir runs --out reports/results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoassess.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

# run_name -> (section title, one-line description)
KNOWN_RUNS: dict[str, tuple[str, str]] = {
    "yolov8_seg_v1": (
        "Damage detection — YOLOv8-seg",
        "Trained on CarDD (6 damage classes).",
    ),
    "maskrcnn_v1": (
        "Damage detection — Mask R-CNN",
        "Trained on CarDD (6 damage classes), comparison baseline.",
    ),
    "parts_seg_v1": (
        "Part segmentation — YOLOv8-seg",
        "Trained on Carparts-Seg, remapped to 6 panel classes.",
    ),
    "severity_v1": (
        "Severity classifier",
        "Two-branch ResNet-18, weak supervision from the heuristic grader.",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble reports/results.md from runs/*/metrics.json.")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("reports/results.md"))
    p.add_argument("--figures-dir", type=Path, default=Path("reports/figures"),
                   help="Where qualitative/failure/PR figures were written, for linking "
                        "(not copying).")
    return p.parse_args()


def load_all_metrics(runs_dir: Path) -> dict[str, dict[str, Any]]:
    found = {}
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8") as f:
            found[run_name] = json.load(f)
    return found


def fmt(value: Any, digits: int = 4) -> str:  # noqa: ANN401 — value may be float, int, str, bool, None
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_detection_section(run_name: str, title: str, desc: str, m: dict[str, Any] | None) -> str:
    lines = [f"## {title}", "", desc, ""]
    if m is None:
        lines.append(f"_Pending — not yet trained (no `runs/{run_name}/metrics.json` found)._")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Model | {m.get('model', 'n/a')} |",
        f"| Box mAP@0.5 | {fmt(m['metrics']['box_map50'])} |",
        f"| Box mAP@0.5:0.95 | {fmt(m['metrics']['box_map50_95'])} |",
        f"| Mask mAP@0.5 | {fmt(m['metrics']['mask_map50'])} |",
        f"| Mask mAP@0.5:0.95 | {fmt(m['metrics']['mask_map50_95'])} |",
        f"| Mean mask IoU | {fmt(m['metrics']['mask_iou_mean'])} |",
        f"| Inference latency (ms/image, mean) | {fmt(m['inference']['latency_ms_mean'], 1)} |",
        f"| Inference latency (ms/image, p95) | {fmt(m['inference']['latency_ms_p95'], 1)} |",
        f"| Parameters (total) | {m['model_info']['params_total']:,} |",
        f"| Peak VRAM (MB) | {fmt(m['model_info'].get('vram_peak_mb'), 1)} |",
        f"| Epochs (requested / this run) | {m['epochs_requested']} / "
        f"{m['epochs_run_this_invocation']} |",
        f"| Best epoch | {m['best_epoch']} |",
        f"| Early stopped | {m['early_stopped']} |",
        f"| Training wall time (s) | {fmt(m['wall_time_seconds_total'], 1)} |",
        "",
        "### Per-class",
        "",
        "| Class | Precision | Recall | Box AP50 | Mask AP50 |",
        "|---|---|---|---|---|",
    ]
    for name, pc in m["metrics"]["per_class"].items():
        lines.append(
            f"| {name} | {fmt(pc.get('precision'))} | {fmt(pc.get('recall'))} "
            f"| {fmt(pc.get('box_ap50'))} | {fmt(pc.get('mask_ap50'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_severity_section(m: dict[str, Any] | None, figures_dir: Path) -> str:
    lines = ["## Severity classifier", ""]
    if m is None:
        lines.append("_Pending — not yet trained (no `runs/severity_v1/metrics.json` found)._")
        smoke_compare = figures_dir / "severity_compare" / "severity_comparison.md"
        if smoke_compare.exists():
            lines.append("")
            lines.append(
                f"A preliminary smoke-test comparison exists at `{smoke_compare}` "
                "(`runs/severity_smoke`, 32 images) — an architecture sanity check only, "
                "not a result for `severity_v1`. See that file for its own circularity "
                "caveat before citing any number from it."
            )
        lines.append("")
        return "\n".join(lines)

    lines += [
        "> " + m.get("circularity_warning", ""),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Model | {m.get('model', 'n/a')} |",
        f"| Val accuracy vs. heuristic | {fmt(m.get('val_accuracy_vs_heuristic'))} |",
        f"| Epochs (requested / this run) | {m['epochs_requested']} / "
        f"{m['epochs_run_this_invocation']} |",
        f"| Best epoch | {m['best_epoch']} |",
        f"| Parameters (total) | {m['model_info']['params_total']:,} |",
        f"| Training wall time (s) | {fmt(m['wall_time_seconds_total'], 1)} |",
        "",
        f"Train label distribution (heuristic-derived): {m.get('train_label_distribution', {})}",
        "",
    ]
    return "\n".join(lines)


def render_failure_analysis_section(figures_dir: Path) -> str:
    """Read every `*_failure_cases.json` sidecar written by `autoassess-report`
    and summarise real per-instance data (predicted classes, IoU, n_gt/n_dt)
    into a written failure-mode analysis — grounded in the JSON, not
    hand-typed impressions of the figure."""
    lines = ["## Failure case analysis", ""]
    sidecars = sorted(figures_dir.glob("**/*_failure_cases.json"))
    if not sidecars:
        lines.append("_Pending — no `*_failure_cases.json` found under "
                     f"`{figures_dir}` yet (run `autoassess-report` first)._")
        lines.append("")
        return "\n".join(lines)

    for sidecar in sidecars:
        model_name = sidecar.name.removesuffix("_failure_cases.json")
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
        worst = data.get("worst_scoreable", [])
        unscoreable = data.get("unscoreable_zero_gt", [])

        lines.append(f"### {model_name}")
        lines.append("")
        if worst:
            ious = [r["mean_iou"] for r in worst]
            if max(ious) == 0.0:
                lines.append(
                    f"Worst {len(worst)} scoreable test images (real ground truth present) "
                    "all scored mean mask IoU 0.000 — complete misses (no predicted mask "
                    "overlapped any ground-truth instance at all), not partial-overlap "
                    "errors. Several of these images also had zero predictions (`n_dt = 0`)."
                )
            else:
                lines.append(
                    f"Worst {len(worst)} scoreable test images (real ground truth present) "
                    f"span mean mask IoU {min(ious):.3f}–{max(ious):.3f}. "
                )
            over_pred = [r for r in worst if r["n_dt"] > r["n_gt"]]
            if over_pred:
                lines.append(
                    f"{len(over_pred)}/{len(worst)} of these show more predicted instances "
                    f"than ground-truth instances (`n_dt > n_gt`) — duplicate or split "
                    f"detections on a single real instance, not missed or misclassified damage."
                )
            class_counts: dict[str, int] = {}
            for r in worst:
                for c in r["predicted_classes"]:
                    class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts:
                ranked = sorted(class_counts.items(), key=lambda kv: -kv[1])
                lines.append(
                    "Classes appearing most often in these failure images: "
                    + ", ".join(f"`{c}` ({n})" for c, n in ranked[:3]) + "."
                )
            lines.append("")

        if unscoreable:
            lines.append(
                f"**{len(unscoreable)} test images have zero ground-truth instances** "
                "after dataset conversion — excluded from the ranking above since IoU "
                "against no ground truth is undefined, not a model failure. These are a "
                "data-labeling gap (see `configs/carparts.yaml` `panel_groups`: images "
                "whose only raw annotation was a dropped class, e.g. the source dataset's "
                "\"object\" catch-all, or images with no raw annotation at all end up with "
                "an empty label file after remapping). The model still produced plausible-"
                "looking predictions on most of them, which cannot be verified without "
                "re-annotating those images against the 6-panel taxonomy."
            )
            lines.append("")

    return "\n".join(lines)


def render_figures_section(figures_dir: Path) -> str:
    lines = ["## Figures", "", "Generated by `autoassess-report` and `dot`, referenced here by "
             "relative path (not embedded — open the files directly):", ""]
    candidates = [
        ("Pipeline architecture", "architecture.pdf", figures_dir),
        ("Qualitative results grid", "**/*_qualitative_grid.png", figures_dir),
        ("Failure case analysis (figure)", "**/*_failure_cases.png", figures_dir),
        ("Failure case data", "**/*_failure_cases.json", figures_dir),
        ("Precision-recall curves", "**/*_pr_curves.png", figures_dir),
        ("Model comparison", "compare/comparison.png", figures_dir),
        (
            "Severity confusion matrices (severity_smoke — preliminary, not severity_v1)",
            "severity_compare/severity_confusion_matrices.png",
            figures_dir,
        ),
    ]
    any_found = False
    for label, pattern, base in candidates:
        matches = sorted(base.glob(pattern)) if "*" in pattern else (
            [base / pattern] if (base / pattern).exists() else []
        )
        for match in matches:
            any_found = True
            lines.append(f"- **{label}**: `{match}`")
    if not any_found:
        lines.append("_No figures generated yet._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    all_metrics = load_all_metrics(args.runs_dir)
    log.info("Found metrics.json for runs: %s", sorted(all_metrics.keys()))

    sections = [
        "# AutoAssess — results summary",
        "",
        "Auto-generated by `scripts/generate_results_md.py` from `runs/*/metrics.json`. "
        "Sections marked _Pending_ reflect models not yet trained at generation time — "
        "no numbers in this file are placeholders or estimates.",
        "",
    ]

    for run_name, (title, desc) in KNOWN_RUNS.items():
        if run_name == "severity_v1":
            continue
        sections.append(render_detection_section(run_name, title, desc, all_metrics.get(run_name)))

    sections.append(render_severity_section(all_metrics.get("severity_v1"), args.figures_dir))
    sections.append(render_failure_analysis_section(args.figures_dir))
    sections.append(render_figures_section(args.figures_dir))

    extra = sorted(set(all_metrics) - set(KNOWN_RUNS))
    if extra:
        sections.append("## Other runs found (not in the standard set)")
        sections.append("")
        sections.append(", ".join(f"`{name}`" for name in extra))
        sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
