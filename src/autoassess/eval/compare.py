"""Compare two trained models' metrics.json files — autoassess-compare.

Loads two runs' ``runs/<name>/metrics.json`` (written by ``train_yolo.py`` and
``train_maskrcnn.py`` in the shared schema defined in ``autoassess.eval.metrics``)
and emits a comparison as both a Markdown table and a matplotlib grouped bar
chart, covering: mAP@0.5, mAP@0.5:0.95, mask IoU, per-class precision/recall,
inference latency (ms/image), parameter count, and peak VRAM.

Usage
-----
    autoassess-compare \\
        --a runs/yolov8_seg_v1/metrics.json \\
        --b runs/maskrcnn_v1/metrics.json \\
        --out-dir reports/figures/compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_COLORS = {"a": "#2a78d6", "b": "#eb6834"}  # dataviz skill categorical slots 1, 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two AutoAssess models' metrics.json files (Markdown + chart)."
    )
    p.add_argument("--a", type=Path, required=True,
                   help="Path to the first model's metrics.json.")
    p.add_argument("--b", type=Path, required=True,
                   help="Path to the second model's metrics.json.")
    p.add_argument("--out-dir", type=Path, default=Path("reports/figures/compare"),
                   help="Output directory for comparison.md and comparison.png "
                        "(default: reports/figures/compare).")
    return p.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run autoassess-train-yolo / autoassess-train-maskrcnn "
            "first to produce a metrics.json."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _fmt(value: Any, digits: int = 4) -> str:  # noqa: ANN401 — value may be float, int, str, or None
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown_table(a: dict[str, Any], b: dict[str, Any]) -> str:
    name_a = f"{a['model_type']} ({a['run_name']})"
    name_b = f"{b['model_type']} ({b['run_name']})"

    lines = [
        f"# Model comparison: {name_a} vs {name_b}",
        "",
        "## Overall metrics",
        "",
        f"| Metric | {name_a} | {name_b} |",
        "|---|---|---|",
        f"| Box mAP@0.5 | {_fmt(a['metrics']['box_map50'])} | {_fmt(b['metrics']['box_map50'])} |",
        f"| Box mAP@0.5:0.95 | {_fmt(a['metrics']['box_map50_95'])} "
        f"| {_fmt(b['metrics']['box_map50_95'])} |",
        f"| Mask mAP@0.5 | {_fmt(a['metrics']['mask_map50'])} "
        f"| {_fmt(b['metrics']['mask_map50'])} |",
        f"| Mask mAP@0.5:0.95 | {_fmt(a['metrics']['mask_map50_95'])} "
        f"| {_fmt(b['metrics']['mask_map50_95'])} |",
        f"| Mean mask IoU | {_fmt(a['metrics']['mask_iou_mean'])} "
        f"| {_fmt(b['metrics']['mask_iou_mean'])} |",
        f"| Inference latency (ms/image, mean) | {_fmt(a['inference']['latency_ms_mean'], 1)} "
        f"| {_fmt(b['inference']['latency_ms_mean'], 1)} |",
        f"| Inference latency (ms/image, p95) | {_fmt(a['inference']['latency_ms_p95'], 1)} "
        f"| {_fmt(b['inference']['latency_ms_p95'], 1)} |",
        f"| Parameters (total) | {a['model_info']['params_total']:,} "
        f"| {b['model_info']['params_total']:,} |",
        f"| Parameters (trainable) | {a['model_info']['params_trainable']:,} "
        f"| {b['model_info']['params_trainable']:,} |",
        f"| Peak VRAM (MB) | {_fmt(a['model_info']['vram_peak_mb'], 1)} "
        f"| {_fmt(b['model_info']['vram_peak_mb'], 1)} |",
        f"| Best epoch | {a['best_epoch']} | {b['best_epoch']} |",
        f"| Early stopped | {a['early_stopped']} | {b['early_stopped']} |",
        f"| Total training wall time (s) | {_fmt(a['wall_time_seconds_total'], 1)} "
        f"| {_fmt(b['wall_time_seconds_total'], 1)} |",
        "",
        "## Per-class precision / recall",
        "",
        f"| Class | {name_a} Precision | {name_a} Recall | {name_b} Precision "
        f"| {name_b} Recall |",
        "|---|---|---|---|---|",
    ]

    class_names = a.get("class_names") or list(a["metrics"]["per_class"].keys())
    for name in class_names:
        pc_a = a["metrics"]["per_class"].get(name, {})
        pc_b = b["metrics"]["per_class"].get(name, {})
        lines.append(
            f"| {name} | {_fmt(pc_a.get('precision'))} | {_fmt(pc_a.get('recall'))} "
            f"| {_fmt(pc_b.get('precision'))} | {_fmt(pc_b.get('recall'))} |"
        )

    return "\n".join(lines) + "\n"


def build_comparison_chart(a: dict[str, Any], b: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    name_a = f"{a['model_type']} ({a['run_name']})"
    name_b = f"{b['model_type']} ({b['run_name']})"
    color_a, color_b = MODEL_COLORS["a"], MODEL_COLORS["b"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    _grouped_bar(
        axes[0, 0], ["mAP@0.5", "mAP@0.5:0.95", "Mask IoU"],
        [a["metrics"]["mask_map50"], a["metrics"]["mask_map50_95"], a["metrics"]["mask_iou_mean"]],
        [b["metrics"]["mask_map50"], b["metrics"]["mask_map50_95"], b["metrics"]["mask_iou_mean"]],
        name_a, name_b, color_a, color_b,
        title="Mask accuracy", value_fmt="{:.3f}",
    )

    class_names = a.get("class_names") or list(a["metrics"]["per_class"].keys())
    recall_a = [a["metrics"]["per_class"].get(c, {}).get("recall", 0.0) for c in class_names]
    recall_b = [b["metrics"]["per_class"].get(c, {}).get("recall", 0.0) for c in class_names]
    _grouped_bar(
        axes[0, 1], class_names, recall_a, recall_b, name_a, name_b, color_a, color_b,
        title="Per-class recall", value_fmt="{:.2f}", rotate_labels=True,
    )

    _grouped_bar(
        axes[1, 0], ["Latency mean", "Latency p95"],
        [a["inference"]["latency_ms_mean"], a["inference"]["latency_ms_p95"]],
        [b["inference"]["latency_ms_mean"], b["inference"]["latency_ms_p95"]],
        name_a, name_b, color_a, color_b,
        title="Inference latency (ms/image)", value_fmt="{:.1f}",
    )

    params_m_a = a["model_info"]["params_total"] / 1e6
    params_m_b = b["model_info"]["params_total"] / 1e6
    vram_a = a["model_info"]["vram_peak_mb"]
    vram_b = b["model_info"]["vram_peak_mb"]
    labels = ["Params (M)"]
    vals_a = [params_m_a]
    vals_b = [params_m_b]
    if vram_a is not None and vram_b is not None:
        labels.append("Peak VRAM (MB)")
        vals_a.append(vram_a)
        vals_b.append(vram_b)
    _grouped_bar(
        axes[1, 1], labels, vals_a, vals_b, name_a, name_b, color_a, color_b,
        title="Model size / memory", value_fmt="{:.1f}",
    )

    fig.suptitle(
        f"AutoAssess model comparison — {a['model_type']} vs {b['model_type']}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grouped_bar(
    ax: Any,  # noqa: ANN401 — matplotlib Axes has no lightweight import-free type here
    labels: list[str],
    values_a: list[float],
    values_b: list[float],
    name_a: str,
    name_b: str,
    color_a: str,
    color_b: str,
    title: str,
    value_fmt: str = "{:.3f}",
    rotate_labels: bool = False,
) -> None:
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36  # <=24px-equivalent thin bars with a visible surface gap between pairs

    bars_a = ax.bar(x - width / 2, values_a, width, label=name_a, color=color_a,
                     edgecolor="none")
    bars_b = ax.bar(x + width / 2, values_b, width, label=name_b, color=color_b,
                     edgecolor="none")

    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                value_fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

    # headroom above the tallest bar so value labels never collide with the legend
    max_height = max([*values_a, *values_b], default=0)
    if max_height > 0:
        ax.set_ylim(top=max_height * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=30 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
        fontsize=8.5,
    )
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(
        fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.08),
        ncol=2, frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.6, color="#d8d8d5", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    a = load_metrics(args.a)
    b = load_metrics(args.b)

    markdown = build_markdown_table(a, b)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "comparison.md"
    md_path.write_text(markdown, encoding="utf-8")

    chart_path = args.out_dir / "comparison.png"
    build_comparison_chart(a, b, chart_path)

    print(markdown)
    print(f"\nWritten: {md_path}")
    print(f"Written: {chart_path}")


if __name__ == "__main__":
    main()
