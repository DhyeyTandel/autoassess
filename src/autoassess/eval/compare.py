"""Compare two trained models' metrics.json files — autoassess-compare.

Loads two runs' ``runs/<name>/metrics.json`` (written by ``train_yolo.py`` and
``train_maskrcnn.py`` in the shared schema defined in ``autoassess.eval.metrics``)
and emits a comparison as both a Markdown table and a matplotlib grouped bar
chart, covering: mAP@0.5, mAP@0.5:0.95, mask IoU, operating-point
precision/recall/F1 (schema v2), per-class AP50, inference latency
(ms/image), parameter count, and peak VRAM. Legacy v1 files (no
``schema_version``) still load; their operating-point fields render as "n/a"
because v1's per-class "precision" is AP50 under the wrong name.

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

from autoassess.eval.scoring import OPERATING_CONF

LEGACY_NOTICE = (
    "Legacy metrics (schema v1): evaluated on the val split before the eval fixes; "
    "re-score with autoassess-eval for test-split numbers."
)
PER_CLASS_HEADERS = [
    f"Precision@{OPERATING_CONF}",
    f"Recall@{OPERATING_CONF}",
    f"F1@{OPERATING_CONF}",
    "Best-F1 conf",
    "Box AP50",
    "Mask AP50",
    "Max recall",
]

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


def is_legacy(m: dict[str, Any]) -> bool:
    """True for pre-fix (schema v1) metrics.json files."""
    return int(m.get("schema_version", 1)) < 2


def eval_split_label(m: dict[str, Any]) -> str:
    split = m.get("eval_split")
    if split:
        return str(split)
    return "val (legacy)"


def macro_prf_cell(m: dict[str, Any], digits: int = 4) -> str:
    """"P / R / F1" macro cell at the operating point (mask); n/a for v1."""
    if is_legacy(m):
        return "n/a"
    mac = m["metrics"].get("operating_point", {}).get("mask")
    if not mac:
        return "n/a"
    return " / ".join(_fmt(float(mac[k]), digits) for k in ("precision", "recall", "f1"))


def headline_rows(m: dict[str, Any], digits: int = 4) -> list[tuple[str, str]]:
    """Label/value pairs for the accuracy block of the headline table.

    v1 files never yield operating-point values: their "precision"/"recall"
    fields are not precision/recall, so those rows are "n/a".
    """
    mt = m["metrics"]
    legacy = is_legacy(m)
    rows = [
        ("Box mAP@0.5", _fmt(mt["box_map50"], digits)),
        ("Box mAP@0.5:0.95", _fmt(mt["box_map50_95"], digits)),
        ("Mask mAP@0.5", _fmt(mt["mask_map50"], digits)),
        ("Mask mAP@0.5:0.95", _fmt(mt["mask_map50_95"], digits)),
        ("Mask IoU (true positives)",
         "n/a" if legacy else _fmt(mt.get("mask_iou_true_positives"), digits)),
        ("Mask IoU (per GT, misses = 0)",
         "n/a" if legacy else _fmt(mt.get("mask_iou_per_gt"), digits)),
        (f"Macro P / R / F1 @ conf {OPERATING_CONF} (mask)", macro_prf_cell(m, digits)),
    ]
    if legacy:
        rows.append(
            ("Mask IoU (legacy, matched pairs only)", _fmt(mt.get("mask_iou_mean"), digits))
        )
    rows.append(("Evaluated on", eval_split_label(m)))
    return rows


def per_class_cells(m: dict[str, Any], name: str, digits: int = 4) -> list[str]:
    """Cells matching ``PER_CLASS_HEADERS`` for one class; "n/a" where v1 lacks them."""
    pc = m["metrics"]["per_class"].get(name, {})
    if is_legacy(m):
        return ["n/a"] * 4 + [
            _fmt(pc.get("box_ap50"), digits), _fmt(pc.get("mask_ap50"), digits), "n/a",
        ]
    return [
        _fmt(pc.get("precision"), digits),
        _fmt(pc.get("recall"), digits),
        _fmt(pc.get("f1"), digits),
        _fmt(pc.get("f1_opt_conf"), 3),
        _fmt(pc.get("box_ap50"), digits),
        _fmt(pc.get("mask_ap50"), digits),
        _fmt(pc.get("max_recall"), digits),
    ]


def per_class_table(m: dict[str, Any]) -> list[str]:
    """Markdown lines for one run's per-class table."""
    lines = [
        "| Class | " + " | ".join(PER_CLASS_HEADERS) + " |",
        "|---" * (len(PER_CLASS_HEADERS) + 1) + "|",
    ]
    names = m.get("class_names") or list(m["metrics"]["per_class"].keys())
    for name in names:
        lines.append(f"| {name} | " + " | ".join(per_class_cells(m, name)) + " |")
    return lines


def _headline_lines(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    rows_a = headline_rows(a)
    rows_b = headline_rows(b)
    labels = [lab for lab, _ in rows_a]
    for lab, _ in rows_b:
        if lab not in labels:
            labels.insert(len(labels) - 1, lab)  # keep "Evaluated on" last
    da, db = dict(rows_a), dict(rows_b)
    return [f"| {lab} | {da.get(lab, 'n/a')} | {db.get(lab, 'n/a')} |" for lab in labels]


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
        *_headline_lines(a, b),
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
        "## Per-class",
        "",
        f"### {name_a}",
        "",
        *per_class_table(a),
        "",
        f"### {name_b}",
        "",
        *per_class_table(b),
    ]
    if is_legacy(a) or is_legacy(b):
        lines += ["", LEGACY_NOTICE]

    return "\n".join(lines) + "\n"


def build_comparison_chart(a: dict[str, Any], b: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    name_a = f"{a['model_type']} ({a['run_name']})"
    name_b = f"{b['model_type']} ({b['run_name']})"
    color_a, color_b = MODEL_COLORS["a"], MODEL_COLORS["b"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    legacy = is_legacy(a) or is_legacy(b)
    mask_labels = ["mAP@0.5", "mAP@0.5:0.95"]
    mask_a = [a["metrics"]["mask_map50"], a["metrics"]["mask_map50_95"]]
    mask_b = [b["metrics"]["mask_map50"], b["metrics"]["mask_map50_95"]]
    if not legacy:  # v1 has no comparable IoU; omit rather than plot a wrong bar
        mask_labels.append("Mask IoU (per GT)")
        mask_a.append(a["metrics"]["mask_iou_per_gt"])
        mask_b.append(b["metrics"]["mask_iou_per_gt"])
    _grouped_bar(
        axes[0, 0], mask_labels, mask_a, mask_b,
        name_a, name_b, color_a, color_b,
        title="Mask accuracy", value_fmt="{:.3f}",
    )

    class_names = a.get("class_names") or list(a["metrics"]["per_class"].keys())
    if legacy:  # no operating-point recall in v1; fall back to the AP50 both files have
        key, title = "mask_ap50", "Per-class mask AP50 (legacy run: no operating-point recall)"
    else:
        key, title = "recall", f"Per-class recall @ conf {OPERATING_CONF} (mask)"
    recall_a = [a["metrics"]["per_class"].get(c, {}).get(key, 0.0) for c in class_names]
    recall_b = [b["metrics"]["per_class"].get(c, {}).get(key, 0.0) for c in class_names]
    _grouped_bar(
        axes[0, 1], class_names, recall_a, recall_b, name_a, name_b, color_a, color_b,
        title=title, value_fmt="{:.2f}", rotate_labels=True,
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
