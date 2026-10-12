# AutoAssess

> Vehicle damage localisation, part identification and severity triage for motor insurance claims.
> Final-year B.Tech Computer Vision course project.

AutoAssess takes one photo of a damaged vehicle. It returns:
- **Damage instances:** a box, a pixel mask and a damage type for each damaged region.
- **The affected panel:** for each damage instance, which car part it sits on.
- **A severity grade:** minor, moderate or severe, for each instance.
- **A routing decision for the claim:** auto-approve, human review, or total-loss review.

It runs as a command-line pipeline (`autoassess-pipeline`) and as a Gradio web demo (`app.py`).

---

## How it works

```
photo ─┬─► damage model (CarDD, YOLOv8n-seg) ──┐
       ├─► damage model (VehiDE, YOLOv8s-seg) ─┤─► merge ─► match each damage to a part ─► severity per instance ─► triage
       └─► parts model (Carparts, YOLOv8n-seg) ────────────────┘
```

- **Two damage models, merged.**
  - The CarDD model reports dent, scratch, crack, glass shatter, broken lamp and flat tyre.
  - The VehiDE model contributes only the classes CarDD lacks: `torn`, `missing_part` and `punctured`.
  - A VehiDE detection is dropped if its mask overlaps a kept detection at IoU ≥ 0.5, whatever its class.
  - Code: `src/autoassess/infer/merge.py`.
- **Part matching by coverage.**
  - Each damage is matched to the panel holding the largest share of its mask: area(damage ∩ part) / area(damage) ≥ 0.5.
  - IoU is the wrong measure here, because it penalises a small dent on a large door even when the dent sits entirely inside it.
  - Code: `src/autoassess/infer/associate.py`.
- **Severity.**
  - The score is a class-weighted fraction of damaged area, bucketed into minor, moderate and severe (`configs/severity.yaml`).
  - The weights and thresholds are hand-picked. See [Limitations](#limitations).
- **Triage.**
  - Any severe instance, or a large total damaged area, goes to total-loss review.
  - All-minor and small goes to auto-approve.
  - Everything else, including photos with no detections, goes to human review.

---

## Results

All numbers are on held-out **test** splits, scored by `autoassess-eval` on a Colab **Tesla T4** at imgsz 640, batch 1. Latency is per image over 30 images. Full tables, per-class results and failure analysis: [`reports/results.md`](reports/results.md).

| Model | Dataset (test images) | Mask mAP@0.5 | Mask mAP@0.5:0.95 | Box mAP@0.5 | Macro F1 @ conf 0.25 | Latency mean / p95 | Peak VRAM |
|---|---|---|---|---|---|---|---|
| YOLOv8n-seg (damage) | CarDD (374) | 0.620 | 0.401 | 0.641 | 0.594 | 30.4 / 36.0 ms | 310 MB |
| YOLOv8s-seg (damage) | VehiDE (1,743) | 0.428 | 0.234 | 0.471 | 0.481 | 29.5 / 41.8 ms | 294 MB |
| YOLOv8n-seg (parts) | Carparts (276) | 0.934 | 0.768 | 0.935 | 0.873 | 16.7 / 22.3 ms | 227 MB |

- **Leakage check.**
  - A perceptual-hash audit flagged near-duplicates of training images in the test splits: 12 in Carparts and 45 in VehiDE.
  - Scoring without them changes mask mAP@0.5 by ≤ 0.002 (Carparts 0.935, VehiDE 0.427).
  - Lists: `reports/split_audit/`.
- **Weak classes.**
  - On CarDD, crack (mask AP50 0.18) and scratch (0.47) are the weakest.
  - On VehiDE, scratch and dent (0.20–0.24) are the weakest.
- **Severity classifier.**
  - A ResNet-18 two-branch model trained on the heuristic's labels agrees with the heuristic on **74.1%** of CarDD test images (n=374).
  - It is weak on *minor*: it recovers only 29% of the photos the heuristic calls minor.
  - This measures agreement with a formula, **not** real grading accuracy.
- **Mask R-CNN.** The comparison model is implemented (`autoassess-train-maskrcnn`) but not trained, so it has no results.

---

## Demo

```bash
uv run python app.py           # http://localhost:7860
```

- **Upload:** a JPEG, PNG or HEIC (iPhone) photo.
- **Output:**
  - the photo with damage masks, labelled `type · severity · part`;
  - a verdict card with severity counts and total damaged area;
  - the full JSON result.
- **Sensitivity control:**
  - *High recall* uses a VehiDE cutoff of 0.07 and flags more torn or missing parts, with more false alarms.
  - *Standard* uses 0.15.
  - The chosen setting is recorded in the JSON.
- **Environment variables:**

| Variable | Default | Effect |
|---|---|---|
| `DAMAGE_MODEL_SOURCE` | `merged` | `merged`, `cardd` or `vehide` |
| `AUTOASSESS_DEVICE` | `cpu` | `cpu`, `mps` or a GPU index |
| `SECONDARY_DAMAGE_CONF` | `0.07` | VehiDE cutoff used by *High recall* |
| `DAMAGE_WEIGHTS`, `SECONDARY_DAMAGE_WEIGHTS`, `PARTS_WEIGHTS` | `runs/…/best.pt` | Override checkpoint paths |

The demo expects trained weights in `runs/yolov8_seg_v1/`, `runs/vehide_seg_v1/` and `runs/parts_seg_v1/`.

---

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

### Data

Datasets live under `data/`, which is gitignored. Each one goes raw → processed in YOLO-seg format, with fixed train/val/test splits.

| Dataset | Source | Prepare |
|---|---|---|
| CarDD (4,000 images, 6 damage classes) | License-gated: download `CarDD_release.zip` from https://cardd-ustc.github.io/ | `python scripts/download_data.py --config configs/base.yaml --zip-path data/raw/CarDD_release.zip`, then `autoassess-convert --config configs/base.yaml --dataset-config configs/cardd.yaml` |
| VehiDE (13,945 images, 7 damage classes) | Kaggle; needs `~/.kaggle/kaggle.json` | `python scripts/download_vehide.py --config configs/base.yaml`, then `autoassess-convert-vehide --config configs/base.yaml --dataset-config configs/vehide.yaml` |
| Carparts-Seg (23 classes remapped to 6 panels) | Ultralytics, auto-download | `autoassess-carparts --config configs/base.yaml --dataset-config configs/carparts.yaml` |

### Train

```bash
autoassess-train-yolo  --dataset-config configs/cardd.yaml    --model yolov8n-seg.pt --name yolov8_seg_v1 --device 0
autoassess-train-yolo  --dataset-config configs/vehide.yaml   --model yolov8s-seg.pt --name vehide_seg_v1 --device 0
autoassess-train-parts --dataset-config configs/carparts.yaml --model yolov8n-seg.pt --name parts_seg_v1  --device 0
autoassess-train-severity --help
```

- **Defaults:** `--batch 8`, `--epochs 50`, `--imgsz 640`, mixed precision on, seed 42.
- **Selection:** the validation split only picks the best checkpoint; reported metrics come from the test split.
- **VehiDE on Colab:** it was trained with `colab/train_vehide_colab.ipynb`.

### Evaluate

```bash
autoassess-eval --weights runs/yolov8_seg_v1/weights/best.pt --dataset-config configs/cardd.yaml --split test --device 0
autoassess-eval --weights runs/parts_seg_v1/weights/best.pt  --dataset-config configs/carparts.yaml --split test \
    --model-type-label yolov8-seg-parts --exclude-list reports/split_audit/carparts_flagged_test.txt
autoassess-severity-eval --weights runs/severity_v1/weights/best.pt --split test --out-dir reports/figures/severity_compare_test
uv run python scripts/generate_results_md.py      # rebuilds reports/results.md from runs/*/metrics.json
```

### Run the pipeline on one photo

```bash
autoassess-pipeline \
  --damage-weights runs/yolov8_seg_v1/weights/best.pt \
  --secondary-damage-weights runs/vehide_seg_v1/weights/best.pt \
  --parts-weights runs/parts_seg_v1/weights/best.pt \
  --source path/to/car.jpg --device cpu --out result.json
```

The command line defaults to the stricter VehiDE cutoff of 0.15; use `--secondary-damage-conf` to change it.

---

## Limitations

- **Severity and triage are not validated.**
  - No dataset here has severity labels, so the severity weights, the minor/moderate/severe thresholds and the triage area limits are all hand-picked.
  - The learned severity model was trained to imitate that formula.
  - Neither has been checked against adjuster decisions or claim payouts. Treat every routing decision as a suggestion for human review.
- **Most damage is not matched to a part.** About 74% of CarDD damage instances get no part, because the parts model has only 6 panels: bonnet, front bumper, rear bumper, door, headlamp and windshield. The source dataset has no fender class at all.
- **High recall costs precision.** At the demo's 0.07 VehiDE cutoff, roughly half the extra torn or missing-part detections are false alarms. Detections within about 0.01 of a cutoff can flip with small pixel changes, such as re-encoding the image.
- **Thin damage is often missed.** A detached bumper on CarDD test image `000042` is only caught in High recall mode, at confidence 0.072. Retraining VehiDE at a higher resolution is the likely fix but hasn't been done.
- **Mask R-CNN is untrained**, so there is no second architecture to compare against.
- **The demo was tested only in Chrome**, at desktop and 375 px phone widths, not on real phones or other browsers.

Every decision, with its trade-offs and dead ends, is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Repository layout

```
autoassess/
├── app.py            # Gradio demo
├── configs/          # base + per-dataset YAML (class order, paths, severity/triage/merge settings)
├── colab/            # Colab training notebook (VehiDE)
├── data/             # gitignored: raw → processed (YOLO-seg) splits
├── docs/             # DECISIONS.md: decision log
├── notebooks/        # EDA and figure generation only
├── reports/          # results.md, figures/, split_audit/
├── runs/             # gitignored: weights, logs, metrics.json per run
├── scripts/          # download, preprocess, split audit, results.md generator
├── src/autoassess/
│   ├── data/         # converters, dataset classes, splits
│   ├── eval/         # COCO-style evaluation, metrics, reports
│   ├── infer/        # pipeline, damage-to-part matching, model merge, overlay rendering
│   ├── models/       # severity heuristic and learned severity model
│   ├── train/        # YOLO, parts, Mask R-CNN and severity trainers
│   └── utils/        # config, logging, seeding
└── tests/            # pytest unit tests (synthetic fixtures; no data or weights needed)
```

## Reproducibility

- **Seeding:** seed 42 everywhere, via `seed_everything`.
- **Config snapshot:** each run saves its full config to `runs/<name>/config.yaml`.
- **Pinned dependencies:** pinned in `uv.lock`. The T4 evaluation pinned `ultralytics==8.4.115` to match it.
- **Tests:** `uv run --extra dev pytest -m "not slow"` runs CPU-only in about 10 seconds.

## License

MIT © Dhyey Tandel. CarDD, VehiDE and Carparts-Seg are distributed under their own licenses and are not included in this repository.
