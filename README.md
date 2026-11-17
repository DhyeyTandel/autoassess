# AutoAssess

> Automated vehicle damage localisation and severity grading for motor insurance claims.
> Final-year B.Tech Computer Vision course project.

---

## Overview

AutoAssess takes a photograph of a damaged vehicle and produces:
1. **Damage localisation** — bounding boxes and instance segmentation masks over each damaged region.
2. **Severity grading** — a per-region severity label (minor / moderate / severe).

The primary model is **YOLOv8-seg** (nano or small variant); Mask R-CNN is included as a comparison baseline.

---

## Repo Structure

```
autoassess/
├── configs/          # YAML experiment configs
├── data/             # GITIGNORED — raw, interim, processed splits
├── notebooks/        # EDA & figure generation only
├── reports/figures/  # Generated plots & tables
├── runs/             # Experiment outputs (weights, logs) — gitignored
├── scripts/          # One-off CLI scripts (download, preprocess, …)
├── src/autoassess/   # Main Python package
│   ├── data/         # Dataset classes, loaders, splitting utils
│   ├── eval/         # Metrics & evaluation loops
│   ├── infer/        # Inference pipeline & visualisation
│   ├── models/       # Model wrappers
│   ├── train/        # Training entrypoints
│   └── utils/        # Shared utilities (logging, seeding, config IO)
└── tests/            # pytest unit tests
```

---

## Quick Start

### 1. Install dependencies (requires [uv](https://docs.astral.sh/uv/))

```bash
uv sync
# with dev extras
uv sync --extra dev
```

### 2. Activate the environment

```bash
source .venv/bin/activate
```

### 3. Prepare data

```bash
python scripts/download_data.py --config configs/base.yaml
python scripts/preprocess.py   --config configs/base.yaml
```

### 4. Train

```bash
autoassess-train \
  --config configs/yolov8_seg.yaml \
  --device 0 \
  --batch  8 \
  --epochs 50 \
  --imgsz  640
```

### 5. Evaluate

```bash
autoassess-eval \
  --config configs/yolov8_seg.yaml \
  --weights runs/yolov8_seg_v1/weights/best.pt \
  --device 0
```

### 6. Infer on a single image

```bash
autoassess-infer \
  --config configs/yolov8_seg.yaml \
  --weights runs/yolov8_seg_v1/weights/best.pt \
  --source  path/to/car.jpg \
  --device  cpu
```

---

## Configuration

All runtime parameters are controlled via YAML configs in `configs/`.
CLI flags **override** config values when both are present.

Key config sections:

| Section | Purpose |
|---------|---------|
| `project` | Name, seed, output root |
| `data` | Dataset paths, split ratios, image size |
| `model` | Architecture, pretrained weights |
| `train` | Epochs, batch, lr, AMP, optimizer |
| `eval` | IoU threshold, confidence threshold |
| `infer` | Confidence, NMS IoU, visualisation flags |

---

## Results

All numbers are measured on the held-out test split and copied from
[`reports/results.md`](reports/results.md), which is generated from
`runs/*/metrics.json` by `scripts/generate_results_md.py`. That file also has
the per-class tables, failure-case analysis and figure paths.

| Task (dataset) | Model | Box mAP@0.5 | Mask mAP@0.5 | Mask mAP@0.5:0.95 | Latency, mean / p95 (ms/img) |
|---|---|---|---|---|---|
| Damage, 6 classes (CarDD) | YOLOv8n-seg | 0.6407 | 0.6203 | 0.4012 | 30.4 / 36.0 |
| Panels, 6 classes (Carparts-Seg) | YOLOv8n-seg | 0.9346 | 0.9342 | 0.7675 | 16.7 / 22.3 |
| Damage, 7 classes (VehiDE) | YOLOv8s-seg | 0.4712 | 0.4283 | 0.2344 | 29.5 / 41.8 |
| Damage (CarDD), baseline | Mask R-CNN | not trained yet | | | |

Things to know when reading these:

- **Per-class spread on CarDD is wide.** Mask AP@0.5 ranges from 0.90 on
  `tire flat` and `glass shatter` down to 0.18 on `crack`. Thin, low-contrast
  damage (`crack`, `scratch`, `dent`) is where the model misses.
- **Near-duplicate check.** A perceptual-hash audit
  ([`reports/split_audit.md`](reports/split_audit.md)) flagged visually similar
  train/test pairs. Re-scoring with those test images removed barely moves the
  numbers: panels mask mAP@0.5 0.9342 to 0.9354 (12 images removed), VehiDE
  0.4283 to 0.4272 (45 removed).
- **VehiDE run stopped at epoch 48 of 50.** Training crashed on a Colab runtime
  reset; the reported weights are from epoch 48.
- **Severity grading is not validated against real labels.** The ResNet-18
  severity classifier reaches 0.78 validation agreement with this project's own
  hand-written severity heuristic. That is agreement with a rule, not
  real-world grading accuracy.
- **Mask R-CNN baseline is pending.** No comparison numbers exist yet.

---

## Reproducibility

- Default random seed: `42` (set via `seed_everything(42)` at run start).
- Full config YAML is saved to `runs/<experiment_name>/config.yaml`.
- Python 3.11, dependency versions pinned in `uv.lock`.

---

## Deploying

The demo is a React (Vite) frontend in `frontend/` served by a FastAPI backend
(`autoassess-web`, port 7860). It ships as one CPU-only Docker image: a Node
stage builds the frontend (`npm ci`, then `npm run build`, which type-checks
with `tsc -b`), and the Python stage (CPU torch wheels, non-root user) serves
the API under `/api` and the built static files.

Local development (two terminals):

```bash
uv run autoassess-web                 # API on :7860 (AUTOASSESS_DEVICE=mps|cpu|0)
cd frontend && npm install && npm run dev   # Vite dev server, proxies /api to the backend
```

Weights are not in git and are mounted read-only at `/app/runs`. Place them on
the host next to `docker-compose.yml`:

```
runs/yolov8_seg_v1/weights/best.pt   # CarDD damage model
runs/vehide_seg_v1/weights/best.pt   # VehiDE damage model
runs/parts_seg_v1/weights/best.pt    # Carparts-Seg parts model
```

Build and start:

```bash
docker compose up -d --build
```

The container is published on `127.0.0.1:7860` only. To expose it, add a
Cloudflare Tunnel public hostname with service `http://localhost:7860`.

---

## License

MIT © Dhyey Tandel
