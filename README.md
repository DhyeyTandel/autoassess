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

> _To be filled in after experiments._

| Model | mAP@50 | mAP@50-95 | Inf. time (ms/img) |
|-------|--------|-----------|-------------------|
| YOLOv8n-seg | — | — | — |
| YOLOv8s-seg | — | — | — |
| Mask R-CNN  | — | — | — |

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
