# AutoAssess — Project Context for AI Assistants

## Project Overview
AutoAssess is an automated vehicle damage localisation and severity grading system for motor insurance claims. This is a final-year B.Tech Computer Vision course project.

**Goal**: Given an image of a damaged vehicle, detect and localise the damaged regions (bounding boxes + masks) and classify severity into discrete grades (e.g. minor / moderate / severe).

---

## Stack
| Concern | Library |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch |
| Detection / Segmentation | Ultralytics YOLOv8 (`yolov8n-seg`, `yolov8s-seg`) |
| Instance Segmentation (alt) | torchvision Mask R-CNN |
| Image Processing | OpenCV |
| Augmentation | albumentations |
| Data Manipulation | pandas |
| Visualisation | matplotlib |
| Dependency Management | `uv` (pyproject.toml + uv.lock) |

---

## Hardware Assumptions
- **Single local GPU** (limited VRAM).
- Default models: `yolov8n-seg` or `yolov8s-seg` — never anything larger without explicit instruction.
- **Mixed precision** is always ON (`amp=True`).
- **Batch size** is always configurable via CLI; default = 8.
- Every training / evaluation script **must** accept these CLI arguments with sensible defaults:
  - `--device` (default: `0` for first GPU, or `cpu`)
  - `--batch` (default: `8`)
  - `--epochs` (default: `50`)
  - `--imgsz` (default: `640`)

---

## Repository Layout
```
autoassess/
├── CLAUDE.md                  ← this file
├── README.md
├── pyproject.toml             ← uv-managed
├── uv.lock                    ← committed
├── .gitignore
├── configs/                   ← YAML experiment configs (never hardcode paths)
│   ├── base.yaml
│   └── yolov8_seg.yaml
├── data/                      ← GITIGNORED
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/                 ← EDA and figure generation ONLY
├── reports/
│   └── figures/
├── runs/                      ← experiment outputs (gitignored content)
│   └── <experiment_name>/
│       ├── weights/
│       └── logs/
├── scripts/                   ← one-off CLI scripts (preprocessing, download, etc.)
└── src/
    └── autoassess/
        ├── __init__.py
        ├── data/              ← dataset classes, loaders, splitting
        ├── models/            ← model definitions and wrappers
        ├── train/             ← training loops / Ultralytics wrappers
        ├── eval/              ← metrics, evaluation loops
        └── infer/             ← inference pipeline, visualisation helpers
```

---

## Coding Conventions

### Paths
- **All paths come from the YAML config**. Never hardcode a path string in source code.
- Use `pathlib.Path` everywhere; never `os.path.join`.
- Config is loaded once at entrypoint and passed down; do not re-load mid-run.

### Logging
- Every script logs to `runs/<experiment_name>/`.
- Use Python's `logging` module configured in a central `src/autoassess/utils/logging.py`.
- Log: start time, config snapshot, GPU info, per-epoch metrics, final summary.

### Reproducibility
- Seed **everything**: `random`, `numpy`, `torch`, `torch.cuda` (via a `seed_everything(seed)` utility).
- Default seed = `42`.
- Store the full config YAML in `runs/<experiment_name>/config.yaml` at run start.

### Type Hints
- All **public** functions and methods must have full type hints (parameters + return type).
- Internal / private helpers (`_foo`) are encouraged but not strictly required.

### CLI Scripts
- Use `argparse` (or `typer` if you prefer, but stay consistent).
- Always add `--help` descriptions to every argument.
- Arguments override config-file values where both exist.

### Notebooks
- Notebooks are for **EDA and figure generation only**.
- Anything reproducible (training, evaluation, preprocessing) goes in `src/` or `scripts/`, not notebooks.
- Keep notebooks clean: strip outputs before committing.

### Testing
- Unit tests go in `tests/` (to be created when needed).
- Use `pytest`.

---

## Key Design Decisions
1. **YOLOv8 first**: Start with Ultralytics YOLOv8n-seg / YOLOv8s-seg as the primary model. Mask R-CNN is an alternative/comparison model.
2. **Severity grading** is a secondary head or post-processing step, not a separate model.
3. **Data pipeline**: raw → interim (cleaned/annotated) → processed (model-ready, YOLO format).
4. **Experiment tracking**: use local logging + CSV metrics; MLflow or W&B can be added later.

---

## What NOT to Do
- Do not write model weights or checkpoints to anywhere except `runs/<name>/weights/`.
- Do not load data from absolute paths or user-home paths — always relative to `project_root` in config.
- Do not use `print()` for logging in src/ — use the `logging` module.
- Do not commit large binary files (datasets, weights) — add to `.gitignore`.
- Do not use notebooks for training or evaluation.

---

## Agent workflow
Implementation work is split between an orchestrator and workers.

- **Orchestrator** (main Claude Code session): plans, writes the dispatch
  for each task, reviews every diff, reruns the quality gates itself, and
  integrates. It does not write implementation code.
- **Worker** (`.claude/agents/sonnet-worker.md`, runs on Sonnet): receives
  exactly one scoped task per dispatch and edits only the files that task
  names. Every behaviour change gets a test that fails before and passes
  after, built on the synthetic COCO fixtures in `tests/conftest.py`. Reports
  raw output of `pytest`, `ruff check`, and `mypy --strict`.
- Workers never retrain models and never read-modify-write anything under
  `runs/` or `data/`. Tests use `tmp_path`.
- Tasks that touch disjoint files may run in parallel; anything sharing a
  file runs in sequence.
- GPU work (evaluating on the test split, training) runs on Colab through the
  `colab-mcp` server and is driven by the orchestrator only, never by a
  worker. Retraining needs explicit approval each time.
