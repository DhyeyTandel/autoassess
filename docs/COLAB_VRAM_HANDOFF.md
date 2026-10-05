# Handoff: T4 VRAM + latency re-score via colab-mcp

Written 2026-10-05 at the end of the eval-fix session. Branch: `fix/eval-correctness`.

## State
- All eval fixes are committed (T0-T7). See `git log main..fix/eval-correctness`.
- Test-split scores for CarDD (`yolov8_seg_v1`), Carparts (`parts_seg_v1`) and VehiDE
  (`vehide_seg_v1`) were produced locally on MPS and are in `reports/results.md`.
- Leak-excluded variants: `runs/parts_seg_v1_eval_test_excl`, `runs/vehide_seg_v1_eval_test_excl`.
  Exclude lists are in `reports/split_audit/<dataset>_flagged_test.txt`.
- Missing: `Peak VRAM` (MPS has no peak counter). Latency is MPS latency, not T4.

## What the new session should do
1. Confirm the colab-mcp tools loaded and a notebook is open on a T4 runtime.
2. The user pushes the branch first: `git push -u origin fix/eval-correctness`.
3. Get data and weights onto the runtime. `runs/` and `data/` are gitignored.
   - Bundle locally: `data/processed/{cardd,carparts,vehide}` (~100 MB) and the three
     `runs/<name>/weights/best.pt` plus `runs/<name>/metrics.json` (~100 MB).
   - `zip -r` follows symlinks by default; check the processed image dirs resolve.
   - Upload to Drive, mount it in Colab, and unzip into the cloned repo.
4. In Colab: clone the branch, `pip install -e .`, then for each model:
   ```
   autoassess-eval --device 0 --split test --weights runs/yolov8_seg_v1/weights/best.pt --dataset-config configs/cardd.yaml
   autoassess-eval --device 0 --split test --weights runs/parts_seg_v1/weights/best.pt --dataset-config configs/carparts.yaml --model-type-label yolov8-seg-parts
   autoassess-eval --device 0 --split test --weights runs/vehide_seg_v1/weights/best.pt --dataset-config configs/vehide.yaml
   ```
   Add the `--exclude-list reports/split_audit/<dataset>_flagged_test.txt` variants for
   carparts and vehide if T4 latency/VRAM is wanted on those too.
5. Copy the `runs/*_eval_test*/metrics.json` files back over the local MPS ones,
   run `uv run python scripts/generate_results_md.py`, and check that the mAPs match the
   MPS run (they should, up to GPU nondeterminism). Only latency and VRAM should change.
6. Do not retrain anything. Mask R-CNN is still untrained.

## Known nits (not fixed)
- `params_total` from the evaluator (3,259,234 for yolov8n-seg) differs from the
  training-time count (3,264,786). The evaluator counts the fused inference model.
- `compare.py` does not show the leak-excluded rows; only `results.md` does.
