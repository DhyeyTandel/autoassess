# Decisions

Append-only log of architectural decisions for this project.
Entries are added via the `/log` command. Do not rewrite or
tidy earlier entries.

## [2026-10-05] T4 re-score for VRAM/latency; init CUDA before peak-VRAM reset

**Problem** — `reports/results.md` had test-split scores from a local MPS run, but Peak VRAM was `n/a` (MPS has no peak-memory counter) and latency was Apple-silicon latency, not the T4 the report targets.

**Options considered**
- Re-run `autoassess-eval --device 0` on a Colab T4 for all three trained models (plus the carparts/vehide `--exclude-list` variants) and overwrite only the eval `metrics.json` files. Chosen.
- Report MPS current-allocated memory as a VRAM stand-in. Rejected: `peak_vram_mb` deliberately returns None on MPS because current allocation understates the peak.
- Leave VRAM as `n/a`. Rejected: the hardware-assumptions section of the report needs a number.

**Decision** — Ran the five evals on a T4 via colab-mcp, with ultralytics pinned to the `uv.lock` version (8.4.115). Colab's `pip install -e .` had resolved 8.4.173, and different postprocessing would have confounded the MPS-vs-T4 mAP check. Copied the metrics back, verified by a canonical-JSON SHA-256 per file, and regenerated `results.md`. Fixed the evaluator crash this exposed: `reset_peak_vram` and `reset_peak_vram_for_yolo` now call `torch.cuda.init()` before `torch.cuda.reset_peak_memory_stats`.

**Tradeoff accepted** — Latency comes from 30 single images, so it is noisy. On the T4, VehiDE-excluded came out slower than on MPS (25.3 → 29.6 ms) and CarDD was unchanged (30.4 ms). These are rough figures, not a benchmark. torch was not pinned (Colab 2.11+cu130 vs local 2.13), only ultralytics.

**What went wrong**
- The first T4 eval crashed with `RuntimeError: Invalid device argument` at `torch.cuda.reset_peak_memory_stats(device)`, called from `evaluate.py:281`. That function does not lazy-init CUDA, and the YOLO model was still on CPU. This path had never run, because every earlier run was on MPS. I reproduced it in isolation on the T4 before fixing it.
- The handoff doc said the data bundle was ~100 MB. The test split alone zipped to 645 MB, and full `data/processed` is ~5.2 GB.
- The connected Colab runtime started CPU-only (`nvidia-smi: command not found`); the user had to switch to a T4.
- The user uploaded the unzipped folder rather than the zip. That was fine, but my first file-count check returned 0 because IPython's `$` expansion broke a shell `for` loop. I recounted in Python: counts matched.
- The fixed evaluator was applied to the Colab clone as a patch before it was committed, so the T4 numbers come from code that matched commit `b96e816` but was not that commit at run time.

**Scale/limits** — mAP agrees with MPS within 0.003 (Carparts identical), which is GPU nondeterminism. Inference peak VRAM: 227 MB (yolov8n-seg parts), 310 MB (yolov8n-seg CarDD), 294 MB (yolov8s-seg VehiDE) at imgsz 640, batch 1. The `torch.cuda.init()` fix is untested on multi-GPU.

## [2026-10-05] Overlay renderer moved to autoassess.infer.visualize with readable labels

**Problem** — Demo overlay labels were unreadable. They used PIL's default ~10 px bitmap font at any image size, coloured text with no background (yellow and orange vanished on bright paint), and a fixed `(x0+2, y0-14)` position, so nearby damages' labels overlapped.

**Options considered**
- Fix `draw_overlay` in place in `app.py`. Rejected: `app.py` builds Gradio Blocks at import and had no tests, so the layout logic would stay untested.
- Move the renderer to `src/autoassess/infer/visualize.py` (where CLAUDE.md puts visualisation helpers), with a pure `layout_labels` function. Chosen.

**Decision** — Font size is `max(14, round(0.028 * min(w, h)))`, with outline width scaled to it. Labels are opaque severity-coloured tags with near-black text, drawn after the alpha composite. Layout is greedy, largest box first: try above the box, then inside it, then below it, then shift down until there is no collision, clamped to the frame. Label text is `type · severity · part`, with "unassigned" shown as "no part".

**Tradeoff accepted** — On crowded images (13–15 instances on CarDD `000042`), the shift-down fallback puts some labels well below their boxes, detached from the damage they describe. There are no leader lines.

**What went wrong** — nothing at implementation time. Earlier in the session I had planned to build a new Gradio demo, because my search for an existing one (`grep --include=*.py`) failed under zsh (`no matches found`) and returned nothing. `app.py` already existed at the repo root; I found it only by `ls`.

**Scale/limits** — Readable up to about 10 labels on a 1000 px image. Past that, labels pile up below the crowded region.

## [2026-10-05] Merge VehiDE into CarDD for damage types CarDD lacks

**Problem** — CarDD's 6 classes have no category for torn or detached parts. On CarDD test `000042` the hanging front-right bumper had no ground-truth annotation, and the CarDD model detected nothing there even at conf 0.01. VehiDE has `torn`, `missing_part` and `punctured`, but is weaker overall (mask mAP50 0.43 vs 0.62).

**Options considered**
- Lower VehiDE's cutoff in the demo so the bumper shows. Rejected: the bumper scored only 0.072 as `missing_part`, and a VehiDE test-split sweep showed precision collapsing at low cutoffs (torn P0.17 at 0.05).
- Retrain VehiDE (it stopped at 48/50 epochs). Deferred: needs Colab GPU time and explicit approval.
- Run both models and merge, using each where it is stronger. Chosen.

**Decision**
- CarDD is primary for all its classes; it is clearly stronger on the shared types (dent AP50 0.53 vs 0.24).
- VehiDE contributes only `torn`, `missing_part` and `punctured` (configurable in the `merge:` section of `configs/severity.yaml`).
- Secondary instances are processed by descending confidence. One is kept only if its mask IoU is below 0.5 against every primary instance and every secondary instance already kept, regardless of class.
- Secondary cutoff is 0.15 from the sweep (missing_part P0.59/R0.69 at 0.15 vs P0.37/R0.75 at 0.05).
- Records carry `source_model`. The demo defaults to `DAMAGE_MODEL_SOURCE=merged`.
- Severity weights for the new classes (torn 1.4, missing_part 1.8, punctured 1.5) are hand-picked priors, the same as the existing ones. [unverified] They are not fit to anything.

**Tradeoff accepted**
- The case that motivated this (the `000042` bumper corner) is still missed at the 0.15 default. The merge adds the capability but does not fix that image.
- Two models per image roughly doubles demo inference time.
- The secondary dedup keeps the higher-confidence label, which can choose the less apt class: on a VehiDE test photo it kept `punctured` over `missing_part` for the hole a fog-lamp housing came out of.
- The sweep used box IoU, not mask IoU.

**What went wrong**
- My first merge spec suppressed secondary detections only against primary ones. On real photos VehiDE stacked `punctured` + `missing_part` on one hole and five overlapping `torn` masks on one grille, because Ultralytics NMS is per-class. I caught it by looking at the demo output and added cross-class dedup in a second change.
- The confidence-sweep script failed twice first:
  - once on `OSError: broken data stream when reading image file` (two corrupt VehiDE test images: `25032020_091214image992948.jpg`, `25032020_091232image529852.jpg`);
  - then with `RuntimeError: Invalid buffer size: 21.25 GiB`, because passing a 1,742-path list to `predict()` batched it all at once.
- The `missing_part` on `000042` scored 0.15 through the app but 0.072 on the raw file. The app re-saves uploads as JPEG before inference. [unverified] That is the assumed cause.

**Scale/limits** — Works best on VehiDE-domain photos (close-ups of Vietnamese insurance claims). On CarDD-style photos VehiDE adds plausible but unconfirmed `torn` detections. There is no ground truth for evaluating the merged output as a whole.

## [2026-10-05] Damage-to-part matching by coverage instead of IoU; parts conf 0.25 → 0.15

**Problem** — Most damage in the demo showed "no part". Two causes: (a) matching used mask IoU, which penalises small damage on a large panel even when the damage lies entirely inside it; (b) on CarDD photos the parts model is under-confident (bonnet at 0.15, windshield at 0.25), so the 0.25 cutoff dropped panels. A third cause can't be fixed here: the 6-class parts model has no fender, quarter panel or grille.

**Options considered**
- Lower only the IoU threshold. Rejected: IoU's dependence on panel size is the bug, and any fixed threshold mis-ranks small vs large damage.
- Coverage = area(damage ∩ part) / area(damage), threshold 0.5. Chosen.
- Retrain the parts model on more of Carparts-Seg's 23 classes. Deferred: needs approval, and the source has no fender class anyway.

**Decision** — Coverage ≥ 0.5, best coverage wins, ties to the higher part confidence; zero coverage is never assigned. Parts cutoff default 0.15. The record field `part_iou` is renamed `part_coverage`, and `--iou-threshold` becomes `--coverage-threshold` (a breaking change for JSON consumers).

**Tradeoff accepted**
- Parts precision on its own test split drops slightly at 0.15 (box P 0.873 → 0.841, R 0.967 → 0.971).
- Coverage assigns a damage to a panel even when it spills well beyond it: 0.56 coverage put fender-area dents on `000513` onto `front_bumper`, which is arguably wrong.

**What went wrong** — nothing in implementation. The first diagnosis on `000042` showed matching was not the main problem there: 8 of 9 damages had zero overlap with any detected part, and the parts model labelled the front bumper `rear_bumper` (0.71).

**Scale/limits**
- On the first 150 CarDD test images (431 damage instances), the share assigned to a part: 14.4% (IoU ≥ 0.10, conf 0.25) → 19.5% (coverage, conf 0.25) → 25.8% (coverage, conf 0.15).
- About 74% still unassigned, mostly on panels with no class.
- The assignment rate measures coverage, not accuracy: CarDD has no part ground truth, so correctness was only spot-checked by eye on two images.

## [2026-10-05] Demo-only VehiDE cutoff 0.07 (high recall); lossless upload save

**Problem** — The user wanted the demo to flag the hanging/detached front-right bumper on CarDD test `000042`. VehiDE scores it `missing_part` at 0.072, below the 0.15 secondary cutoff logged in the merge entry above. Once the cutoff was lowered, the demo still missed it, while the CLI did not.

**Options considered**
- Lower the demo's VehiDE cutoff to 0.07. Chosen by the user, knowing the precision cost.
- Retrain VehiDE (longer, larger imgsz). Not chosen now: needs Colab GPU time and per-run approval, with no guarantee it fixes this image.
- Retrain CarDD. Rejected: CarDD's ground truth never labels this bumper, so it cannot learn it.

**Decision**
- `app.py` default `SECONDARY_DAMAGE_CONF` is now 0.07, documented as a high-recall demo setting.
- `autoassess-pipeline` keeps 0.15. This reverses the merge entry's default for the demo only.
- `app.py` now saves uploads as RGB PNG to a per-request `tempfile.mkstemp` file, deleted in a `finally`. It previously wrote JPEG (PIL default quality 75) to a fixed `/tmp/autoassess_upload.jpg`.

**Tradeoff accepted**
- On VehiDE test, missing_part precision falls from 0.59 to about 0.45, and torn from 0.33 to about 0.20. Roughly half the extra VehiDE detections are false alarms; `000042` goes from 14 to 18 instances.
- The cutoff sits 0.002 below this image's score, so it is tuned to one example and will flip on small pixel changes. That is a demo convenience, not a calibrated operating point.

**What went wrong**
- After lowering the cutoff, the live app still missed the bumper. Cause: the JPEG re-encode alone moved the score from 0.072 to 0.066 (q75; 0.067 at q95), while a lossless PNG re-save kept 0.072.
- The same sensitivity runs the other way: after the PNG change, the parts model lost the bonnet on `000042` (it was at the 0.15 cutoff). The bonnet dent and scratch went back to "no part".
- The fixed `/tmp` path also let concurrent requests overwrite each other's upload. A worker had flagged this earlier, and it was fixed in the same change.

**Scale/limits** — Any detection within about 0.01 of a cutoff is unstable across re-encodes, resizes and phone-camera processing. This applies to the parts model at 0.15 as much as to VehiDE at 0.07. Uploads are still decoded in full and written once per request, which is fine for a single-user demo.
