# KITTI Monocular 3D Evaluation — Design

Date: 2026-11-08
Status: Draft (awaiting approval)

## Intent & why

The 3D stage (`ForegroundDepth3DRegressor`) can be trained in three feature
modes — `fusion`, `rgb`, `depth` — but there is currently no quantitative way
to tell them apart. Validation only tracks a normalised MSE loss, which is not
interpretable and not comparable to any published baseline.

We need an evaluation script, `eval_3d.py`, that produces **official
KITTI-style 3D and BEV Average Precision tables**. Two concrete uses, both
required:

1. **Ablation report** (primary, near-term): a table comparing `fusion` vs
   `rgb` vs `depth` so the thesis can show what each branch contributes.
2. **Model tracking**: run the same metric on any checkpoint to see whether a
   change helped.

The metric must speak the same language as MonoDETR (KITTI AP40, Easy/
Moderate/Hard, rotated IoU) from day one, so later improvements can be compared
to MonoDETR directly without redoing the numbers.

## Scope — this grew into an ordered sequence

Choosing the Chen split as canonical means `eval_3d.py` alone is not enough; the
models must be retrained on that split first. The work is now four ordered
pieces (brainstormed/planned one at a time):

1. **Split infrastructure.** Add Chen `ImageSets/train.txt` + `val.txt`; make
   `prepare_dataset.py` (RF-DETR COCO) and the 3D dataset split by these lists
   instead of the seed-42 random shuffle. Code + tests.
2. **`eval_3d.py`.** The evaluation tool described in this document. Can be
   built and unit-tested (synthetic IoU/AP cases) before any retraining; its
   end-to-end run needs piece 1 and new checkpoints.
3. **Retrain on the Chen train split.** RF-DETR on Chen train, then the 3 × 3D
   feature-mode models (`fusion`, `rgb`, `depth`). Mostly execution/compute, not
   new code.
4. **Produce the tables.** Run `eval_3d.py` on Chen val for each mode and emit
   the ablation + MonoDETR-comparable table.

This document specifies piece 2 in full and piece 1's contract (the split
source). Pieces 3–4 are execution steps. Dependencies: 1 → 3; (1,3) → 4; 2 is
independent until its end-to-end run.

## Goals

- Compute **3D AP** and **BEV AP** using the official KITTI protocol:
  - 40-point recall interpolation (AP40, the current KITTI standard).
  - Rotated 3D / BEV IoU.
  - Per evaluated class, per difficulty (Easy / Moderate / Hard).
  - Standard IoU thresholds: Car at 0.7 **and** 0.5; Pedestrian and Cyclist at
    0.5. (Only classes present in the checkpoint are reported.)
- Evaluate on the **standard KITTI (Chen et al. 3DOP) validation split** —
  `val.txt` = 3769 images, `train.txt` = 3712 images, partitioning the 7481
  training frames — so the AP numbers are directly comparable to published
  MonoDETR results. This split becomes the single canonical split shared by
  RF-DETR training, the 3D stage, and this evaluation.
- Score each detection with the **RF-DETR 2D confidence** (the detection
  pipeline is RF-DETR 2D proposals → frozen DINO features → 3D head).
- Emit results as (a) a human-readable table printed to the console and
  (b) a JSON file for archiving.
- Produce the ablation comparison: given several result JSONs (one per
  feature mode), print a single side-by-side markdown table ready to paste
  into the report.

## Non-goals / anti-goals

- **Not** the KITTI online test server. The official test set has no labels;
  we evaluate on our own held-out split. The design makes that explicit in the
  output so numbers are not mistaken for test-server AP.
- **Not** changing the model architecture or the detector design. (The dataset
  split logic **does** change — to the Chen split — and the models **must** be
  retrained on it; see "Scope" below. The current seed-42 checkpoints cannot be
  evaluated honestly on the Chen val split because their training data overlaps
  it.)
- **Not** adding a numba / `kitti-object-eval-python` dependency. We rely on
  `shapely` (already installed) plus numpy.
- AOS (orientation AP) and 2D-bbox AP are **optional stretch**, added only if
  they fall out cheaply; the core deliverable is 3D AP + BEV AP.
- **Anti-goal:** silently reporting AP that is not comparable to MonoDETR
  (wrong recall-point count, axis-aligned IoU, or wrong difficulty rules). If
  we cannot match the protocol, the script must say so rather than print a
  misleading number.

## Constraints & environment

- Python 3.10+, torch, Pillow, rfdetr[train]==1.11.2, shapely 2.1.2, numpy
  2.5.4, scipy 1.18.1. numba is **not** available and must not be required.
- Device selection mirrors the existing `device()` helper (CUDA → MPS → CPU),
  overridable with `--device`.
- Data layout matches `train_3d.py` defaults:
  - images `data/data_object_image_2/training/image_2`
  - labels `data/training/label_2`
  - calib  `data/data_object_calib/training/calib`
- The seed-42 random split in `train_3d.py` / `prepare_dataset.py` is
  **replaced** by the Chen `ImageSets` lists. Evaluation reads `val.txt` and
  selects exactly those frames (that also have label + 12-value P2).

## The how — decided

### Prediction pipeline (per validation image)
1. Run RF-DETR (`RFDETRNano.predict`) at a low score threshold (`--score-thr`,
   default 0.05 so AP integrates over the full PR curve) → 2D boxes, scores,
   class names.
2. Keep detections whose class is in the checkpoint's `classes`.
3. Build the same `boxes` / `geometry` / `class_ids` tensors as `infer_3d.py`,
   run `RFDETRDinoAdapter` + `ForegroundDepth3DRegressor`, de-normalise with the
   checkpoint's `target_mean` / `target_std`.
4. Decode to KITTI parameters: `(h, w, l, x, y, z, ry=atan2(sin, cos))`, with
   `score` = RF-DETR confidence.

### Ground truth & difficulty
- Extend the shared `_labels` parser (in `prepare_dataset.py`) to also keep
  `truncation` (field 1) and `occlusion` (field 2). Additive keys; existing
  callers read fixed keys, so this is backward-compatible.
- Assign KITTI difficulty from bbox pixel height, occlusion, truncation:
  - Easy: height ≥ 40, occlusion ≤ 0, truncation ≤ 0.15
  - Moderate: height ≥ 25, occlusion ≤ 1, truncation ≤ 0.30
  - Hard: height ≥ 25, occlusion ≤ 2, truncation ≤ 0.50
- Matching is **not** strict same-class. Following the KITTI devkit, a
  detection of evaluated class `C` may match a GT box whose class is `C` **or a
  neighbour of `C`**; a match landing on a neighbour (or on an ignored GT) is
  **don't-care** — it is removed from the detection pool and counts as neither
  TP nor FP. Neighbours: Car↔Van, Pedestrian↔Person_sitting, Cyclist has none.
  GT of any other non-evaluated class (Truck, Tram, Misc) is a normal box: a
  detection overlapping it with no valid same-class/neighbour GT is a false
  positive. DontCare regions are ignored via 2D-IoU overlap. GT below the
  current difficulty's thresholds are treated as don't-care (ignored), not
  counted as false negatives.

### IoU
- BEV IoU: build the two rotated ground rectangles from `(x, z, l, w, ry)` and
  intersect with `shapely.Polygon.intersection`.
- 3D IoU: `bev_intersection_area × height_overlap / (vol_a + vol_b − inter_vol)`,
  with vertical overlap from the `y`/`h` extents (KITTI `y` = box bottom).

### AP40
- Replicate the KITTI devkit `eval.cpp` algorithm exactly (the one MonoDETR's
  numbers come from), in pure numpy — **not** a plain PR-curve interpolation:
  - Per class/difficulty/IoU-threshold, greedily assign detections (score desc)
    to unmatched GT per the matching + don't-care rules above.
  - Derive the 40 score thresholds with the devkit `getThresholds` walk over
    the sorted TP scores (recall positions 1/40 … 40/40, recall=0 excluded),
    not evenly spaced score values.
  - At each threshold compute precision, take the running max (monotone
    envelope), and report `AP = sum(precisions) / 40`.
- The IoU/difficulty/neighbour logic is shared across the 2D-matching used to
  pick the greedy assignment, so a detection matches GT by the metric under
  test (3D IoU for 3D AP, BEV IoU for BEV AP).

### CLI & output
- `eval_3d.py --3d-checkpoint <pt> --rfdetr-checkpoint <pth> [--device] [--seed 42]`
  `[--score-thr 0.05] [--image-dir ...] [--label-dir ...] [--calib-dir ...]`
  `[--output runs/monocular-3d/eval-<mode>.json] [--limit N]`.
  - Prints a table: rows = class × IoU threshold, columns = Easy/Mod/Hard, for
    both 3D AP and BEV AP; writes the same to JSON (plus metadata: checkpoint
    path, feature_mode, split seed, image count, "held-out split, not KITTI
    test server").
  - `--limit N` evaluates the first N valid images (fast smoke / tracking).
- `eval_3d.py --compare a.json b.json ...` → prints a single markdown table
  keyed by feature_mode for the ablation report; no model is loaded.

## Testing (TDD)

Unit (no torch/rfdetr needed where possible):
- Rotated BEV IoU and 3D IoU on hand-computed boxes (identical, disjoint,
  half-overlap, rotated 90°).
- AP40 on synthetic detections with a known precision/recall curve (perfect
  detector → AP = 1; a known partial case → known value).
- Difficulty assignment across the boundary values of height/occlusion/
  truncation.
- `_labels` now returns truncation/occlusion and still parses the existing
  fields; the sample line `Car 0.00 0 1.58 ...` → truncation 0.0, occlusion 0.
- Split reconstruction returns the identical valid indices as `train_3d.py`
  for a given seed.
- `--compare` builds the expected markdown table from two small JSONs.

Integration (guarded so it skips without checkpoints): a `--limit`ed run on a
couple of real frames produces a JSON with the expected schema.

## What's left to the builder

- Exact table formatting and column order (follow the MonoDETR convention:
  3D AP block then BEV AP block).
- Module layout: a small pure-logic module (IoU, AP40, difficulty, split) that
  the tests import, with `eval_3d.py` as the torch/rfdetr entry point — mirrors
  how `monocular_3d.py` / `train_3d.py` are split today.
- Whether to also emit a one-line `mean 3D AP (Moderate)` summary for quick
  tracking.
