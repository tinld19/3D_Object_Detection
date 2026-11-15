# KITTI Monocular 3D Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the KITTI pipeline to the standard Chen train/val split and add `eval_3d.py`, which reports official KITTI 3D AP and BEV AP (AP40) comparable to MonoDETR.

**Architecture:** A pure-logic module `kitti_eval.py` (rotated IoU via shapely, KITTI difficulty, devkit matching with neighbour/don't-care rules, AP40, and the ablation-comparison table) that needs neither torch nor rfdetr, so it is unit-testable in isolation. A thin entry point `eval_3d.py` loads the RF-DETR + 3D-head checkpoints, runs them over the Chen val frames exactly as `infer_3d.py` does, assembles predictions and ground truth, and calls `kitti_eval`. The split moves from the seed-42 random shuffle to committed Chen `ImageSets` lists, shared by `prepare_dataset.py` (RF-DETR COCO), `train_3d.py`, and `eval_3d.py`.

**Tech Stack:** Python 3.10+, numpy, shapely 2.1.2, torch + rfdetr[train]==1.11.2 (entry point only), unittest.

**Spec:** `docs/superpowers/specs/2026-11-08-kitti-3d-eval-design.md`

## Global Constraints

- Evaluation split is the Chen `ImageSets`: `train.txt` = 3712 ids, `val.txt` = 3769 ids, partitioning 7481 frames. Committed under `kitti_imagesets/` (repo-tracked; `data/` is gitignored).
- Image ids are 6-digit zero-padded strings, one per line (e.g. `000123`).
- No new heavyweight dependency: IoU uses shapely (installed); AP40 is pure numpy. numba must not be required.
- Evaluated classes and IoU thresholds: Car at IoU 0.7 **and** 0.5; Pedestrian at 0.5; Cyclist at 0.5. Only classes present in the 3D checkpoint are reported.
- Difficulty (bbox pixel height / occlusion / truncation): Easy ≥40 / ≤0 / ≤0.15; Moderate ≥25 / ≤1 / ≤0.30; Hard ≥25 / ≤2 / ≤0.50.
- Neighbours (a detection matching one is don't-care, not FP): Car↔Van, Pedestrian↔Person_sitting, Cyclist none.
- AP40 replicates the KITTI devkit `getThresholds` behaviour: 40 recall positions 1/40…40/40 (recall 0 excluded), monotone precision envelope, `AP = sum(precisions)/40`.
- KITTI label field order: `class trunc occ alpha x1 y1 x2 y2 h w l X Y Z ry`. Dimensions kept as `[h, w, l]`, location `[X, Y, Z]`.
- Data dirs (defaults, as in `train_3d.py`): images `data/data_object_image_2/training/image_2`, labels `data/training/label_2`, calib `data/data_object_calib/training/calib`.

## Review Focus

- **Val frame with no label file or only DontCare/empty labels:** eval must treat it as an image with zero valid GT (predictions on it become FPs), not crash or skip silently. → test in Task 9.
- **Degenerate predicted cuboid (z ≤ 0 or zero-area footprint):** rotated-IoU must return 0.0, never raise or return NaN. → test in Task 5.
- **Shapely intersection returning an empty / non-polygon geometry:** area must read as 0.0. → test in Task 5.
- **A class with zero valid GT at a difficulty (e.g. no Cyclist in val subset):** AP for that cell is reported as `nan`/omitted, not `0.0`, so it is not mistaken for a real miss. → test in Task 8.
- **Predicted class present in checkpoint but not an evaluated class (Van/Truck):** excluded from its own AP, yet Van GT still serves as a Car neighbour. → test in Task 7.

---

### Task 1: Keep truncation and occlusion in the label parser

**Files:**
- Modify: `prepare_dataset.py` (`_labels`)
- Test: `tests/test_prepare_dataset.py`

**Interfaces:**
- Produces: `_labels(path, width, height)` entries now also carry `"truncation": float` (field 1) and `"occlusion": int` (field 2). Existing keys unchanged.

- [ ] **Step 1: Write the failing test** — `test_labels_keep_truncation_and_occlusion`: write a one-line label file `Car 0.15 1 1.58 539.13 182.36 566.23 206.67 1.41 1.59 4.47 -3.59 2.02 45.01 1.50`, parse with a 1242×375 size, assert the single entry has `truncation == 0.15` and `occlusion == 1` alongside the existing `class_name == "Car"`.
- [ ] **Step 2: Run it, expect KeyError/assertion failure.** Run: `python -m unittest tests.test_prepare_dataset -v`
- [ ] **Step 3: Add the two keys in `_labels`** — read `float(fields[1])` and `int(float(fields[2]))`.
- [ ] **Step 4: Run the full `tests.test_prepare_dataset`, expect PASS** (existing conversion tests must still pass — keys are additive).
- [ ] **Step 5: Commit** (`feat: keep truncation/occlusion in KITTI label parser`).

---

### Task 2: Commit the Chen split and add a split reader

**Files:**
- Create: `kitti_imagesets/train.txt`, `kitti_imagesets/val.txt`
- Modify: `prepare_dataset.py` (add `_read_split`)
- Test: `tests/test_prepare_dataset.py`

**Interfaces:**
- Produces: `_read_split(path: Path) -> list[str]` — ordered, de-duplicated 6-digit stems; raises `FileNotFoundError` if missing.

- [ ] **Step 1: Fetch and commit the real lists** — download the standard Chen split (e.g. `https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/{train,val}.txt`) into `kitti_imagesets/`. Verify locally: `train.txt` has 3712 ids, `val.txt` has 3769, disjoint, union = 7481, every line matches `^\d{6}$`.
- [ ] **Step 2: Write the failing test** — `test_read_split_parses_ids` (tmp fixture of three ids → `["000000","000003","000007"]`) and `test_committed_chen_split_partitions_7481`: read both committed files, assert lengths 3712 / 3769, `set(train).isdisjoint(val)`, union size 7481, all ids match `^\d{6}$`.
- [ ] **Step 3: Run it, expect failure** (`_read_split` undefined).
- [ ] **Step 4: Implement `_read_split`** — strip, drop blanks, preserve order, de-dup.
- [ ] **Step 5: Run the tests, expect PASS.**
- [ ] **Step 6: Commit** (`feat: add Chen KITTI ImageSets split and reader`).

---

### Task 3: Split the RF-DETR COCO conversion by ImageSets

**Files:**
- Modify: `prepare_dataset.py` (`convert_dataset`, `main`)
- Test: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: `_read_split` (Task 2).
- Produces: `convert_dataset(image_dir, label_dir, calib_dir, output_dir, split_dir: Path)` — assigns each frame to `train`/`valid` by membership in `split_dir/train.txt` vs `split_dir/val.txt`; frames in neither are skipped. The `seed`/`val_fraction` random split is removed. `main` gains `--split-dir` (default `Path("kitti_imagesets")`).

- [ ] **Step 1: Write the failing test** — `test_convert_splits_by_imagesets`: build a tmp tree with 3 synthetic PNG+label+calib frames `000000/000001/000002`, a `split_dir` with `train.txt`={000000,000002}, `val.txt`={000001}; run `convert_dataset`; assert `train/_annotations.coco.json` has images for 000000+000002 and `valid` has 000001.
- [ ] **Step 2: Run it, expect failure** (signature mismatch).
- [ ] **Step 3: Rewrite `convert_dataset`'s split** — replace the `random.Random(seed).shuffle` + `val_fraction` slice with ImageSets membership; keep COCO writing unchanged.
- [ ] **Step 4: Run the full `tests.test_prepare_dataset`, expect PASS.**
- [ ] **Step 5: Commit** (`feat: split RF-DETR COCO by Chen ImageSets`).

---

### Task 4: Split the 3D stage by ImageSets

**Files:**
- Modify: `train_3d.py` (extract split into a function, call `_read_split`)
- Test: `tests/test_train_3d.py`

**Interfaces:**
- Consumes: `_read_split` (Task 2).
- Produces: `split_indices(stems: list[str], val_ids: set[str]) -> tuple[list[int], list[int]]` — returns `(train_indices, valid_indices)` into `stems`; a stem in `val_ids` goes to valid, everything else to train; order preserved. `main` gains `--split-dir` (default `Path("kitti_imagesets")`) and drops the seed-based shuffle for the split (seed still seeds torch).

- [ ] **Step 1: Write the failing test** — `test_split_indices_partitions_by_val_ids`: `stems=["000000","000001","000002","000003"]`, `val_ids={"000001","000003"}` → `([0,2],[1,3])`; and a stem absent from both still lands in train.
- [ ] **Step 2: Run it, expect failure** (`split_indices` undefined).
- [ ] **Step 3: Implement `split_indices`** and wire `main` to build `stems` from `dataset.samples[i]["image"].stem`, read `val.txt` via `_read_split`, and derive train/valid indices from it; normaliser still fits on `train_indices`.
- [ ] **Step 4: Run `tests.test_train_3d`, expect PASS** (keep the existing `set_seed` test green).
- [ ] **Step 5: Commit** (`feat: split 3D stage by Chen ImageSets`).

---

### Task 5: Rotated BEV and 3D IoU

**Files:**
- Create: `kitti_eval.py`
- Test: `tests/test_kitti_eval.py`

**Interfaces:**
- Produces:
  - `bev_iou(a, b) -> float` where a box is `(x, z, l, w, ry)` (camera BEV; at `ry=0` footprint spans `l` in x and `w` in z, per `infer_3d.project_cuboid`).
  - `iou_3d(a, b) -> float` where a box is `(h, w, l, x, y, z, ry)`; vertical extent is `[y - h, y]` (KITTI `y` = box bottom). `3D IoU = inter_area * h_overlap / (vol_a + vol_b - inter_area * h_overlap)`.
  - Both return `0.0` for degenerate or non-overlapping boxes and never raise.

- [ ] **Step 1: Write the failing tests** —
  - `test_bev_iou_identical_is_one`: identical box → `1.0`.
  - `test_bev_iou_disjoint_is_zero`: far-apart boxes → `0.0`.
  - `test_bev_iou_half_overlap`: A`(0,0,4,2,0)`, B`(2,0,4,2,0)` → `≈ 4/12` (assertAlmostEqual 0.3333).
  - `test_bev_iou_degenerate_returns_zero`: a box with `l=0` → `0.0` (no raise).
  - `test_iou_3d_identical_is_one` and `test_iou_3d_half_height_overlap`: same footprint, `h=2` boxes offset in `y` so vertical overlap is 1 of 2 → `≈ 1/3`.
- [ ] **Step 2: Run them, expect failure** (module/functions undefined).
- [ ] **Step 3: Implement** using `shapely.geometry.Polygon` for the rotated footprint intersection (guard empty/invalid geometry → area 0.0) and numpy for volumes.
- [ ] **Step 4: Run `tests.test_kitti_eval`, expect PASS.**
- [ ] **Step 5: Commit** (`feat: rotated BEV/3D IoU for KITTI eval`).

---

### Task 6: KITTI difficulty assignment

**Files:**
- Modify: `kitti_eval.py`
- Test: `tests/test_kitti_eval.py`

**Interfaces:**
- Produces: `object_difficulty(bbox_height_px: float, occlusion: int, truncation: float) -> int | None` — `0` easy, `1` moderate, `2` hard, `None` if below Hard (ignored). An object qualifying for Easy also satisfies Moderate/Hard; callers evaluate difficulty `d` over all GT with `object_difficulty <= d`.

- [ ] **Step 1: Write the failing tests** covering the boundaries: `(41,0,0.10)→0`, `(40,0,0.15)→0`, `(30,1,0.20)→1`, `(26,2,0.40)→2`, `(24,0,0.0)→None`, `(50,3,0.0)→None`, `(50,0,0.60)→None`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement `object_difficulty`** (check Easy, then Moderate, then Hard thresholds; all three conditions per level).
- [ ] **Step 4: Run `tests.test_kitti_eval`, expect PASS.**
- [ ] **Step 5: Commit** (`feat: KITTI difficulty assignment`).

---

### Task 7: Per-image matching with neighbour / don't-care rules

**Files:**
- Modify: `kitti_eval.py`
- Test: `tests/test_kitti_eval.py`

**Interfaces:**
- Consumes: an `iou_fn(pred_box, gt_box) -> float` (Task 5), `object_difficulty` (Task 6).
- Produces: `match_image(preds, gts, iou_fn, iou_thr, difficulty, eval_class, neighbours) -> tuple[list[tuple[float, bool]], int]` — matching is **within a single image** (a detection can only match ground truth in its own image; `evaluate` in Task 8 calls this per image and pools the results).
  - `preds`: this image's detections of the evaluated class, each `{"score", "box"}`; `gts`: this image's ground truth, each `{"box", "class", "difficulty"}` (`difficulty` is `object_difficulty` or `None`).
  - Greedy by score desc; a pred may match GT of `eval_class` or of a `neighbours[eval_class]` class. Matching an evaluated-class GT above `iou_thr` whose `difficulty <= difficulty` → TP; matching a neighbour GT, an ignored-difficulty GT, or a too-small/DontCare GT → **removed** (neither TP nor FP). Unmatched preds → FP `(score, False)`.
  - Returns `(scored_flags, num_valid_gt)` where `num_valid_gt` counts only this image's evaluated-class GT with `difficulty <= difficulty`.
- `NEIGHBOURS = {"Car": {"Van"}, "Pedestrian": {"Person_sitting"}, "Cyclist": set()}`.

- [ ] **Step 1: Write the failing tests** —
  - `test_match_counts_tp_and_fp`: 1 GT Car (easy), 2 preds (one overlapping ≥thr, one not) → flags `[(s_hi,True),(s_lo,False)]`, `num_valid_gt=1`.
  - `test_match_on_neighbour_is_dontcare`: pred Car overlapping only a Van GT → no TP, no FP (pred removed), `num_valid_gt=0`.
  - `test_match_on_other_class_is_fp`: pred Car overlapping only a Truck GT → FP, `num_valid_gt=0`.
  - `test_match_ignores_too_hard_gt`: GT with `difficulty=None` consumes a matching pred as don't-care and is not counted in `num_valid_gt`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement `match_class`** (sort desc; one-to-one greedy; prefer highest-IoU unmatched GT; apply the removal rules above).
- [ ] **Step 4: Run `tests.test_kitti_eval`, expect PASS.**
- [ ] **Step 5: Commit** (`feat: KITTI devkit matching with neighbour/don't-care`).

---

### Task 8: AP40 and the per-checkpoint evaluation

**Files:**
- Modify: `kitti_eval.py`
- Test: `tests/test_kitti_eval.py`

**Interfaces:**
- Consumes: `match_image` (Task 7).
- Produces:
  - `ap40(scored_flags: list[tuple[float, bool]], num_valid_gt: int) -> float` — devkit `getThresholds` over 40 recall positions on the **pooled** flags; `nan` if `num_valid_gt == 0`.
  - `evaluate(images, classes, iou_thresholds, metrics) -> dict` where `images` is a list of per-frame `{"preds", "gts"}` (preds/gts as in Task 7, boxes in the representation each metric's `iou_fn` expects). For each class × metric × iou_thr × difficulty it calls `match_image` on every frame, concatenates the `scored_flags` and sums `num_valid_gt` across frames, then `ap40`. Returns `result[class][metric][iou_thr][difficulty] = ap` for `metric in {"3d","bev"}`, `difficulty in {"easy","moderate","hard"}`.

- [ ] **Step 1: Write the failing tests** —
  - `test_ap40_perfect_is_one`: all-TP flags with `num_valid_gt = n` → `1.0`.
  - `test_ap40_half_recall`: 1 TP, `num_valid_gt=2` → `0.5` (20 of 40 recall points reachable at precision 1).
  - `test_ap40_fp_before_tp`: flags `[(0.95,False),(0.9,True)]`, `num_valid_gt=1` → `0.5` (precision 0.5 at the only reachable recall).
  - `test_ap40_no_gt_is_nan`: `num_valid_gt=0` → `math.isnan`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement `ap40`** (replicate devkit `getThresholds`: walk TP scores, sample 40 recall positions, monotone precision envelope, mean over 40) and `evaluate` (loop classes × metrics × thresholds × difficulties calling `match_class` + `ap40`).
- [ ] **Step 4: Run `tests.test_kitti_eval`, expect PASS.**
- [ ] **Step 5: Commit** (`feat: AP40 and per-checkpoint KITTI evaluation`).

---

### Task 9: `eval_3d.py` entry point, tables, and `--compare`

**Files:**
- Create: `eval_3d.py`
- Modify: `kitti_eval.py` (add `format_table`, `format_comparison`)
- Test: `tests/test_kitti_eval.py`, `tests/test_eval_3d.py`

**Interfaces:**
- Consumes: `evaluate` (Task 8); `RFDETRDinoAdapter`, `ForegroundDepth3DRegressor` (`monocular_3d`); `_calibration`, `_labels`, `_read_split` (`prepare_dataset`); decoding identical to `infer_3d.main` (`prediction * std + mean`, `ry = atan2(sin, cos)`).
- Produces:
  - `kitti_eval.format_table(result, feature_mode) -> str` — console/markdown table (3D AP block then BEV AP block; rows class×IoU, cols Easy/Mod/Hard).
  - `kitti_eval.format_comparison(results: dict[str, dict]) -> str` — markdown table keyed by `feature_mode`.
  - `eval_3d.main` CLI: `--3d-checkpoint --rfdetr-checkpoint [--device] [--split-dir kitti_imagesets] [--score-thr 0.05] [--image-dir ...] [--label-dir ...] [--calib-dir ...] [--limit N] [--output runs/monocular-3d/eval-<mode>.json]`; and `--compare a.json b.json ...` which only loads JSONs and prints `format_comparison`. JSON carries metadata: checkpoint path, `feature_mode`, split name, image count, and the note "held-out Chen val split".

- [ ] **Step 1 (Review Focus): write `tests.test_kitti_eval` cases** — `test_format_comparison_builds_markdown` (two tiny result dicts → a row per feature_mode) and `test_evaluate_nan_cell_for_absent_class` (a class with no valid GT → `nan` cell, not `0.0`).
- [ ] **Step 2 (Review Focus): write `tests.test_eval_3d`** — `test_assemble_gt_handles_label_free_frame`: a val stem whose label file is empty/DontCare-only yields zero valid GT (predictions there would be FPs), no exception. Keep torch/rfdetr-dependent paths behind `unittest.skipUnless(importlib.util.find_spec("rfdetr"), ...)`.
- [ ] **Step 3: Run them, expect failure.**
- [ ] **Step 4: Implement `format_table`, `format_comparison`,** and `eval_3d.main` (iterate val stems with `--limit`; for each: RF-DETR predict at `--score-thr`, keep checkpoint classes, build boxes/geometry as `infer_3d`, run head, decode, collect preds; read GT via `_labels`; call `evaluate`; print `format_table`, write JSON). `--compare` branch loads JSONs → `format_comparison`.
- [ ] **Step 5: Run `tests.test_kitti_eval` and `tests.test_eval_3d`, expect PASS.**
- [ ] **Step 6: Update `README.md`** with an "Evaluate 3D (KITTI AP)" section (single-checkpoint run and `--compare`), noting results are on the held-out Chen val split.
- [ ] **Step 7: Commit** (`feat: eval_3d.py KITTI 3D/BEV AP tables and ablation compare`).

---

## Notes for the executor

- Tasks 1–4 (split infra) and 5–9 (eval) are testable without a GPU. The end-to-end `eval_3d.py` run and the ablation table need the Chen-split retraining (RF-DETR + the three 3D modes), which happens on the remote GPU server after this code lands.
- Follow the repo's style: no new top-level deps beyond shapely; `unittest`; keep modules focused as `monocular_3d.py` / `train_3d.py` already are.
