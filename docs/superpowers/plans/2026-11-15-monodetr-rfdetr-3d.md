# Depth-aware Mono3D DETR on RF-DETR Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build approach B — a depth-aware monocular-3D DETR whose feature map and 2D proposals come from a frozen RF-DETR (DINOv2), with a new trainable encoder, a CNN depth predictor + soft-argmax depth formula, a depth-aware transformer decoder, split multibin heads, Hungarian set loss, and KITTI AP40 evaluation.

**Architecture:** Frozen RF-DETR (DINOv2 + head) gives a 256-ch feature map `F0` and 2D proposals (both cacheable). A new encoder lifts `F0` → `F`. A CNN depth predictor turns `F` into an 80-bin LID depth map and depth embeddings. Proposal-seeded queries pass through 3 decoder layers (self-attn → depth cross-attn → visual cross-attn); split heads predict class/box2d/center3d/depth(+σ)/size/orientation(multibin). Depth per query = ensemble `d_reg ⊕ d_geo ⊕ D(ref)` with Laplacian uncertainty. Trained with Hungarian matching on the Chen split; evaluated with the existing `kitti_eval.py` AP40 module.

**Tech Stack:** Python 3.14, torch 2.14 (CPU locally, CUDA on an RTX 4090), rfdetr[train]==1.11.2, scipy (Hungarian), shapely (IoU, in eval only), numpy, Pillow. Tests use `unittest` + synthetic tensors + injected loaders (no checkpoint, no GPU required).

**Spec:** `docs/superpowers/specs/2026-11-15-monodetr-rfdetr-3d-design.md` (read it; this plan argues from it).

## Global Constraints

- DINOv2 **and** RF-DETR's head stay **frozen**; the only trainable parameters are the new encoder, depth predictor, decoder, and heads. Any frozen module must have `requires_grad=False` and run under `no_grad` / `eval`.
- Reuse, do not duplicate: `depth_bin`, `foreground_depth_targets`, `encode_yaw`, `RFDETRDinoAdapter`, `device()`/`set_seed()` (in `train_3d.py`), the Chen split infra (`prepare_dataset._read_split`, `kitti_imagesets/`), and all of `kitti_eval.py`.
- Approach A stays working and untouched: `monocular_3d.py` (except additive, backward-compatible changes), `train_3d.py`, `infer_3d.py`, `train.py`, `prepare_dataset.py`.
- Depth: 80 LID bins, range 2–60 m. Orientation: multibin, **K=12** bins over observation angle `alpha ∈ [-π, π)`. Queries: **N=50**. Decoder: **3** layers. Classes: **Car, Pedestrian, Cyclist** (the classes `_labels` keeps).
- `P2` is rescaled by the actual per-axis resize factors; `d_geo = f_y · H_phys / h_box_px` uses rescaled `f_y` and input-pixel box height, with a divide-by-zero guard.
- New code lives in flat `mono3d_*.py` modules (repo convention) with one responsibility each; tests in `tests/test_mono3d_*.py` using `unittest`.
- **Test runner:** `pytest` is NOT installed; run tests with `unittest`. A module: `.venv/bin/python -m unittest tests.test_mono3d_targets -v`. Full suite: `.venv/bin/python -m unittest discover -s tests -q`. Use `.venv/bin/python` (the venv holds torch 2.14/scipy/shapely).
- Device helper mirrors `train_3d.device()` (CUDA → MPS → CPU), `--device` override. AMP + optional gradient accumulation; target batch 8 on a 24 GB card.
- B checkpoints carry `arch="b"`, `classes`, per-class mean dims, and target norms. Eval JSON carries `arch` and `feature_mode`.

## Review Focus

- **Image with zero valid GT objects.** Collate must emit empty target tensors; the set loss must produce only the no-object classification term and finite gradients. → test in Task 8 (loss) + Task 2 (collate).
- **Fewer RF-DETR proposals than N (incl. zero).** Query seeding must pad to N with learned queries and never index-error. → test in Task 5.
- **Object metric depth outside [2, 60] m.** `depth_bin` returns `None`; the foreground depth target must mask those pixels and the ensemble/`d_geo` must stay finite. → test in Task 1 + Task 4.
- **Degenerate 2D box (`h_box_px → 0`).** `d_geo` must not divide by zero; a guarded minimum height. → test in Task 1.
- **Non-square / aspect-changing resize.** `P2` rescale must apply independent x/y factors so `d_geo` and center back-projection stay correct; a wrong scale silently corrupts every `z`. → test in Task 1.

---

### Task 1: Geometry & target builders (`mono3d_targets.py`)

**Files:**
- Create: `mono3d_targets.py`
- Test: `tests/test_mono3d_targets.py`

**Interfaces:**
- Consumes: `depth_bin` from `monocular_3d`.
- Produces (all pure, numpy/math; no torch needed):
  - `bin_value(index: int, depth_min=2.0, depth_max=60.0, bins=80) -> float` — inverse LID: `depth_min + delta*index*(index+1)/2`.
  - `rescale_p2(p2: list[float], orig_size: tuple[int,int], in_size: tuple[int,int]) -> list[float]` — scale `f_x,c_x` by `W_in/W_orig`, `f_y,c_y` by `H_in/H_orig`; returns 12-value P2.
  - `geometric_depth(f_y: float, phys_height: float, box_height_px: float) -> float` — `f_y*phys_height/max(box_height_px, 1.0)`.
  - `ray_angle(x: float, z: float) -> float` = `atan2(x, z)`.
  - `alpha_from_ry(ry, x, z) -> float` and `ry_from_alpha(alpha, x, z) -> float`, both wrapped to `[-π, π)`.
  - `multibin_target(alpha: float, bins=12) -> tuple[int, float]` — `(bin_index, residual)` with residual in `[-π/bins, π/bins]`.
  - `multibin_decode(bin_index: int, residual: float, bins=12) -> float` — alpha in `[-π, π)`.
  - `project_center(x, y, z, p2) -> tuple[float,float]` — pixel of the 3D center via P2.
  - `backproject_center(u, v, z, p2) -> tuple[float,float]` — `(x, y)` from pixel + depth + P2.

- [ ] **Step 1: Write failing tests** (`tests/test_mono3d_targets.py`): `bin_value` round-trips `depth_bin` for indices 0,1,40,79; `bin_value` of a mid index matches a hand value; `rescale_p2` on `(1242,375)->(384,384)` scales `f_x` by `384/1242` and `f_y` by `384/375` independently; `geometric_depth` with `box_height_px=0` returns a finite number (guard); `alpha_from_ry`→`ry_from_alpha` round-trips 8 angles incl. ±π; `multibin_target`→`multibin_decode` round-trips alpha within one bin; `project_center`/`backproject_center` round-trip a known `(x,y,z)` under a sample P2; depth outside [2,60] → `depth_bin` returns None (asserts the guard the painter relies on).
- [ ] **Step 2: Run, verify they fail** (`.venv/bin/python -m unittest tests.test_mono3d_targets -v`).
- [ ] **Step 3: Implement** the helpers above in `mono3d_targets.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 2: Dataset, padded collate, proposal/feature cache (`mono3d_data.py`)

**Files:**
- Create: `mono3d_data.py`
- Test: `tests/test_mono3d_data.py`
- Modify (additive only): none required if helpers are imported.

**Interfaces:**
- Consumes: Task 1 helpers; `_labels`, `_image_size`, `_calibration`, `_read_split` from `prepare_dataset`; `foreground_depth_targets`, `depth_bin` from `monocular_3d`.
- Produces:
  - `class Mono3DDetrDataset(Dataset)` with `classes`, `class_ids`, per-class `mean_dims` (computed over the split), `__getitem__` returning a dict: `image` (CxHxW float), `p2` (rescaled 12), `orig_size`, `targets` = dict of tensors keyed `labels[g], boxes2d[g,4] (cx,cy,w,h normalized), center3d[g,2] (normalized pixel), depth[g], dims[g,3], alpha[g], orient_bin[g], orient_res[g]`, `depth_map_target[h,w]`, `depth_map_mask[h,w]`, and `flip` flag applied.
  - `collate_mono3d(batch) -> dict` — stacks images/p2; targets kept as a list of per-image dicts (variable `g`); depth maps padded/stacked.
  - `horizontal_flip(sample)` applied with probability `--flip-prob` in train mode (mirrors `boxes2d`, `center3d`, `alpha`/`ry`, and P2 principal point).
  - `class ProposalCache` — `get(stem) -> proposals | None`, `put(stem, proposals)`; proposals = dict `boxes2d[P,4], scores[P], classes[P]`; backed by `.npz` files under a cache dir. Feature cache (`F0`) same shape of API, optional.

- [ ] **Step 1: Write failing tests:** build a tiny synthetic KITTI sample on disk (1 image + label + calib via a temp dir, reuse the fixture style in `tests/test_monocular_3d.py`); assert `__getitem__` target shapes and that `dims`/`alpha`/`orient_bin` match Task 1 helpers for a known label; assert **empty-GT** image yields zero-length target tensors (Review Focus); assert `collate_mono3d` on a 2-image batch keeps a 2-element target list and stacks images `[2,3,H,W]`; assert `horizontal_flip` mirrors a known box center to `W-cx` and negates alpha per convention; assert `ProposalCache` round-trips a dict through a temp `.npz`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_data.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 3: Frozen feature extractor + trainable encoder (`mono3d_encoder.py`)

**Files:**
- Create: `mono3d_encoder.py`
- Test: `tests/test_mono3d_encoder.py`
- Modify (additive): `monocular_3d.py` only if `RFDETRDinoAdapter` needs a public method to also return proposals — add a separate `RFDETRProposalSource` rather than changing the adapter's contract.

**Interfaces:**
- Consumes: `RFDETRDinoAdapter` (frozen `F0`, 256-ch) from `monocular_3d`; the `loader=` injection pattern for tests.
- Produces:
  - `class RFDETRProposalSource(nn.Module)` — wraps a frozen RF-DETR; `forward(images) -> list[dict]` proposals `{boxes2d, scores, classes}` under `no_grad`; constructed with an injectable `loader`/`predict_fn` so tests pass a stub.
  - `class TrainableEncoder(nn.Module)` — `__init__(channels=256, layers=2, heads=8)`; `forward(f0: Tensor[B,256,h,w]) -> Tensor[B,256,h,w]` via flatten→self-attention(+2D pos emb)→reshape. All params trainable.

- [ ] **Step 1: Write failing tests:** `TrainableEncoder` forward on `[2,256,24,24]` returns same shape and **all params have grad** after a backward on the output sum; `RFDETRProposalSource` with a stub `predict_fn` returns the stub proposals and its params (if any) carry `requires_grad=False`; feeding a non-256-channel map raises.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_encoder.py` (and, if needed, the additive `RFDETRProposalSource` helper referencing the stable `RFDETRNano.predict` API).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 4: Depth predictor + soft-argmax formula (`mono3d_depth.py`)

**Files:**
- Create: `mono3d_depth.py`
- Test: `tests/test_mono3d_depth.py`

**Interfaces:**
- Consumes: `bin_value` (Task 1); `foreground_depth_targets` (reuse for supervision shape).
- Produces:
  - `class DepthPredictor(nn.Module)` — `__init__(channels=256, bins=80)`; `forward(f: Tensor[B,256,h,w]) -> (logits: Tensor[B,bins+1,h,w], depth_map: Tensor[B,h,w], depth_embed: Tensor[B,256,h,w])`.
  - `soft_argmax_depth(logits, bins=80, depth_min=2.0, depth_max=60.0) -> Tensor` — softmax over the **foreground** bins only (drop the background bin), weighted sum of `bin_value(i)`.
  - `sample_depth_at(depth_map: Tensor[B,h,w], ref_points: Tensor[B,N,2]) -> Tensor[B,N]` — bilinear sample of the depth map at normalized reference points.

- [ ] **Step 1: Write failing tests:** `soft_argmax_depth` on logits that are a hard one-hot at bin `i` (background bin set to −inf) returns `bin_value(i)` within tol, for i=0,10,79; background bin is excluded (a huge background logit does not change the result); `DepthPredictor` forward shapes on `[1,256,24,24]`; `sample_depth_at` at the center ref point returns the center depth value; depth outside range still yields a finite map value (Review Focus).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_depth.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 5: Query seeding + depth-aware decoder (`mono3d_decoder.py`)

**Files:**
- Create: `mono3d_decoder.py`
- Test: `tests/test_mono3d_decoder.py`

**Interfaces:**
- Consumes: Task 3 `F`, Task 4 `depth_embed`; proposals from Task 3.
- Produces:
  - `seed_queries(proposals, f, num_queries=50, jitter=False) -> (content: Tensor[B,N,256], ref_points: Tensor[B,N,2], valid_mask)` — ROI-pool `f` at each proposal box for content, box center for `ref_points`, pad with learned embeddings to N; `jitter=True` perturbs centers/sizes and randomly drops proposals (train only).
  - `class DepthAwareDecoderLayer(nn.Module)` — self-attn → depth cross-attn (to `depth_embed`) → visual cross-attn (to `F`), each toggled by `use_depth`/`use_visual`; refines `ref_points`.
  - `class DepthAwareDecoder(nn.Module)` — `__init__(layers=3, feature_mode="fusion")`; `forward(content, ref_points, f, depth_embed) -> (hidden_per_layer: list[Tensor[B,N,256]], ref_points_per_layer)`. `feature_mode` sets `use_depth`/`use_visual` per the design's ablation table.

- [ ] **Step 1: Write failing tests:** `seed_queries` with **3 proposals and N=50** returns content `[B,50,256]` and pads the remainder (Review Focus); with **0 proposals** still returns `[B,50,256]` (all learned) and never index-errors; `DepthAwareDecoder` forward returns 3 hidden states of `[1,50,256]`; `feature_mode="rgb"` runs with `depth_embed=None` (depth path off) and `feature_mode="depth"` runs without visual cross-attn; a backward populates decoder grads.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_decoder.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 6: Split heads + depth ensemble + decode (`mono3d_heads.py`)

**Files:**
- Create: `mono3d_heads.py`
- Test: `tests/test_mono3d_heads.py`

**Interfaces:**
- Consumes: Task 1 (`geometric_depth`, `backproject_center`, `multibin_decode`); Task 4 (`sample_depth_at`).
- Produces:
  - `class Mono3DHeads(nn.Module)` — `__init__(num_classes, bins=12, mean_dims)`; `forward(hidden: Tensor[B,N,256]) -> dict` with `logits[B,N,C], boxes2d[B,N,4], center3d[B,N,2], d_reg[B,N], d_sigma[B,N], dims[B,N,3], orient_bin[B,N,bins], orient_res[B,N,bins]`.
  - `depth_ensemble(d_reg, d_geo, d_map, d_sigma, mode) -> (depth, weight_info)` — combine the active sources (per ablation table; `rgb` mode drops `d_map`), uncertainty-aware; returns final depth per query.
  - `decode_boxes(outputs, ref_points, d_geo_inputs, depth_map, p2, mode) -> list[dict]` — per query `{score, class, box=(h,w,l,x,y,z,ry)}`, using `dims`=mean+residual, `z`=`depth_ensemble`, `(x,y)`=`backproject_center`, `ry`=`ry_from_alpha(multibin_decode(...))`.

- [ ] **Step 1: Write failing tests:** `Mono3DHeads` forward shapes on `[1,50,256]`; `depth_ensemble` in `rgb` mode uses only `d_reg,d_geo` (passing `d_map` must not change output); in `fusion` mode a change in `d_map` **does** change output; `decode_boxes` on hand-built outputs reproduces a known `(h,w,l,x,y,z,ry)` (uses Task 1 round-trip); score = sigmoid(max logit).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_heads.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 7: Assemble the model (`mono3d_model.py`)

**Files:**
- Create: `mono3d_model.py`
- Test: `tests/test_mono3d_model.py`

**Interfaces:**
- Consumes: Tasks 3–6.
- Produces:
  - `class MonoDetr3D(nn.Module)` — `__init__(num_classes, mean_dims, feature_mode="fusion", num_queries=50, decoder_layers=3)`; `forward(f0, proposals, p2, jitter=False) -> dict` with `aux` list (per decoder layer); each layer entry = `Mono3DHeads` outputs **plus** the ensembled `depth[B,N]` and `d_sigma[B,N]` (computed via Task 6 `depth_ensemble` from `d_reg`, `d_geo`, and the sampled `d_map`), so the loss and `decode_boxes` consume the same ensembled depth. Final entry also carries `depth_logits`/`depth_map`. Holds `TrainableEncoder`, `DepthPredictor` (skipped in `rgb` mode), `DepthAwareDecoder`, `Mono3DHeads`.
  - `predict(f0, proposals, p2, score_thr, mode) -> list[dict]` — calls `decode_boxes`, filters by `score_thr`, caps at `num_queries`, **no NMS**.

- [ ] **Step 1: Write failing tests:** `MonoDetr3D` forward on synthetic `f0=[1,256,24,24]` + stub proposals returns per-layer aux of length 3 and a final `depth_map`; all three `feature_mode`s run; `rgb` mode builds no `DepthPredictor` (asserts attribute is None); `predict` returns a list of decoded dicts with `score>=score_thr`; a backward over the loss-less output sum populates only trainable grads (encoder/decoder/heads), none on frozen inputs.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_model.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 8: Hungarian matching + set loss (`mono3d_loss.py`)

**Files:**
- Create: `mono3d_loss.py`
- Test: `tests/test_mono3d_loss.py`

**Interfaces:**
- Consumes: Task 7 outputs; Task 2 target dicts; `scipy.optimize.linear_sum_assignment`.
- Produces:
  - `hungarian_match(outputs_layer, targets) -> list[(pred_idx, gt_idx)]` — cost = focal(class)+L1(box2d)+GIoU(box2d)+center3d+depth; per image.
  - `class SetCriterion(nn.Module)` — `__init__(num_classes, weights)`; `forward(outputs, targets) -> dict[str,Tensor]` with keys `loss_class, loss_box, loss_giou, loss_center, loss_depth (Laplacian σ on the ensembled `depth`/`d_sigma` from Task 7 vs target `depth`), loss_dim, loss_orient (bin CE + residual), loss_depthmap (CE + L1), total`; sums auxiliary losses over all decoder layers; unmatched queries get no-object class loss only.

- [ ] **Step 1: Write failing tests:** `hungarian_match` on 2 GT and outputs whose queries 0,1 are near-perfect returns pairings `{0,1}`; `SetCriterion` on **empty targets** returns a finite `total` driven only by `loss_class` (Review Focus); perfect predictions give near-zero box/center/depth losses; `loss_depth` increases as predicted σ mismatches error (uncertainty behaves); aux losses add across 3 layers (total scales vs a 1-layer input).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `mono3d_loss.py`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 9: Training entry point (`train_mono3d_detr.py`)

**Files:**
- Create: `train_mono3d_detr.py`
- Test: `tests/test_train_mono3d_detr.py`

**Interfaces:**
- Consumes: Tasks 2,3,7,8; `device`/`set_seed` from `train_3d`; Chen split via `_read_split`.
- Produces:
  - `build_mean_dims(dataset) -> dict` and `split_indices` reuse (from `train_3d`).
  - `main()` with args `--rfdetr-checkpoint` (required), `--image-dir/--label-dir/--calib-dir`, `--split-dir` (default `kitti_imagesets`), `--feature-mode {fusion,rgb,depth}`, `--freeze-rfdetr` (always true; flag documents it), `--epochs`, `--batch-size 8`, `--grad-accum`, `--amp`, `--lr`, `--flip-prob`, `--cache-dir`, `--seed`, `--device`, `--output runs/mono3d-detr/<mode>.pt`.
  - `save_checkpoint(path, model, meta)` — writes `arch="b"`, `classes`, `mean_dims`, `feature_mode`, `target` norms.

- [ ] **Step 1: Write failing tests:** arg parsing defaults (split-dir, batch-size 8, output path per mode); `save_checkpoint` writes a dict with `arch=="b"` and `mean_dims`; `build_mean_dims` on a synthetic dataset returns per-class means; a **one-step** train on a 1-image synthetic dataset + a stubbed proposal source + tiny `f0` runs `loss.backward()`+`optimizer.step()` without error (integration smoke, CPU, `--limit`-style).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `train_mono3d_detr.py` (lazy-import torch/rfdetr inside `main`).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 10: Evaluation for arch B (`eval_3d.py`)

**Files:**
- Modify: `eval_3d.py`
- Test: `tests/test_eval_3d.py` (extend)

**Interfaces:**
- Consumes: Task 7 `MonoDetr3D.predict`; all of `kitti_eval.py` unchanged.
- Produces:
  - Decode dispatch in `evaluate_checkpoint` by checkpoint `arch`: `"a"` → existing path; `"b"` → run frozen RF-DETR proposals + `MonoDetr3D.predict` → pred dicts `{score,class,box}` (same shape the metric consumes).
  - JSON gains `arch`; `format_comparison` keys each column by `f"{arch}/{feature_mode}"` so A's and B's identical mode names don't collide.
  - `LIMITATIONS` updated for B (frozen DINOv2 + RF-DETR proposals, held-out split).

- [ ] **Step 1: Write failing tests:** a stub arch-`b` checkpoint + stub model/proposals through `evaluate_checkpoint` produces the metric-ready pred schema; `format_comparison` on one A JSON and one B JSON both `feature_mode="fusion"` yields two distinct columns `A/fusion`, `B/fusion`; existing arch-`a` tests still pass unchanged.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** the dispatch + compare keying in `eval_3d.py`.
- [ ] **Step 4: Run full suite** (`.venv/bin/python -m unittest discover -s tests -q`) — all prior + new tests green.
- [ ] **Step 5: Commit.**

---

## Server execution (after the branch is merged; not a coded task)

Per mode `m ∈ {fusion, rgb, depth}` on the RTX 4090, after `python prepare_dataset.py` and RF-DETR training on the Chen split:

```bash
python train_mono3d_detr.py --rfdetr-checkpoint <rfdetr.pth> \
  --feature-mode $m --amp --batch-size 8 --cache-dir runs/cache \
  --device cuda --output runs/mono3d-detr/$m.pt
python eval_3d.py --3d-checkpoint runs/mono3d-detr/$m.pt \
  --rfdetr-checkpoint <rfdetr.pth> --device cuda \
  --output runs/mono3d-detr/eval-$m.json
```
Then `python eval_3d.py --compare runs/mono3d-detr/eval-*.json [runs/monocular-3d/eval-*.json]` for the B ablation + optional A baseline table. Sanity gates before trusting AP: RF-DETR proposal 2D recall on val, and depth-map mean abs error on foreground.
