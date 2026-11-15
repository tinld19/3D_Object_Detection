# Depth-aware Mono3D DETR on RF-DETR Features — Design

Date: 2026-11-15
Status: Draft (awaiting approval)

## Intent & why

The current 3D stage (`ForegroundDepth3DRegressor`, approach "A") is a
two-stage, per-box regressor: RF-DETR emits a 2D box, a frozen DINOv2 feature
map is ROI-pooled, and a single flat MLP regresses `(h,w,l,x,y,z,sin,cos)`.
Two structural weaknesses cap its accuracy and make it uninteresting as a
thesis contribution:

1. **Depth is guessed, not computed.** `z` is regressed directly by the flat
   MLP. The foreground LID depth map (`depth_logits`) is only an auxiliary
   supervised branch; it never feeds the final `z`.
2. **No attention, no decoder, no set prediction.** Each box is processed in
   isolation; there is no cross-object reasoning and no end-to-end learning of
   the features for the 3D task (the backbone and its proposals are frozen).

On the strict KITTI 3D AP@0.7 metric this design is expected to land around
M3D-RPN / MonoFlex territory (Car Moderate ≈ 9–15%), **below** MonoDETR
(≈ 20). It cannot realistically beat the published baseline.

This project builds approach **B**: a **depth-aware monocular-3D DETR** whose
feature extractor and 2D proposals come from **RF-DETR (DINOv2 backbone)**,
trained end-to-end (DINOv2 **and** RF-DETR frozen; a new encoder plus the
depth-aware 3D head trained on the frozen feature map, so features and
proposals are cacheable). It replaces the flat regressor with a MonoDETR-style
depth-aware transformer decoder and split prediction heads, and derives depth
from a formula over a learned foreground depth map rather than guessing it.

**Thesis contribution (novelty).** MonoDETR uses a ResNet backbone and a plain
DETR encoder with fully-learned object queries. Approach B instead (a) uses a
**foundation-model feature extractor (DINOv2 via RF-DETR)** for monocular 3D
set-prediction, and (b) initializes decoder queries from **RF-DETR 2D
proposals** (proposal-guided queries) rather than learning them blind. This
combination has no exact published KITTI baseline and is the defensible novel
claim; it also gives faster convergence than vanilla MonoDETR. RF-DETR stays a
first-class component (feature extractor + query seeding), not a discarded
pre-stage.

## Goals

- A new end-to-end model, trained with Hungarian set-prediction loss, that
  takes a single RGB image and outputs 3D detections `(class, score, h, w, l,
  x, y, z, ry)` with a confidence usable by the KITTI AP metric.
- **Depth by formula, not direct regression.** A CNN depth predictor produces
  a foreground depth map over 80 LID bins (2–60 m); per-query depth is an
  **ensemble** of (i) direct regressed depth, (ii) geometric depth from the
  predicted physical height and 2D box height, and (iii) the depth-map value
  sampled at the query's reference point — combined with a Laplacian
  **uncertainty** `σ`.
- **Depth-aware decoder** with per-layer self-attention, depth cross-attention
  (queries ↔ depth embeddings), and visual cross-attention (queries ↔ RF-DETR
  features).
- **Split heads** per query: class, 2D box (for matching + the 3D-center
  projection), 3D-center offset, depth (+uncertainty), size `(h,w,l)`,
  orientation (**multibin**, 12 bins).
- **Proposal-guided queries** seeded from RF-DETR 2D detections (reference
  point = box center, initial content = ROI feature), padded to a fixed N=50.
- **Multi-class** training (Car / Pedestrian / Cyclist); Car is the headline
  comparison, all three go in the ablation/tracking tables.
- Reuse the existing **Chen 3DOP split** (3712 train / 3769 val) and the
  existing **KITTI AP40 metric module** (`kitti_eval.py`) unchanged, so numbers
  remain directly comparable to MonoDETR.
- An **ablation** within B: fusion (both cross-attns) vs rgb-only (visual
  cross-attn, depth branch off) vs depth-only (depth cross-attn, visual off),
  reusing the same `--compare` table story as before.

## Non-goals / anti-goals

- **Not** fine-tuning the DINOv2 weights themselves. DINOv2 stays frozen, as in
  RF-DETR's own design; "fine-tune" means the RF-DETR projector/encoder and the
  new 3D head are trainable.
- **Not** the KITTI online test server. We evaluate on the held-out Chen val
  split; output metadata says so (reusing the existing `LIMITATIONS` pattern).
- **Not** a new metric. `kitti_eval.py` (rotated IoU, AP40, difficulty,
  devkit matching) is frozen; only how predictions are produced changes.
- **Not** retaining approach A as the primary model. A's code may remain as a
  **baseline row** for the thesis table, but B is the deliverable. The
  `--compare` tooling should still ingest A's JSON for that baseline row.
- **Anti-goal:** reporting B's AP as architecturally apples-to-apples with
  MonoDETR without disclosure. The report must state: DINOv2 frozen, RF-DETR
  pretrained+fine-tuned, proposal-guided queries. The number is comparable on
  the metric axis; the setup differs and must be named.
- **Anti-goal:** silently breaking the Chen-split comparability or the AP40
  protocol.

## Constraints & environment

- Python 3.10+, torch, Pillow, `rfdetr[train]==1.11.2`, shapely ≥ 2.0, numpy,
  scipy. numba must not be required.
- Device helper mirrors existing `device()` (CUDA → MPS → CPU), `--device`
  override. Training is expected on the remote GPU server; the Mac is for
  code + CPU smoke tests (`--limit`).
- Data layout matches `train_3d.py` defaults (image_2 / label_2 / calib).
  Split lists come from `kitti_imagesets/{train,val}.txt`.
- Workflow: code written/tested on Mac (M4), data in Cloudflare R2, pulled +
  trained on the GPU server. The design must be runnable headless on the server
  with a single train command per ablation mode.
- Reuse existing pieces rather than reimplement: `depth_bin`,
  `foreground_depth_targets`, `encode_yaw` (orientation target helper),
  `RFDETRDinoAdapter` (extended to expose trainable encoder output), the Chen
  split infra in `prepare_dataset.py` / `train_3d.py`, and all of
  `kitti_eval.py`.

## The how — decided

### Feature extraction (RF-DETR frozen; our own trainable encoder) — option (a)
- `RFDETRDinoAdapter` returns the frozen 256-ch DINOv2 feature map `F0` (as
  today, no grad). RF-DETR's own detection head stays frozen and is run via its
  stable public predict API to produce **2D proposals** (boxes, scores,
  classes) for query seeding.
- A **new trainable encoder** we own (a few self-attention / lightweight
  deformable layers) sits on top of `F0` → encoded feature map `F`. This is the
  end-to-end "fine-tune" surface; DINOv2 and RF-DETR stay frozen, so both `F0`
  and the proposals are **cacheable to disk** — on the RTX 4090 the trainable
  graph is small and feature caching removes the frozen backbone from every
  step. No surgery into rfdetr internals beyond the public API.

### Depth predictor (new CNN + formula)
- CNN on `F` → logits over `depth_bins+1` (80 LID bins + background), i.e. the
  existing foreground-depth-map target shape from `foreground_depth_targets`.
- **LID bin→metric** value `bin_value(i)` is the inverse of `depth_bin`'s LID
  mapping (closed form); the **depth map** is `D(u,v) = Σ_i softmax(logits)_i ·
  bin_value(i)` (soft-argmax). This is the "công thức" — no direct map
  guessing.
- Produces **depth embeddings** (a learned projection of the depth features +
  depth positional encoding) that the decoder attends to.
- Supervised by `foreground_depth_targets` (cross-entropy on bins + a map
  regression term), reusing the existing target builder.

### Proposal-guided queries (new, the RF-DETR element)
- From RF-DETR proposals (top-N by score, N=50): reference point = normalized
  2D box center; initial query content embedding = ROI-pooled `F` at that box
  (reuse the existing `_pool` idea) + class embedding of the proposal.
- Fewer than N proposals → pad with learned query embeddings; this keeps a
  fixed query count and lets the decoder recover missed proposals.

### Depth-aware decoder (new, 3 layers)
Each layer, in order:
1. **Self-attention** among the N queries.
2. **Depth cross-attention**: queries attend to the depth embeddings.
3. **Visual cross-attention**: queries attend to `F` (at/around their
   reference points).
Reference points are refined per layer (iterative, DETR-style). Ablation modes
toggle layers 2 (depth) and 3 (visual).

### Split prediction heads (new, per query, per decoder layer for aux loss)
- **class**: `num_classes` logits (focal loss). Score = sigmoid(class).
- **box2d**: 4-d normalized box (matching cost + GIoU loss + the anchor for the
  3D-center projection).
- **center3d**: 2-d offset giving the projected 3D-center pixel; with depth and
  `P2` it back-projects to `(x, y)`.
- **depth**: `(d_reg, σ)`; final depth = learned/weighted ensemble of
  `d_reg`, `d_geo = f_y · H_phys / h_box_px` (from predicted height + 2D box
  height), and `D(reference)`; depth loss is Laplacian-uncertainty on the
  ensemble.
- **size**: `(h, w, l)` residual over per-class mean dims.
- **orientation**: multibin — `K=12` bin logits + per-bin residual; decoded to
  `ry` via the observation-angle→`ry` convention already used by the dataset.

### Matching & loss (Hungarian set prediction)
- Cost: focal(class) + L1(box2d) + GIoU(box2d) + 3D-center + depth terms
  (MonoDETR-style weights as the starting point, tunable).
- Loss = the matched costs above + size + multibin orientation + the
  foreground-depth-map loss; auxiliary losses on every decoder layer.
- Unmatched queries → "no object" class loss only.

### Training entry point
- New `train_mono3d_detr.py` (sibling of `train_3d.py`): args
  `--rfdetr-checkpoint` (required), `--split-dir` (default `kitti_imagesets`),
  `--feature-mode {fusion,rgb,depth}`, `--freeze-encoder/--fine-tune-encoder`,
  `--epochs`, `--lr`, `--device`, `--output`. One command per ablation mode.
- The model, dataset, decoder, heads, and loss live in focused modules (the
  plan fixes exact file boundaries); `monocular_3d.py`'s reusable dataset/
  target helpers are shared, not duplicated.

### Inference & evaluation
- New decode path: run RF-DETR (proposals) → model → per-query
  `(class, score, h,w,l,x,y,z,ry)`; keep queries above `--score-thr`.
- **`eval_3d.py` is reworked** to call this decode instead of the A-style
  "RF-DETR box → 3D head" path. `kitti_eval.py` (AP40 / IoU / matching /
  difficulty / `format_table` / `format_comparison`) is untouched. The
  `--compare` flow still produces the ablation + baseline (A) table.

## What's left to the builder

- Exact module/file layout and the exact decoder/attention tensor shapes
  (follow the plan's Interfaces; keep files focused).
- Exact loss and matching-cost weights (start from MonoDETR; tune on the
  server). The ensemble-weighting mechanism for the three depth sources
  (learned scalars vs uncertainty-weighted) is the builder's call within the
  "ensemble + Laplacian σ" contract.
- Multibin decode details (bin centers, residual range) consistent with the
  dataset's existing yaw convention.
- Whether RF-DETR proposals are cached to disk for the server run or computed
  on the fly (perf, not correctness).
- The precise closed form for `bin_value(i)` (inverse of `depth_bin`), pinned
  with a unit test against `depth_bin` round-trip.

## Open workflow note (not architecture)

The in-progress branch `feat/kitti-3d-eval` (A-style eval tooling, incl.
`kitti_eval.py` which B depends on) is complete but **not yet merged**. Before
B's implementation starts, that branch should be integrated (merge/PR) so
`kitti_eval.py` is on the base B builds from. This is raised at plan/execution
setup, not decided here.

## Builder-check clarifications (resolved — my calls)

A fresh reviewer read this doc against the existing code. The items below were
either answered by the project/conversation or settled by me; they are now
part of the contract. The three items needing the author are in the next
section.

### Input resolution & intrinsics (was ambiguous)
- The existing dataset squashes 1242×375 → 384×384 anisotropically, which the
  current A-path tolerates but B's `d_geo` and depth map need pixel-honest
  intrinsics. **Call:** `P2` is rescaled by the **actual per-axis resize
  factors** (`f_x,c_x` by `W_in/W_orig`, `f_y,c_y` by `H_in/H_orig`); `d_geo =
  f_y · H_phys / h_box_px` uses the rescaled `f_y` and the box height in input
  pixels. Prefer **aspect-preserving resize + pad** over squash **iff** the
  RF-DETR Nano checkpoint accepts it; otherwise keep the square input the
  checkpoint was trained at and apply the anisotropic `P2` scaling honestly.
  The exact input resolution is whatever the RF-DETR checkpoint requires and is
  pinned in the plan with a test. **Known risk:** a coarse 384² feature map
  (~24×24) limits depth-map resolution and therefore AP; recorded, not a
  blocker.

### Training data pipeline & targets (new code needed)
- `encode_yaw` (sin/cos) does **not** cover multibin; new target builders are
  in scope: observation angle `alpha`, 3D-center projection target, and
  `K=12` multibin (bin-class + residual) targets.
- `train_3d.py`'s `batch_size=1` collate is A-only and stays untouched. B gets
  a **new padded collate** (variable GT per image) and real batching; if the
  server VRAM forces it, **gradient accumulation + AMP** instead of large
  batches (sizing fixed once the GPU budget is known — see next section).
- **Augmentation:** horizontal flip in scope (standard for MonoDETR, material
  for AP), with correct `P2` (principal-point mirror) and `alpha`/`ry` mirror.
  Photometric jitter optional, builder's call.
- **Classes:** positive training/eval classes are **Car, Pedestrian,
  Cyclist** (the classes `_labels` keeps after `IGNORED_CLASSES`). Van / Truck
  / Misc / Tram / Person_sitting / DontCare are **not** positive classes; they
  follow the existing KITTI neighbour / don't-care rules already encoded in
  `kitti_eval.py` and are never matched as a positive during training.

### RF-DETR class alignment (answered)
- RF-DETR is fine-tuned via `prepare_dataset.py` on the **same** KITTI class set
  (Car/Ped/Cyc), so proposal classes align with B's classes. The seed query
  embeds the proposal's class, but **the decoder re-predicts class**, so a
  mismatched or missing proposal class is tolerated.

### Proposal seeding & the train/val gap (new concern, resolved)
- RF-DETR is trained on the Chen **train** split, so proposals on training
  images are unrealistically good. **Call:** apply **proposal jitter** (noise
  on box center/size, random drop, and a few random/learned pad queries) during
  training — DETR-denoising in spirit — so the decoder does not overfit to
  perfect seeds. Reference points are refined **relative to the seed** per
  decoder layer. Proposals do **not** receive gradient (RF-DETR used for
  proposals is frozen; see author question on trainability).

### Ablation modes — exact active components
| | visual cross-attn | depth cross-attn | depth predictor + map loss | depth ensemble |
|---|---|---|---|---|
| **fusion** | on | on | on | `d_reg ⊕ d_geo ⊕ D(ref)` |
| **rgb** | on | off | off | `d_reg ⊕ d_geo` |
| **depth** | off | on | on | `d_reg ⊕ d_geo ⊕ D(ref)` |
- Query seeding from RF-DETR ROI features of `F` is **not** an ablated axis; it
  stays on in all three modes. Class/box2d/center/size/orientation heads stay
  active in all modes; only the depth-map head + depth cross-attn toggle.

### Depth-map supervision (detail)
- Cross-entropy over all `depth_bins+1` logits (background included) from
  `foreground_depth_targets`; **soft-argmax over foreground bins only**
  (background excluded) for the metric `D(u,v)`; an L1 term on `D` vs the
  painted bin-center metric depth. CE/L1 weights tunable.

### Eval & checkpoint contract (A stays working)
- `train_3d.py` and `infer_3d.py` (approach A) stay functional and untouched.
- B checkpoints carry an `arch: "b"` tag and the per-class mean dims / target
  norms they need; `eval_3d.py` dispatches decode by that tag (A checkpoints
  keep decoding as today).
- DETR-style decode: **no NMS**, cap at N=50 detections/image, `--score-thr`
  default 0.05 so AP integrates the full PR curve; out-of-image 3D centers are
  decoded as-is and left to IoU.
- Each eval JSON records both `arch` and `feature_mode`; `--compare` keys rows
  by a combined label (e.g. `B/fusion`, `A/fusion`) so A's and B's identical
  mode names do **not** collide, letting the A baseline sit beside B's modes in
  one table.

### Staged validation gates (plan will enforce)
- Before full 3D training is judged: (1) RF-DETR proposal 2D recall on val,
  (2) depth-map error (mean abs metric error on foreground) — cheap sanity
  gates so a broken feature/depth path is caught before blaming the 3D AP.

## Author decisions (resolved 2026-11-15)

1. **GPU budget:** single **RTX 4090 (24 GB)**. Plan targets this: feature map
   `F0` + proposals **cached to disk** (frozen, so valid), **AMP** on, start at
   **batch 8** with **gradient accumulation** if a mode's memory needs it; the
   trainable graph (new encoder + decoder + heads) is small enough to fit.
   Epoch/wall-clock sizing is the plan's to confirm empirically against this
   card.
2. **RF-DETR trainability:** **option (a)** — DINOv2 and RF-DETR's head stay
   frozen; a new trainable encoder on top of `F0` is the fine-tune surface.
   Proposals and `F0` are cacheable. (Reflected in "Feature extraction" above.)
3. **Branch:** **merge `feat/kitti-3d-eval` into `main` first**, then build B
   on a fresh branch off `main`. The merge is the plan's setup task.
