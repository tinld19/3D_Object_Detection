# RF-DETR Foreground-Depth 3D Stage Design

## Intent

Replace the current crop-CNN 3D regressor with a second-stage 3D model that
uses the DINOv2 visual representation from a fine-tuned RF-DETR detector and
an end-to-end, MonoDETR-inspired foreground-depth branch.  The result should
predict KITTI 3D box attributes for each matched/proposed 2D object without a
dense depth data set or a separate full-scene depth foundation model.

## Goals

- Keep RF-DETR as the trained 2D detector and keep its current training CLI.
- Reuse its DINOv2 backbone representation as frozen RGB features for the new
  3D training stage.
- Learn a foreground depth-bin map from KITTI object `location.z` labels.
- Fuse RGB, foreground-depth and calibration/box features before predicting
  `h, w, l, x, y, z, sin(yaw), cos(yaw)`.
- Train on ground-truth boxes; run inference on RF-DETR boxes later.
- Preserve a CPU/MPS-capable import and test path; do not start training or
  download weights automatically.

## Non-goals

- Do not modify RF-DETR's detector architecture or its 2D fine-tuning flow.
- Do not add a full-frame metric-depth model, LiDAR, masks, dense depth labels,
  cross-attention, or an instance-segmentation stage.
- Do not rely on undocumented RF-DETR object-query tensors: its public API
  exposes detections, not feature/query extraction.

## Approaches considered

1. **Selected — frozen RF-DETR DINOv2 feature adapter plus a new 3D stage.**
   It preserves the user's selected two-stage pipeline and prevents 3D loss
   from destabilising the already fine-tuned 2D detector.  The adapter pins an
   RF-DETR Nano `rfdetr==1.11.2` and validates the extractor's tensor shape at load time,
   because this is not a documented public feature API.
2. **A separate vanilla DINOv2 checkpoint.** This has a stable API but does
   not reuse the RF-DETR/KITTI-adapted visual backbone the user requested.
3. **An end-to-end MonoDETR replacement.** It gives query-level depth
   cross-attention but replaces RF-DETR rather than extending it, so it was
   rejected for this project.

## Architecture

```text
full KITTI RGB ─→ frozen RF-DETR DINOv2 adapter ─→ RGB map (B,C,H/16,W/16)
                                             ├─→ two 3×3 convolutions
                                             │    → depth map logits
                                             │    → depth feature map
GT/predicted 2D box ─────────────────────────┼─→ RoIAlign RGB + depth maps
KITTI P2 intrinsics + normalized box ─────────┘
             pooled RGB + depth + geometry ─→ small MLP → 8 3D attributes
```

The depth map has 81 channels: 80 foreground metric-depth bins over 2–60 m
using MonoDETR's linear-increasing discretisation and one background bin.
For training, each object box fills its region with the bin for its ground
truth `location.z`; at overlaps, the nearer object wins.  The depth-loss mask
is full-frame at feature resolution, except objects outside 2–60 m are masked
out of depth CE while retaining their valid 3D regression target.  Valid
objects use the repository's established policy: every properly formed,
in-image-clipped KITTI label except `DontCare`, with its existing `rotation_y`
convention.  The 3D loss remains normalised MSE on
the existing eight targets.  The total loss is `3d_mse + depth_ce`; both terms
have a CLI weight, defaulting to 1.

Each 2D RoI is selected on both maps and adaptively averaged to a vector. A
single linear projection fuses their concatenation with the existing class,
normalised box and P2-intrinsic features. This is intentionally simpler than
MonoDETR's transformer cross-attention: it is a two-stage model with one 3D
candidate per object, not an end-to-end query detector.

## Data and interfaces

- `KittiForegroundDepthDataset` returns a full image, all valid objects for
  that image, P2, GT boxes, depth-bin targets, and existing 3D targets.
- A deterministic image-level 90/10 split prevents object crops from the same
  frame leaking into validation.
- `RFDETRDinoAdapter(checkpoint, device)` supports only a fine-tuned
  `RFDETRNano` checkpoint from `rfdetr==1.11.2` and is lazy-imported. It errors clearly
  if `rfdetr` is absent, the checkpoint is absent, or the pinned RF-DETR model
  no longer exposes a usable DINO feature map.
- `ForegroundDepth3DRegressor` accepts the adapter feature map plus image-level
  annotations, returns per-object 3D predictions and depth logits.
- `train_3d.py` is replaced with the new image-level training CLI and requires
  `--rfdetr-checkpoint`; the old crop model is removed rather than retained.
- The checkpoint stores the RF-DETR checkpoint path, class ordering, target
  normalisation statistics, depth bin limits/count, and 3D-stage weights.

## Files

- Replace `monocular_3d.py` with the dataset, depth-target function, model and
  adapter-facing interfaces.
- Replace `train_3d.py` with image-level two-stage training.
- Update `requirements.txt` to pin a compatible RF-DETR release.
- Update `README.md` with preparation, required RF-DETR checkpoint, training,
  and the boundary between detector and 3D-stage training.
- Replace/add focused tests for foreground-depth target assignment, depth-bin
  boundaries, image-level split, and feature/depth/3D output shapes using a
  fake feature adapter; no model download is required.

## Risks and validation

The RF-DETR package is currently absent locally, and its public docs do not
promise a feature-extraction API. The implementation therefore makes its
private-adapter dependency explicit and verifies it against the pinned package
before using it. If its model internals cannot yield a spatial DINO map, work
stops before model code is written; the user has approved pinning or vendoring
that RF-DETR version rather than switching to a standalone DINOv2 extractor.

Before completion, tests will confirm target and tensor invariants, Python
syntax/import validation will pass without optional weights, and a one-batch
CPU smoke test will run only if the user has installed RF-DETR and supplied a
checkpoint. Training metrics to compare are KITTI AP3D and validation 3D MSE;
depth CE is diagnostic only.
