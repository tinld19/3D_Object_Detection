# RF-DETR Foreground-Depth 3D Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crop-based KITTI 3D regressor with a frozen RF-DETR DINOv2 feature adapter, MonoDETR-style foreground-depth supervision, and a fused per-object 3D head.

**Architecture:** RF-DETR Nano remains the independently trained 2D detector. The image-level 3D stage reads its frozen, projected DINOv2 map, learns a depth-bin feature/map from it, pools RGB and depth features within each ground-truth 2D box, and regresses the existing eight 3D values. A thin, version-pinned adapter owns the non-public RF-DETR integration; all model tests use a fake feature tensor and do not download a model.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, `rfdetr==1.11.2`, Pillow, standard-library `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-rfdetr-foreground-depth-design.md`

## Global Constraints

- Preserve source data under `data/`; do not start data conversion or model training automatically.
- Keep RF-DETR as a separately fine-tuned 2D detector; the 3D stage only supports `RFDETRNano` from `rfdetr==1.11.2`.
- The DINO adapter must be frozen and fail clearly for a missing package, checkpoint, or incompatible feature map.
- Only valid existing KITTI labels except `DontCare` are 3D objects; retain their existing clipped boxes and yaw encoding.
- Use 80 LID foreground depth bins for 2–60 m plus one background bin; labels outside this range retain 3D regression but are ignored by depth CE.
- Use ground-truth boxes for 3D training; proposal-based 3D inference is not added in this change.
- Preserve unrelated user changes already present in `README.md` and `train.py`.

## Review Focus

- A depth target with overlapping boxes assigns the nearer valid object; Task 1 adds an overlap test.
- An object below 2 m or above 60 m produces an ignored depth region but a valid 3D target; Task 1 adds range-boundary tests.
- A box clipped to a one-pixel-wide image edge creates a nonempty feature-level foreground target; Task 1 adds a clipping test.
- A batch with different object counts retains one 3D output per object without padding labels into loss; Task 2 adds a two-image/object-count test.
- RF-DETR's private backbone layout changes or a checkpoint is missing; Task 3 adds adapter validation tests with fake contexts and clear error assertions.

---

### Task 1: Image-level KITTI records and foreground-depth targets

**Files:**
- Modify: `monocular_3d.py`
- Modify: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: existing `_calibration`, `_image_size`, `_labels`, and KITTI labels.
- Produces: `depth_bin(depth: float, depth_min: float = 2.0, depth_max: float = 60.0, bins: int = 80) -> int | None`, `foreground_depth_targets(objects: list[dict], image_size: tuple[int, int], feature_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]`, and `KittiForegroundDepthDataset` yielding an image-level record.

- [ ] **Step 1: Write failing target tests**

```python
def test_foreground_depth_target_uses_nearer_overlap_and_background():
    target, loss_mask = foreground_depth_targets([near, far], (100, 100), (10, 10))
    self.assertEqual(depth_bin(5.0), target[3, 3].item())
    self.assertEqual(80, target[0, 0].item())

def test_out_of_range_depth_is_ignored():
    _, loss_mask = foreground_depth_targets([too_near], (100, 100), (10, 10))
    self.assertIsNone(depth_bin(1.9))
    self.assertIsNone(depth_bin(60.1))
    self.assertFalse(loss_mask[2, 2].item())
```

- [ ] **Step 2: Run the target tests to verify they fail**

Run: `python3 -m unittest tests.test_prepare_dataset.ConvertDatasetTest.test_foreground_depth_target_uses_nearer_overlap_and_background tests.test_prepare_dataset.ConvertDatasetTest.test_out_of_range_depth_is_ignored`

Expected: FAIL because the foreground-depth functions do not exist.

- [ ] **Step 3: Implement the target functions and replace crop samples with image records in `monocular_3d.py`**

`foreground_depth_targets` must map the existing clipped `xywh` boxes to feature pixels, initialise all pixels to background bin `80` with a true loss mask, paint only in-range depths, and write objects in descending `z` order so lower `z` overwrites overlaps. An out-of-range box writes a false loss-mask region unless overwritten by a nearer valid object. `KittiForegroundDepthDataset.__getitem__(index: int) -> dict` must resize full RGB frames to the RF-DETR Nano 384-square input, retain normalised boxes and normalised P2 geometry, and return per-object class IDs, 3D targets, and depth values; use an image-level record rather than one crop per object.

- [ ] **Step 4: Add image-level split and clipping tests**

```python
def test_dataset_returns_all_frame_objects_and_clipped_depth_target():
    sample = dataset[0]
    self.assertEqual(2, sample["class_ids"].numel())
    target, _ = foreground_depth_targets(sample["objects"], sample["image_size"], (8, 12))
    self.assertTrue((target != 80).any())
```

- [ ] **Step 5: Run Task 1 tests**

Run: `python3 -m unittest tests.test_prepare_dataset -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add monocular_3d.py tests/test_prepare_dataset.py
git commit -m "feat: add KITTI foreground depth targets"
```

### Task 2: Depth branch, feature fusion, and 3D prediction head

**Files:**
- Modify: `monocular_3d.py`
- Create: `tests/test_monocular_3d.py`

**Interfaces:**
- Consumes: a frozen RGB feature map with shape `(B, 256, H, W)`, object boxes normalised to `[0, 1]`, class IDs, P2/box geometry, depth-target maps, and `KittiForegroundDepthDataset` records from Task 1.
- Produces: `ForegroundDepth3DRegressor(num_classes: int, feature_channels: int = 256, depth_bins: int = 80)` and `forward(rgb_features: torch.Tensor, boxes: list[torch.Tensor], class_ids: list[torch.Tensor], geometry: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]`, returning one `(objects_i, 8)` prediction tensor per image and `(B, 81, H, W)` depth logits.

- [ ] **Step 1: Write failing shape and gradient tests with synthetic feature maps**

```python
def test_model_returns_one_3d_prediction_per_object_and_depth_logits():
    predictions, logits = model(features, [boxes0, boxes1], [ids0, ids1], [geometry0, geometry1])
    self.assertEqual([(2, 8), (1, 8)], [tuple(item.shape) for item in predictions])
    self.assertEqual((2, 81, 8, 12), tuple(logits.shape))
```

- [ ] **Step 2: Run the model test to verify it fails**

Run: `python3 -m unittest tests.test_monocular_3d.ForegroundDepthModelTest.test_model_returns_one_3d_prediction_per_object_and_depth_logits -v`

Expected: FAIL because `ForegroundDepth3DRegressor` does not exist.

- [ ] **Step 3: Implement `ForegroundDepth3DRegressor` in `monocular_3d.py`**

Project 256-channel RGB input to a 256-channel depth feature with two `3×3` convolutions and ReLU, then predict 81 depth logits with `1×1` convolution. For each normalised 2D box, crop RGB and depth maps with nonempty integer bounds and adaptive-average-pool each crop; concatenate the two pooled vectors, class embedding, and existing eight geometry values, then use the smallest two-layer MLP that outputs eight values. Do not add transformer/cross-attention modules.

- [ ] **Step 4: Add a loss test for varying object counts**

```python
def test_combined_loss_backpropagates_for_two_frames_with_different_object_counts():
    loss = three_d_mse + cross_entropy(depth_logits, depth_target, ignore_index=-100)
    loss.backward()
    self.assertIsNotNone(model.depth_predictor[0].weight.grad)
```

- [ ] **Step 5: Run Task 2 tests**

Run: `python3 -m unittest tests.test_monocular_3d -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add monocular_3d.py tests/test_monocular_3d.py
git commit -m "feat: fuse foreground depth into 3D head"
```

### Task 3: Pinned RF-DETR DINO adapter and 3D training CLI

**Files:**
- Modify: `monocular_3d.py`
- Modify: `train_3d.py`
- Modify: `requirements.txt`
- Modify: `README.md`
- Create: `tests/test_rfdetr_adapter.py`

**Interfaces:**
- Consumes: `rfdetr==1.11.2`, a fine-tuned RF-DETR Nano checkpoint path, image tensors, and Task 1/Task 2 interfaces.
- Produces: `RFDETRDinoAdapter(checkpoint: Path, device: str)`, `forward(images: torch.Tensor) -> torch.Tensor`, and a `train_3d.py --rfdetr-checkpoint PATH` image-level training entry point.

- [ ] **Step 1: Write failing adapter validation tests with fake RF-DETR modules**

```python
def test_adapter_rejects_missing_checkpoint():
    with self.assertRaisesRegex(FileNotFoundError, "RF-DETR checkpoint"):
        RFDETRDinoAdapter(Path("missing.pt"), "cpu")

def test_adapter_rejects_non_spatial_backbone_output():
    with self.assertRaisesRegex(RuntimeError, "spatial DINO feature map"):
        adapter(torch.zeros(1, 3, 384, 384))
```

- [ ] **Step 2: Run the adapter tests to verify they fail**

Run: `python3 -m unittest tests.test_rfdetr_adapter -v`

Expected: FAIL because `RFDETRDinoAdapter` does not exist.

- [ ] **Step 3: Implement `RFDETRDinoAdapter` in `monocular_3d.py` against the pinned package**

Lazy-import RF-DETR, load only a Nano checkpoint, retrieve its DINOv2 backbone and existing 256-channel RF-DETR projector by the pinned package's verified module path, freeze every parameter and evaluate it. Convert their output into one `(B, 256, H, W)` feature map and reject anything else. Keep adapter loading code isolated so package-internal changes have one failure point.

- [ ] **Step 4: Replace crop training with image-level training in `train_3d.py`**

Add required `--rfdetr-checkpoint`, `--depth-loss-weight`, and existing device/training options. Build the adapter once, keep it in eval/no-grad mode, use a batch-size-one identity collate function, build Task 1 targets after obtaining each adapter feature map's spatial size, calculate normalised 3D MSE across each frame's object tensors plus cross entropy against `target.masked_fill(~loss_mask, -100)` with `ignore_index=-100`, and save adapter metadata, target normalisation, class names, depth-bin settings, and 3D-stage state dict.

- [ ] **Step 5: Pin/document the dependency and interface**

Set `rfdetr[train]==1.11.2` in `requirements.txt`. Update README to distinguish 2D RF-DETR training from 3D-stage training, state the required fine-tuned Nano checkpoint, document foreground depth's 2–60 m scope, and state that no full-scene pretrained depth estimator is used. Preserve the user's unrelated README edits.

- [ ] **Step 6: Run the unit suite and syntax checks**

Run: `python3 -m unittest discover -v && python3 -m py_compile prepare_dataset.py monocular_3d.py train.py train_3d.py`

Expected: PASS without RF-DETR installed because adapter imports remain lazy and tests use fakes.

- [ ] **Step 7: Run the installed-package compatibility smoke check when a checkpoint is supplied**

Run: `python3 train_3d.py --rfdetr-checkpoint PATH --epochs 1 --batch-size 1 --device cpu`

Expected: the adapter verifies a spatial feature map, starts one epoch, and writes a checkpoint; do not run this command automatically.

- [ ] **Step 8: Commit Task 3**

```bash
git add monocular_3d.py train_3d.py requirements.txt README.md tests/test_rfdetr_adapter.py
git commit -m "feat: train RF-DETR foreground-depth 3D stage"
```

### Task 4: Final regression and integration review

**Files:**
- Modify: `README.md` only if the verified CLI differs from Task 3 documentation.

**Interfaces:**
- Consumes: all Task 1–3 public functions and command line arguments.
- Produces: a verified codebase and documented manual smoke-test command.

- [ ] **Step 1: Run all deterministic checks on the final worktree**

Run: `python3 -m unittest discover -v && python3 -m py_compile prepare_dataset.py monocular_3d.py train.py train_3d.py`

Expected: PASS.

- [ ] **Step 2: Inspect the diff for source-data writes and accidental RF-DETR detector changes**

Run: `git diff --check && git diff -- prepare_dataset.py train.py`

Expected: no whitespace errors and no unintended source-data or 2D-detector changes.

- [ ] **Step 3: Commit documentation correction only if verification changed it**

```bash
git add README.md
git commit -m "docs: clarify foreground depth 3D training"
```
