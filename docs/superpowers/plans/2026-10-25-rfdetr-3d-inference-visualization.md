# RF-DETR 3D Inference Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI that writes a KITTI image with RF-DETR proposal-backed, projected 3D cuboids from the trained 3D-stage model.

**Architecture:** `infer_3d.py` loads RF-DETR Nano once and shares it with the frozen DINO adapter. It matches detector proposals to the 3D checkpoint's classes by name, predicts and denormalizes 3D parameters, then projects KITTI bottom-centre cuboids with `P2` onto the original image.

**Tech Stack:** Python, PyTorch, Pillow, `rfdetr==1.11.2`, standard-library `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-25-rfdetr-3d-inference-visualization-design.md`

## Global Constraints

- Require both an RF-DETR Nano checkpoint and a trained 3D-stage checkpoint.
- Use the detector's public `predict()` API for 2D proposals; do not modify detector weights or training.
- Use the existing `RFDETRDinoAdapter` and `ForegroundDepth3DRegressor` with a 384-square adapter input.
- Render cuboids on the original image using its matching KITTI `P2` calibration.
- Skip detections whose class name is not represented in the 3D checkpoint and boxes with non-positive projected depth.
- Add no dependencies and perform no Git write operations, per user instruction.

## Review Focus

- KITTI location is bottom-centre: projection test covers vertical corners at `y=0` and `y=-h`.
- Yaw orientation: projection test covers a nonzero yaw and asserts finite pixel coordinates.
- Bad calibration: CLI rejects a missing or non-12-value `P2` before inference.
- Unknown detector class: inference skips it instead of indexing a wrong class embedding.
- Detector returns no boxes: inference still writes an unchanged output image.

---

### Task 1: Cuboid geometry and render helpers

**Files:**
- Create: `infer_3d.py`
- Create: `tests/test_infer_3d.py`

**Interfaces:**
- Produces: `project_cuboid(dimensions, location, yaw, p2) -> list[tuple[float, float]]` and `draw_cuboid(image, points, label) -> None`.

- [ ] **Step 1: Write failing cuboid projection tests**

```python
def test_project_cuboid_returns_eight_finite_points_for_kitti_bottom_centre_box():
    points = project_cuboid([2, 2, 4], [0, 2, 10], 0.25, P2)
    self.assertEqual(8, len(points))
    self.assertTrue(all(math.isfinite(value) for point in points for value in point))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_infer_3d -v`
Expected: FAIL because `infer_3d` does not exist.

- [ ] **Step 3: Implement geometry and rendering helpers in `infer_3d.py`**

Build KITTI bottom-centre corners using `h,w,l`, rotate around camera Y by yaw, project homogeneous corners through a 3×4 `P2`, and draw the 12 fixed cuboid edges with Pillow.

- [ ] **Step 4: Run the projection tests**

Run: `.venv/bin/python -m unittest tests.test_infer_3d -v`
Expected: PASS.

### Task 2: Single-image 3D inference CLI

**Files:**
- Modify: `infer_3d.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `--image PATH`, `--calib PATH`, `--rfdetr-checkpoint PATH`, `--3d-checkpoint PATH`, `--output PATH`, optional `--confidence` and `--device`.
- Produces: a PNG/JPEG at `--output` with projected cuboids.

- [ ] **Step 1: Load checkpoints and prepare per-proposal 3D inputs**

Load RF-DETR Nano from its checkpoint, pass that instance to `RFDETRDinoAdapter` so the backbone is not loaded twice, and use `detector.predict(image, threshold=confidence)`. Convert original-pixel `xyxy` boxes to normalized `xywh`; combine each box with normalized `P2` intrinsics; map `detections.data["class_name"]` through the saved 3D `classes` list.

- [ ] **Step 2: Predict, denormalize, and render**

Load the saved 3D state dict and normalizer statistics, run the 3D head once for all mapped proposals, recover yaw via `atan2(sin_yaw, cos_yaw)`, and draw each valid cuboid plus class, 2D confidence, and predicted depth label on the original RGB image.

- [ ] **Step 3: Add input validation**

Fail clearly for missing checkpoint/image/calibration files and a missing or malformed `P2`; create `--output` parent directories. If no matched proposals remain, save the unchanged image successfully.

- [ ] **Step 4: Document a server smoke command**

Document `infer_3d.py` in README with the trained `runs/monocular-3d/smoke.pt`, matching raw KITTI image and calibration file, and RF-DETR EMA checkpoint.

- [ ] **Step 5: Run deterministic checks**

Run: `.venv/bin/python -m unittest discover -v && .venv/bin/python -m py_compile infer_3d.py`
Expected: PASS.

- [ ] **Step 6: Run the server smoke command manually when both checkpoints are available**

Run: `python infer_3d.py --image IMAGE.png --calib IMAGE.txt --rfdetr-checkpoint CHECKPOINT.pth --3d-checkpoint runs/monocular-3d/smoke.pt --device cuda --output runs/monocular-3d/inference.png`
Expected: an output image with zero or more projected cuboids and no model-loading error.
