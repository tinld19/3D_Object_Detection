# RF-DETR KITTI Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, reproducible RF-DETR fine-tuning scaffold for the local KITTI detection data.

**Architecture:** A single converter creates RF-DETR-compatible COCO train/valid splits while retaining KITTI 3D fields as extra annotation metadata. A separate CLI selects MPS or CPU and calls the upstream `RFDETRNano` training API.

**Tech Stack:** Python 3.10+, standard library, `rfdetr[train]`.

**Spec:** `docs/superpowers/specs/2026-09-07-rf-detr-kitti-design.md`

## Global Constraints

- Do not modify source data under `data/`.
- Do not run model training.
- Preserve `dimensions`, `location`, and `rotation_y` per accepted KITTI annotation.
- Support MPS and CPU without CUDA requirements.

---

### Task 1: KITTI to COCO conversion

**Files:**
- Create: `prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`

**Interfaces:**
- Produces: `convert_dataset(image_dir: Path, label_dir: Path, calib_dir: Path, output_dir: Path, val_fraction: float, seed: int) -> None`

- [ ] Write a failing standard-library test with one image, one `Car`, and one `DontCare` label.
- [ ] Run `python -m unittest tests.test_prepare_dataset` and confirm the import fails.
- [ ] Implement parsing, deterministic splitting, image copying, COCO JSON output, and 3D metadata preservation.
- [ ] Run `python -m unittest tests.test_prepare_dataset` and confirm it passes.

### Task 2: Training entry point and usage

**Files:**
- Create: `train.py`
- Create: `requirements.txt`
- Create: `README.md`

**Interfaces:**
- Consumes: a COCO directory produced by `prepare_dataset.py`.
- Produces: RF-DETR checkpoints under the user-selected output directory.

- [ ] Add a CLI that chooses MPS where available, otherwise CPU, and forwards dataset and training options to `RFDETRNano.train`.
- [ ] Pin the required package floor and document installation, conversion, training, constraints, and MPS behavior.
- [ ] Run Python syntax checks for all project scripts.

### Task 3: Monocular 3D regressor

**Files:**
- Create: `monocular_3d.py`
- Create: `train_3d.py`
- Modify: `tests/test_prepare_dataset.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `KittiMonocular3DDataset`, `Monocular3DRegressor`, and a checkpoint containing model weights, class names, and target normalisation statistics.

- [ ] Write a failing test asserting yaw is encoded as sine and cosine.
- [ ] Implement a compact crop CNN with class/bbox/intrinsics features and a KITTI dataset reader.
- [ ] Implement training with deterministic split, MSE optimisation, MPS/CPU device selection, and a self-contained checkpoint.
- [ ] Run standard-library tests and Python syntax checks.
