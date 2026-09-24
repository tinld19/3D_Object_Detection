# RF-DETR KITTI Detection Design

## Goal

Prepare the bundled KITTI training split for RF-DETR and train a monocular 3D regressor that runs on Apple Silicon MPS or CPU.

## Scope

- Convert the 7,481 KITTI training images, 2D boxes, 3D boxes, and camera calibration into RF-DETR's COCO dataset layout.
- Use a deterministic 90/10 train/valid split.
- Preserve KITTI 3D values (`dimensions`, `location`, `rotation_y`) as extra COCO annotation fields.
- Fine-tune `RFDETRNano` for 2D object detection with the official `rfdetr` package.
- Train a compact PyTorch monocular 3D regressor from KITTI crops, class labels, 2D boxes, and camera intrinsics.

## Non-goals

- RF-DETR cannot train a native 3D bounding-box head; 3D training is a separate model, not a modification to RF-DETR.
- No training run, model download, or data conversion is performed automatically.
- LiDAR input and prediction-time 3D reconstruction are deferred.

## Dataset

Source folders are:

- `data/data_object_image_2/training/image_2`
- `data/training/label_2`
- `data/data_object_calib/training/calib`

The converter ignores KITTI `DontCare` labels and invalid 2D boxes, and uses the remaining KITTI class names without remapping.

## Output and training

`dataset/` contains `train/` and `valid/`, with copied PNGs and `_annotations.coco.json` in each split. `train.py` invokes `RFDETRNano.train()` with `device="mps"` on Apple Silicon when available, otherwise `cpu`.

`train_3d.py` uses the original KITTI image, label, and calibration files. Its CNN receives the cropped object image plus the object's class ID, normalised 2D box, and `P2` intrinsics. It regresses normalised `h,w,l,x,y,z,sin(rotation_y),cos(rotation_y)` with MSE loss. The checkpoint stores target normalisation statistics and the class list for repeatable inference.

## Verification

A standard-library `unittest` fixture creates one synthetic KITTI frame and asserts that the converter emits a valid COCO annotation, preserves its 3D metadata, and drops `DontCare`. A second test verifies the yaw encoding so 3D targets do not have a discontinuity at ±π.
