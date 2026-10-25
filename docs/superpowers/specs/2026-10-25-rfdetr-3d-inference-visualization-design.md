# RF-DETR 3D Inference Visualization Design

## Intent

Run the trained RF-DETR Nano detector and the separate foreground-depth 3D
head on one KITTI image, then write that image with projected 3D cuboids.
This is a qualitative inference check, not KITTI evaluation.

## Approaches

1. **Selected — project cuboids from predicted metric boxes.** RF-DETR
   supplies 2D proposals; the 3D head predicts dimensions, camera location,
   and yaw; the image's `P2` matrix projects the eight cuboid corners. This
   visibly validates the complete 2D-to-3D path.
2. **Draw 2D boxes with 3D text.** This is simpler but cannot reveal wrong
   yaw, dimensions, or camera-space location, so it was rejected.

## Scope

Add one `infer_3d.py` CLI. It receives an image, matching KITTI calibration
file, RF-DETR Nano checkpoint, 3D-stage checkpoint, confidence threshold, and
output path. It loads the detector once, reuses it in the DINO adapter,
matches proposals to 3D classes by class name, and skips detector classes the
3D checkpoint did not train on.

The script resizes the image to 384 square only for the DINO adapter. RF-DETR
boxes remain in original-image pixels; their normalized coordinates and `P2`
intrinsics feed the 3D head. Predicted normalized targets are denormalized,
`sin/cos` becomes yaw through `atan2`, and KITTI's bottom-centre box convention
is used to make eight corners. Corners are rotated around the camera Y axis,
projected by `P2`, and drawn as 12 cuboid edges on the original image.

It uses only PyTorch, Pillow, and existing RF-DETR dependencies. It does not
train, evaluate AP, modify detector weights, render boxes with non-positive
camera depth, or add a general video/batch API.

## Validation

Unit-test cuboid projection with an identity-like `P2` and a known box,
ensuring eight finite image points and twelve valid edges. Syntax-check the
new CLI. A real server smoke command writes a PNG once both checkpoints and a
KITTI image/calibration pair are available.
