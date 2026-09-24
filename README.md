# RF-DETR on KITTI camera data

This project fine-tunes [RF-DETR](https://github.com/roboflow/rf-detr) for **2D detection** on KITTI camera images. RF-DETR does not provide a native 3D-box training head. The generated COCO annotations retain KITTI `dimensions`, `location`, `rotation_y`, and camera calibration for a later 3D post-processing stage.

## Setup

Python 3.10+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the local data

The defaults match the existing folders under `data/` and write a deterministic 90/10 train/valid COCO split to `dataset/`.

```bash
python prepare_dataset.py
```

Use `--output-dir` to write elsewhere. Source data is never changed.

## Train

```bash
python train.py --epochs 50 --batch-size 2
```

On Apple Silicon, the script selects MPS when available; otherwise it uses CPU. Override it explicitly with `--device mps` or `--device cpu`. CPU training is supported but will be slow. Training output defaults to `runs/rfdetr-kitti/`.

## Train monocular 3D

The 3D model is intentionally separate from RF-DETR: it learns KITTI `h,w,l,x,y,z,rotation_y` from each object crop, its 2D box, class, and camera `P2` intrinsics. It is the supervised monocular 3D stage to run after the 2D detector.

```bash
python train_3d.py --epochs 30 --batch-size 8
```

The resulting `runs/monocular-3d/best.pt` stores weights, class names, and target-normalisation statistics. It uses MPS when supported, otherwise CPU. No 3D training is run automatically.

## Verify conversion

```bash
python3 -m unittest tests.test_prepare_dataset
```
