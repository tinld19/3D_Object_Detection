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
python train.py --epochs 50 --batch-size 8 --device cuda --num-workers 4
```

On Apple Silicon, the script selects MPS when available; otherwise it uses CPU. Override it explicitly with `--device mps` or `--device cpu`. CPU training is supported but will be slow. Training output defaults to `runs/rfdetr-kitti/`.

Early stopping monitors validation box mAP and stops after 15 epochs without improvement. The default `rich` progress bar displays training status; use `--progress-bar tqdm` or `--progress-bar none` to change it. RF-DETR Nano's default is 384×384. MonoDETR's KITTI setting is 1280×384, which RF-DETR cannot reproduce with its scalar `resolution` option; `--resolution 1280 --batch-size 1` is only a high-resolution square experiment, not a directly comparable MonoDETR baseline.

## Train monocular 3D

The 3D model is intentionally separate from RF-DETR. It freezes the 256-channel projected DINOv2 map from a fine-tuned RF-DETR Nano checkpoint, learns a MonoDETR-style foreground depth-bin map, and fuses the RGB/depth ROI features with the 2D box, class, and camera `P2` intrinsics to predict KITTI `h,w,l,x,y,z,rotation_y`.

```bash
python train_3d.py --rfdetr-checkpoint runs/rfdetr-kitti/checkpoint_best_total.pth --epochs 30
```

The RF-DETR checkpoint is required and is loaded only as a frozen feature extractor. Foreground depth uses 80 LID bins over 2–60 m plus a background bin; no full-scene pretrained depth estimator is used. The resulting `runs/monocular-3d/best.pt` stores the 3D head weights, class names, normalisation statistics, depth settings, and checkpoint path. It selects CUDA, then MPS, then CPU unless `--device` is specified. No 3D training is run automatically.

## Visualize 3D inference

Use a raw KITTI image and its matching calibration file. The script gets 2D proposals from RF-DETR, runs the trained 3D head, and projects each predicted 3D cuboid on the original image.

```bash
python infer_3d.py \
  --image data/kitti_raw/KITTI/training/image_2/000000.png \
  --calib data/kitti_raw/KITTI/training/calib/000000.txt \
  --rfdetr-checkpoint runs/rfdetr-kitti/checkpoint_best_ema.pth \
  --3d-checkpoint runs/monocular-3d/smoke.pt \
  --device cuda \
  --output runs/monocular-3d/inference-000000.png
```

Only detector classes represented in the 3D checkpoint are rendered. A first-epoch `smoke.pt` is only a pipeline check; train the 3D stage longer before interpreting cuboid quality.

## Verify conversion

```bash
python3 -m unittest tests.test_prepare_dataset
```
