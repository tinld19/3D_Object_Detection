"""Evaluate a trained 3D-stage checkpoint with KITTI 3D/BEV AP40 on the Chen val split."""

import argparse
import json
import math
from pathlib import Path

import kitti_eval
from prepare_dataset import _calibration, _labels, _read_split

EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")
IOU_THRESHOLDS = {"Car": [0.7, 0.5], "Pedestrian": [0.5], "Cyclist": [0.5]}
NOTE = "held-out Chen val split, not the KITTI test server"
SPLIT_NAME = "Chen val"


LIMITATIONS = [
    "DontCare regions are not suppressed via 2D-IoU overlap; low-score detections in DontCare regions count as false positives (conservatively lowers AP).",
    "Matching is single-pass score-greedy, not the devkit's per-threshold overlap-greedy matching (usually <0.5 AP difference).",
]


def assemble_gt(label_path, width, height):
    """Ground truths for one frame; label-free or DontCare-only frames give []."""
    gts = []
    for label in _labels(Path(label_path), width, height):
        h, w, l = label["dimensions"]
        x, y, z = label["location"]
        gts.append({
            "box": (h, w, l, x, y, z, label["rotation_y"]),
            "class": label["class_name"],
            "difficulty": kitti_eval.object_difficulty(
                label["bbox_height_raw"], label["occlusion"], label["truncation"]),
        })
    return gts


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def compare(paths):
    results = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        results[data["feature_mode"]] = {
            cls: {m: {float(t): d for t, d in thr.items()} for m, thr in metrics.items()}
            for cls, metrics in data["result"].items()}
    print(kitti_eval.format_comparison(results))


def evaluate_checkpoint(args):
    try:
        import torch
        from PIL import Image
        from rfdetr import RFDETRNano
    except ImportError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
    from monocular_3d import ForegroundDepth3DRegressor, RFDETRDinoAdapter

    dev = args.device or _device()
    state = torch.load(args.three_d_checkpoint, map_location=dev)
    classes = state["classes"]
    class_ids = {name: i for i, name in enumerate(classes)}
    feature_mode = state.get("feature_mode", "fusion")
    three_d = ForegroundDepth3DRegressor(len(classes), feature_mode=feature_mode).to(dev)
    three_d.load_state_dict(state["state_dict"])
    three_d.eval()
    mean = torch.as_tensor(state["target_mean"], device=dev)
    std = torch.as_tensor(state["target_std"], device=dev)
    detector = RFDETRNano.from_checkpoint(args.rfdetr_checkpoint, device=dev)
    adapter = RFDETRDinoAdapter(args.rfdetr_checkpoint, loader=lambda _p: detector).to(dev)

    stems = _read_split(args.split_dir / "val.txt")
    if args.limit:
        stems = stems[:args.limit]
    images = []
    for stem in stems:
        p2 = _calibration(args.calib_dir / f"{stem}.txt").get("P2")
        if len(p2 or []) != 12:
            raise SystemExit(f"Calibration has no 12-value P2 matrix: {stem}")
        with Image.open(args.image_dir / f"{stem}.png") as source:
            image = source.convert("RGB")
        width, height = image.size
        gts = assemble_gt(args.label_dir / f"{stem}.txt", width, height)
        detections = detector.predict(image, threshold=args.score_thr, include_source_image=False)
        matched = [
            (box, float(score), str(name))
            for box, score, name in zip(detections.xyxy, detections.confidence, detections.data["class_name"])
            if str(name) in class_ids
        ]
        preds = []
        if matched:
            resized = image.resize((384, 384))
            pixels = torch.tensor(bytearray(resized.tobytes()), dtype=torch.float32).view(384, 384, 3)
            boxes = torch.tensor(
                [[l / width, t / height, (r - l) / width, (b - t) / height]
                 for (l, t, r, b), _, _ in matched], dtype=torch.float32, device=dev)
            geometry = torch.cat((boxes, torch.tensor(
                [p2[0] / width, p2[5] / height, p2[2] / width, p2[6] / height],
                dtype=torch.float32, device=dev).expand(len(matched), -1)), dim=1)
            class_tensor = torch.tensor([class_ids[n] for _, _, n in matched], dtype=torch.long, device=dev)
            with torch.no_grad():
                features = adapter(pixels.permute(2, 0, 1).div_(255).unsqueeze(0).to(dev))
                predictions, _ = three_d(features, [boxes], [class_tensor], [geometry])
            for (_, score, name), prediction in zip(matched, predictions[0]):
                h, w, l, x, y, z, sine, cosine = (prediction * std + mean).tolist()
                preds.append({"score": score, "class": name,
                              "box": (h, w, l, x, y, z, math.atan2(sine, cosine))})
        images.append({"preds": preds, "gts": gts})

    eval_classes = [c for c in classes if c in EVAL_CLASSES]
    thresholds = {c: IOU_THRESHOLDS[c] for c in eval_classes}
    result = kitti_eval.evaluate(images, eval_classes, thresholds, ["3d", "bev"])
    print(kitti_eval.format_table(result, feature_mode))
    output = args.output or Path(f"runs/monocular-3d/eval-{feature_mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "3d_checkpoint": str(args.three_d_checkpoint),
        "rfdetr_checkpoint": str(args.rfdetr_checkpoint),
        "feature_mode": feature_mode,
        "split": SPLIT_NAME,
        "num_images": len(images),
        "note": NOTE,
        "limitations": LIMITATIONS,
        "result": result,
    }, indent=2))
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--3d-checkpoint", dest="three_d_checkpoint", type=Path)
    parser.add_argument("--rfdetr-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--split-dir", type=Path, default=Path("kitti_imagesets"))
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--image-dir", type=Path, default=Path("data/data_object_image_2/training/image_2"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/training/label_2"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/data_object_calib/training/calib"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.compare:
        return compare(args.compare)
    if not args.three_d_checkpoint or not args.rfdetr_checkpoint:
        parser.error("--3d-checkpoint and --rfdetr-checkpoint are required unless --compare is used")
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
