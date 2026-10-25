"""Draw RF-DETR proposal-backed 3D cuboids from a trained 3D-stage checkpoint."""

import argparse
import math
from pathlib import Path
import sys

from monocular_3d import ForegroundDepth3DRegressor, RFDETRDinoAdapter
from prepare_dataset import _calibration


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def project_cuboid(dimensions, location, yaw, p2):
    """Project a KITTI bottom-centre cuboid through its P2 calibration."""
    height, width, length = dimensions
    x, y, z = location
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local = (
        (length / 2, 0, width / 2), (length / 2, 0, -width / 2),
        (-length / 2, 0, -width / 2), (-length / 2, 0, width / 2),
        (length / 2, -height, width / 2), (length / 2, -height, -width / 2),
        (-length / 2, -height, -width / 2), (-length / 2, -height, width / 2),
    )
    points = []
    for local_x, local_y, local_z in local:
        camera_x = cosine * local_x + sine * local_z + x
        camera_z = -sine * local_x + cosine * local_z + z
        projected_z = p2[8] * camera_x + p2[9] * (local_y + y) + p2[10] * camera_z + p2[11]
        if projected_z <= 0:
            raise ValueError("Cuboid lies behind the camera.")
        projected_x = p2[0] * camera_x + p2[1] * (local_y + y) + p2[2] * camera_z + p2[3]
        projected_y = p2[4] * camera_x + p2[5] * (local_y + y) + p2[6] * camera_z + p2[7]
        points.append((projected_x / projected_z, projected_y / projected_z))
    return points


def checkpoint_matches(saved_path, supplied_path):
    return Path(saved_path).resolve() == supplied_path.resolve()


def draw_cuboid(image, points, label):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for start, end in EDGES:
        draw.line((points[start], points[end]), fill="lime", width=2)
    left, top = min(x for x, _ in points), min(y for _, y in points)
    draw.text((left, max(0, top - 12)), label, fill="lime", stroke_width=1, stroke_fill="black")


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--rfdetr-checkpoint", type=Path, required=True)
    parser.add_argument("--3d-checkpoint", dest="three_d_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    args = parser.parse_args()
    for path in (args.image, args.calib, args.rfdetr_checkpoint, args.three_d_checkpoint):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
    p2 = _calibration(args.calib).get("P2")
    if len(p2 or []) != 12:
        raise SystemExit(f"Calibration has no 12-value P2 matrix: {args.calib}")
    try:
        import torch
        from rfdetr import RFDETRNano
        from PIL import Image
    except ImportError as error:
        raise SystemExit('Install dependencies first: pip install -r requirements.txt') from error

    selected_device = args.device or device()
    state = torch.load(args.three_d_checkpoint, map_location=selected_device)
    saved_rfdetr = state.get("rfdetr_checkpoint")
    if saved_rfdetr and not checkpoint_matches(saved_rfdetr, args.rfdetr_checkpoint):
        print(
            f"Warning: 3D stage was trained with RF-DETR checkpoint {saved_rfdetr}, "
            f"but inference uses {args.rfdetr_checkpoint}.",
            file=sys.stderr,
        )
    classes = state["classes"]
    class_ids = {name: index for index, name in enumerate(classes)}
    three_d = ForegroundDepth3DRegressor(len(classes)).to(selected_device)
    three_d.load_state_dict(state["state_dict"])
    three_d.eval()
    mean, std = torch.as_tensor(state["target_mean"], device=selected_device), torch.as_tensor(state["target_std"], device=selected_device)
    detector = RFDETRNano.from_checkpoint(args.rfdetr_checkpoint, device=selected_device)
    adapter = RFDETRDinoAdapter(args.rfdetr_checkpoint, loader=lambda _path: detector).to(selected_device)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    image_width, image_height = image.size
    detections = detector.predict(image, threshold=args.confidence, include_source_image=False)
    matched = [
        (box, float(score), str(name))
        for box, score, name in zip(detections.xyxy, detections.confidence, detections.data["class_name"])
        if str(name) in class_ids
    ]
    if matched:
        resized = image.resize((384, 384))
        pixels = torch.tensor(bytearray(resized.tobytes()), dtype=torch.float32).view(384, 384, 3)
        boxes = torch.tensor(
            [[left / image_width, top / image_height, (right - left) / image_width, (bottom - top) / image_height]
             for (left, top, right, bottom), _, _ in matched], dtype=torch.float32, device=selected_device,
        )
        geometry = torch.cat((boxes, torch.tensor(
            [p2[0] / image_width, p2[5] / image_height, p2[2] / image_width, p2[6] / image_height],
            dtype=torch.float32, device=selected_device,
        ).expand(len(matched), -1)), dim=1)
        class_tensor = torch.tensor([class_ids[name] for _, _, name in matched], dtype=torch.long, device=selected_device)
        with torch.no_grad():
            features = adapter(pixels.permute(2, 0, 1).div_(255).unsqueeze(0).to(selected_device))
            predictions, _ = three_d(features, [boxes], [class_tensor], [geometry])
        for (_, confidence, name), prediction in zip(matched, predictions[0]):
            height, width, length, x, y, z, sine, cosine = (prediction * std + mean).tolist()
            try:
                points = project_cuboid((height, width, length), (x, y, z), math.atan2(sine, cosine), p2)
            except ValueError:
                continue
            draw_cuboid(image, points, f"{name} {confidence:.2f} z={z:.1f}m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.output} with {len(matched)} matched 3D proposals.")


if __name__ == "__main__":
    main()
