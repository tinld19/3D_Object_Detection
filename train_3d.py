"""Train the frozen RF-DETR DINO + foreground-depth KITTI 3D stage."""

import argparse
import random
from pathlib import Path

from monocular_3d import (
    ForegroundDepth3DRegressor,
    KittiForegroundDepthDataset,
    RFDETRDinoAdapter,
    foreground_depth_targets,
)


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _one(batch):
    return batch[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/data_object_image_2/training/image_2"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/training/label_2"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/data_object_calib/training/calib"))
    parser.add_argument("--rfdetr-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/monocular-3d/best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    args = parser.parse_args()
    try:
        import torch
        from torch.nn import functional as functional
        from torch.utils.data import DataLoader, Subset
    except ImportError as error:
        raise SystemExit('Install dependencies first: pip install -r requirements.txt') from error

    dataset = KittiForegroundDepthDataset(args.image_dir, args.label_dir, args.calib_dir)
    if len(dataset) < 2:
        raise SystemExit("At least two labelled KITTI frames are required for the train/validation split.")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    valid_count = max(1, round(len(indices) * 0.1))
    train_indices, valid_indices = indices[valid_count:], indices[:valid_count]
    targets = dataset.raw_targets(train_indices)
    dataset.set_target_normalizer(targets.mean(0), targets.std(0, correction=0))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=_one)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=_one)
    selected_device = args.device or device()
    adapter = RFDETRDinoAdapter(args.rfdetr_checkpoint).to(selected_device)
    model = ForegroundDepth3DRegressor(len(dataset.classes)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def loss_for(sample):
        features = adapter(sample["image"].unsqueeze(0).to(selected_device))
        prediction, depth_logits = model(
            features,
            [sample["boxes"].to(selected_device)],
            [sample["class_ids"].to(selected_device)],
            [sample["geometry"].to(selected_device)],
        )
        depth_target, loss_mask = foreground_depth_targets(sample["objects"], sample["image_size"], depth_logits.shape[-2:])
        depth_target[~loss_mask] = -100
        depth_loss = functional.cross_entropy(depth_logits, depth_target.unsqueeze(0).to(selected_device), ignore_index=-100)
        return functional.mse_loss(prediction[0], sample["targets"].to(selected_device)) + args.depth_loss_weight * depth_loss

    for epoch in range(1, args.epochs + 1):
        model.train()
        for sample in train_loader:
            optimizer.zero_grad()
            loss_for(sample).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [loss_for(sample).item() for sample in valid_loader]
        valid_loss = sum(losses) / len(losses)
        print(f"epoch={epoch} valid_loss={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": dataset.classes,
                    "target_mean": dataset.target_mean,
                    "target_std": dataset.target_std,
                    "rfdetr_checkpoint": str(args.rfdetr_checkpoint),
                    "depth": {"bins": 80, "min": 2.0, "max": 60.0},
                },
                args.output,
            )


if __name__ == "__main__":
    main()
