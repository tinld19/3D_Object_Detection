"""Train the approach-B Mono3D DETR on frozen RF-DETR features and proposals."""

import argparse
from pathlib import Path

from prepare_dataset import _read_split
from train_3d import device, set_seed, split_indices


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=Path("data/data_object_image_2/training/image_2"))
    p.add_argument("--label-dir", type=Path, default=Path("data/training/label_2"))
    p.add_argument("--calib-dir", type=Path, default=Path("data/data_object_calib/training/calib"))
    p.add_argument("--split-dir", type=Path, default=Path("kitti_imagesets"))
    p.add_argument("--rfdetr-checkpoint", type=Path, required=True)
    p.add_argument("--freeze-rfdetr", action="store_true", default=True, help="RF-DETR is always frozen.")
    p.add_argument("--feature-mode", choices=("fusion", "rgb", "depth"), default="fusion")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flip-prob", type=float, default=0.0)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.output is None:
        args.output = Path("runs/mono3d-detr") / f"{args.feature_mode}.pt"
    args.image_size, args.num_queries, args.decoder_layers = 384, 50, 3
    return args


def save_checkpoint(path, model, meta: dict) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**meta, "arch": "b", "state_dict": model.state_dict()}, path)


def _step(model, criterion, batch, feature_fn, proposal_fn, dev):
    import torch

    images = batch["image"].to(dev)
    with torch.no_grad():
        f0 = feature_fn(images)
        proposals = proposal_fn(images)
    outputs = model(f0, proposals, batch["p2"].tolist(), tuple(images.shape[-2:]))
    targets = [{k: v.to(dev) for k, v in t.items()} for t in batch["targets"]]
    return criterion(outputs, targets, depth_map_target=batch["depth_map_target"].to(dev),
                     depth_map_mask=batch["depth_map_mask"].to(dev))


def train_epoch(model, criterion, loader, optimizer, feature_fn, proposal_fn, dev,
                grad_accum=1, scaler=None, amp=False):
    """One training epoch; returns the mean total loss."""
    import torch

    model.train()
    use_amp = amp and str(dev).startswith("cuda")
    total, count = 0.0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss = _step(model, criterion, batch, feature_fn, proposal_fn, dev)["total"]
        scaled = loss / grad_accum
        (scaler.scale(scaled) if scaler else scaled).backward()
        if (i + 1) % grad_accum == 0 or i + 1 == len(loader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def evaluate(model, criterion, loader, feature_fn, proposal_fn, dev):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            losses.append(_step(model, criterion, batch, feature_fn, proposal_fn, dev)["total"].item())
    return sum(losses) / max(len(losses), 1)


def run_training(args, feature_fn, proposal_fn):
    import torch
    from torch.utils.data import DataLoader, Subset

    from mono3d_data import Mono3DDetrDataset, collate_mono3d
    from mono3d_loss import SetCriterion
    from mono3d_model import MonoDetr3D

    set_seed(args.seed)
    dev = args.device or device()
    dataset = Mono3DDetrDataset(args.image_dir, args.label_dir, args.calib_dir, (1, 1),
                                image_size=args.image_size, flip_prob=0.0)
    # Discover the feature grid from one forward so depth-map targets align with the model.
    with torch.no_grad():
        f0 = feature_fn(dataset[0]["image"].unsqueeze(0).to(dev))
    dataset.feature_size = tuple(f0.shape[-2:])
    dataset.flip_prob = args.flip_prob
    stems = [s["stem"] for s in dataset.samples]
    train_idx, valid_idx = split_indices(stems, set(_read_split(args.split_dir / "val.txt")))
    if not train_idx:
        raise SystemExit("ImageSets split leaves an empty training set.")
    mean_dims = torch.tensor([dataset.mean_dims[c] for c in dataset.classes], dtype=torch.float32)
    model = MonoDetr3D(len(dataset.classes), mean_dims, feature_mode=args.feature_mode,
                       num_queries=args.num_queries, decoder_layers=args.decoder_layers).to(dev)
    criterion = SetCriterion(len(dataset.classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp = args.amp and str(dev).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if amp else None
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_mono3d)
    valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=args.batch_size, collate_fn=collate_mono3d)
    meta = {"classes": dataset.classes, "class_ids": dataset.class_ids, "mean_dims": dataset.mean_dims,
            "feature_mode": args.feature_mode, "feature_size": tuple(dataset.feature_size),
            "image_size": args.image_size, "num_queries": args.num_queries,
            "decoder_layers": args.decoder_layers, "rfdetr_checkpoint": str(args.rfdetr_checkpoint)}
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, feature_fn, proposal_fn,
                                 dev, args.grad_accum, scaler, amp)
        valid_loss = evaluate(model, criterion, valid_loader, feature_fn, proposal_fn, dev) if valid_idx else train_loss
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            save_checkpoint(args.output, model, meta)
    return best


def main() -> None:
    args = parse_args()
    try:
        import torch  # noqa: F401
        from mono3d_encoder import RFDETRProposalSource
        from monocular_3d import RFDETRDinoAdapter
    except ImportError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
    dev = args.device or device()
    adapter = RFDETRDinoAdapter(args.rfdetr_checkpoint).to(dev).eval()
    source = RFDETRProposalSource(args.rfdetr_checkpoint)
    run_training(args, adapter, source)


if __name__ == "__main__":
    main()
