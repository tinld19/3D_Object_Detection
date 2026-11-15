"""Frozen RF-DETR proposal source and trainable feature encoder for Mono3D DETR."""

import torch
from torch import nn


def _to_tensor(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype)
    return torch.as_tensor(value, dtype=dtype)


class RFDETRProposalSource(nn.Module):
    """Runs a frozen RF-DETR and returns normalized-cxcywh 2D proposals.

    ``predict_fn(image)`` takes one image (Tensor[3,H,W], values in [0,1]) and
    returns an object/dict exposing ``xyxy`` [P,4] pixels, ``confidence`` [P]
    and ``class_id`` [P] (attribute or key access, as supervision.Detections).
    """

    def __init__(self, checkpoint, predict_fn=None):
        super().__init__()
        self.checkpoint = checkpoint
        self._predict_fn = predict_fn
        for p in self.parameters():
            p.requires_grad_(False)

    def _get_predict(self):
        if self._predict_fn is None:
            from rfdetr import RFDETRNano  # lazy: only needed in production

            model = RFDETRNano.from_checkpoint(str(self.checkpoint))
            self._predict_fn = lambda image: model.predict(image)
        return self._predict_fn

    @staticmethod
    def _field(det, name):
        return det[name] if isinstance(det, dict) else getattr(det, name)

    @torch.no_grad()
    def forward(self, images):
        predict = self._get_predict()
        height, width = images.shape[-2:]
        out = []
        for image in images:
            det = predict(image)
            xyxy = _to_tensor(self._field(det, "xyxy"), torch.float32).reshape(-1, 4)
            x1, y1, x2, y2 = xyxy.unbind(-1)
            boxes = torch.stack(
                [
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
                dim=-1,
            ).clamp(0.0, 1.0)
            out.append(
                {
                    "boxes2d": boxes,
                    "scores": _to_tensor(self._field(det, "confidence"), torch.float32).reshape(-1),
                    "classes": _to_tensor(self._field(det, "class_id"), torch.long).reshape(-1),
                }
            )
        return out


def _sincos_2d(h, w, dim, device, dtype):
    """2D sinusoidal positional embedding, shape [h*w, dim]."""
    if dim % 4 != 0:
        raise ValueError("channels must be divisible by 4 for 2D sin/cos embedding")
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega
    ey = torch.cat([ys.sin(), ys.cos()], dim=1)  # [h, dim/2]
    ex = torch.cat([xs.sin(), xs.cos()], dim=1)  # [w, dim/2]
    pos = torch.cat(
        [ey[:, None, :].expand(h, w, -1), ex[None, :, :].expand(h, w, -1)], dim=-1
    )
    return pos.reshape(h * w, dim).to(dtype)


class TrainableEncoder(nn.Module):
    """Transformer encoder lifting frozen F0 [B,C,h,w] into trainable F."""

    def __init__(self, channels=256, layers=2, heads=8):
        super().__init__()
        self.channels = channels
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, f0):
        if f0.dim() != 4 or f0.shape[1] != self.channels:
            raise ValueError(
                f"TrainableEncoder expects [B,{self.channels},h,w], got {tuple(f0.shape)}"
            )
        b, c, h, w = f0.shape
        seq = f0.flatten(2).transpose(1, 2)
        seq = seq + _sincos_2d(h, w, c, f0.device, f0.dtype)[None]
        seq = self.norm(self.encoder(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)
