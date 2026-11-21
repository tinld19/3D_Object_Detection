"""Depth predictor and soft-argmax LID depth formula for the Mono3D DETR."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mono3d_targets import bin_value


def _bin_values(bins, depth_min, depth_max, like):
    return torch.tensor(
        [bin_value(i, depth_min, depth_max, bins) for i in range(bins)],
        dtype=like.dtype, device=like.device,
    )


def soft_argmax_depth(logits, bins=80, depth_min=2.0, depth_max=60.0):
    """Expected depth over the foreground bins; the last (background) channel is dropped.

    logits: [..., bins+1, h, w] -> [..., h, w]. Always a convex combination of
    bin values, hence finite and within [depth_min, bin_value(bins-1)].
    """
    fg = logits.narrow(-3, 0, bins)
    probs = fg.softmax(dim=-3)
    values = _bin_values(bins, depth_min, depth_max, probs).view(bins, 1, 1)
    return (probs * values).sum(dim=-3)


class DepthPredictor(nn.Module):
    def __init__(self, channels=256, bins=80, depth_min=2.0, depth_max=60.0):
        super().__init__()
        self.bins, self.depth_min, self.depth_max = bins, depth_min, depth_max
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.logits_head = nn.Conv2d(channels, bins + 1, 1)
        self.embed_head = nn.Conv2d(channels, channels, 1)

    def forward(self, f):
        h = self.body(f)
        logits = self.logits_head(h)
        depth_map = soft_argmax_depth(logits, self.bins, self.depth_min, self.depth_max)
        return logits, depth_map, self.embed_head(h)


def sample_depth_at(depth_map, ref_points):
    """Bilinear sample depth_map [B,h,w] at normalized (x,y) in [0,1] ref_points [B,N,2] -> [B,N]."""
    grid = (ref_points * 2.0 - 1.0).unsqueeze(2)  # [B,N,1,2]
    out = F.grid_sample(depth_map.unsqueeze(1), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out.view(depth_map.shape[0], -1)
