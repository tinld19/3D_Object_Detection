"""Proposal-guided query seeding and depth-aware transformer decoder for Mono3D DETR."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional

FEATURE_MODES = ("fusion", "rgb", "depth")


def _roi_pool(feature: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Average-pool feature [C,h,w] over a normalized (cx,cy,w,h) box -> [C]."""
    _, height, width = feature.shape
    cx, cy, bw, bh = (float(v) for v in box)
    x0, x1 = cx - bw / 2, cx + bw / 2
    y0, y1 = cy - bh / 2, cy + bh / 2
    left = max(0, min(width - 1, math.floor(x0 * width)))
    top = max(0, min(height - 1, math.floor(y0 * height)))
    right = max(left + 1, min(width, math.ceil(x1 * width)))
    bottom = max(top + 1, min(height, math.ceil(y1 * height)))
    return functional.adaptive_avg_pool2d(feature[:, top:bottom, left:right], 1).flatten()


def seed_queries(proposals, f, num_queries=50, jitter=False, learned_embed=None):
    """Seed decoder queries from 2D proposals; pad with learned embeddings.

    learned_embed: optional [N,C] tensor (e.g. an nn.Parameter) used for padded
    slots; zeros when omitted. Returns (content [B,N,C], ref_points [B,N,2],
    valid_mask [B,N] bool).
    """
    batch, channels = f.shape[0], f.shape[1]
    if learned_embed is None:
        learned_embed = f.new_zeros(num_queries, channels)
    contents, refs, masks = [], [], []
    for index in range(batch):
        boxes = proposals[index]["boxes2d"].detach().to(f.device).float()
        scores = proposals[index]["scores"].detach().to(f.device).float()
        if boxes.shape[0] > 0:
            order = scores.argsort(descending=True)
            if jitter:
                keep = torch.rand(order.numel(), device=f.device) > 0.1
                order = order[keep]
            order = order[:num_queries]
            boxes = boxes[order]
            if jitter and boxes.shape[0] > 0:
                noise = torch.randn_like(boxes) * 0.02
                noise[:, 2:] *= 2
                boxes = boxes + noise * boxes[:, 2:].repeat(1, 2).clamp(min=0.05)
                boxes = torch.cat((boxes[:, :2].clamp(0, 1), boxes[:, 2:].clamp(0.01, 1)), dim=1)
        else:
            boxes = boxes.reshape(0, 4)
        count = boxes.shape[0]
        pooled = (
            torch.stack([_roi_pool(f[index], box) for box in boxes])
            if count
            else f.new_zeros(0, channels)
        )
        content = torch.cat((pooled, learned_embed[count:num_queries].to(f.dtype)), dim=0)
        ref = torch.cat((boxes[:, :2], f.new_full((num_queries - count, 2), 0.5)), dim=0)
        mask = torch.zeros(num_queries, dtype=torch.bool, device=f.device)
        mask[:count] = True
        contents.append(content)
        refs.append(ref)
        masks.append(mask)
    return torch.stack(contents), torch.stack(refs), torch.stack(masks)


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, channels=256, heads=8, ffn=1024, use_depth=True, use_visual=True, dropout=0.0):
        super().__init__()
        self.use_depth = use_depth
        self.use_visual = use_visual
        self.pos_mlp = nn.Sequential(nn.Linear(2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.self_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.depth_cross_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.visual_cross_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn), nn.ReLU(), nn.Linear(ffn, channels))
        self.norm4 = nn.LayerNorm(channels)
        self.ref_delta = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 2))

    def forward(self, content, ref_points, f_tokens, depth_tokens):
        pos = self.pos_mlp(ref_points)
        q = content + pos
        content = self.norm1(content + self.self_attn(q, q, content, need_weights=False)[0])
        if self.use_depth:
            q = content + pos
            content = self.norm2(content + self.depth_cross_attn(q, depth_tokens, depth_tokens, need_weights=False)[0])
        if self.use_visual:
            q = content + pos
            content = self.norm3(content + self.visual_cross_attn(q, f_tokens, f_tokens, need_weights=False)[0])
        content = self.norm4(content + self.ffn(content))
        logit = torch.logit(ref_points.clamp(1e-4, 1 - 1e-4)) + self.ref_delta(content)
        return content, torch.sigmoid(logit)


class DepthAwareDecoder(nn.Module):
    def __init__(self, layers=3, feature_mode="fusion", channels=256, heads=8, num_queries=50):
        super().__init__()
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
        self.feature_mode = feature_mode
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, channels) * 0.02)
        use_depth = feature_mode in ("fusion", "depth")
        use_visual = feature_mode in ("fusion", "rgb")
        self.layers = nn.ModuleList(
            DepthAwareDecoderLayer(channels, heads, use_depth=use_depth, use_visual=use_visual)
            for _ in range(layers)
        )

    def seed(self, proposals, f, jitter=False):
        return seed_queries(proposals, f, self.num_queries, jitter, learned_embed=self.query_embed)

    def forward(self, content, ref_points, f, depth_embed):
        use_depth = self.layers[0].use_depth
        use_visual = self.layers[0].use_visual
        f_tokens = f.flatten(2).transpose(1, 2) if use_visual else None
        depth_tokens = depth_embed.flatten(2).transpose(1, 2) if use_depth else None
        hidden, refs = [], []
        for layer in self.layers:
            content, ref_points = layer(content, ref_points, f_tokens, depth_tokens)
            hidden.append(content)
            refs.append(ref_points)
        return hidden, refs
