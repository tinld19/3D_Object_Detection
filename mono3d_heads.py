"""Split prediction heads, depth ensemble and KITTI box decode for Mono3D DETR."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mono3d_depth import sample_depth_at
from mono3d_targets import (backproject_center, multibin_decode,
                            ry_from_alpha)

# Relative-error priors (std as a fraction of depth) for the non-regressed sources.
GEO_REL_SIGMA = 0.15
MAP_REL_SIGMA = 0.10
_EPS = 1e-6


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out))


class Mono3DHeads(nn.Module):
    def __init__(self, num_classes, bins=12, mean_dims=None, d_model=256):
        super().__init__()
        self.num_classes, self.bins = num_classes, bins
        if mean_dims is None:
            mean_dims = torch.zeros(num_classes, 3)
        mean_dims = torch.as_tensor(mean_dims, dtype=torch.float32)
        assert mean_dims.shape == (num_classes, 3)
        self.register_buffer("mean_dims", mean_dims)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.box2d_head = _mlp(d_model, d_model, 4)
        self.center3d_head = _mlp(d_model, d_model, 2)
        self.depth_head = _mlp(d_model, d_model, 2)  # (d_reg raw, d_sigma raw)
        self.dims_head = _mlp(d_model, d_model, num_classes * 3)  # per-class residuals
        self.orient_head = _mlp(d_model, d_model, bins * 2)

    def forward(self, hidden):
        B, N, _ = hidden.shape
        logits = self.cls_head(hidden)
        depth_raw = self.depth_head(hidden)
        res = self.dims_head(hidden).view(B, N, self.num_classes, 3)
        dims_all = self.mean_dims.view(1, 1, -1, 3) + res
        cls = logits.argmax(-1)  # dims of the predicted class
        dims = dims_all.gather(2, cls[..., None, None].expand(B, N, 1, 3)).squeeze(2)
        orient = self.orient_head(hidden)
        return {
            "logits": logits,
            "boxes2d": self.box2d_head(hidden).sigmoid(),
            "center3d": self.center3d_head(hidden).sigmoid(),
            "d_reg": F.softplus(depth_raw[..., 0]) + _EPS,
            "d_sigma": F.softplus(depth_raw[..., 1]) + _EPS,
            "dims": dims,
            "dims_all": dims_all,  # [B,N,C,3], for losses using the GT class
            "orient_bin": orient[..., :self.bins],
            "orient_res": orient[..., self.bins:],
        }


def depth_ensemble(d_reg, d_geo, d_map, d_sigma, mode):
    """Inverse-variance fusion of the active depth sources.

    Variances: reg = d_sigma^2 (learned), geo = (GEO_REL_SIGMA*d_geo)^2,
    map = (MAP_REL_SIGMA*d_map)^2. mode 'rgb' uses only reg+geo (d_map ignored);
    other modes ('fusion'/'depth') use all three and require d_map.
    Returns (depth, {"reg","geo"[,"map"]: weights summing to 1}).
    """
    srcs = {"reg": (d_reg, d_sigma), "geo": (d_geo, GEO_REL_SIGMA * d_geo)}
    if mode != "rgb":
        if d_map is None:
            raise ValueError(f"mode '{mode}' requires d_map")
        srcs["map"] = (d_map, MAP_REL_SIGMA * d_map)
    inv = {k: 1.0 / (s.clamp_min(_EPS) ** 2) for k, (_, s) in srcs.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    depth = sum(weights[k] * srcs[k][0] for k in srcs)
    return depth, weights


def decode_boxes(outputs, ref_points, boxes2d_px_hw, depth_map, p2, mode):
    """Decode head outputs to KITTI boxes.

    boxes2d_px_hw: (img_h, img_w) pixel size of the image the p2 refers to.
    center3d is normalized (u/img_w, v/img_h) -> pixel (u, v) for backprojection.
    p2: flat 12-list (one per image, or a single list shared by the batch).
    Returns per image a list (per query) of {score, class, box=(h,w,l,x,y,z,ry)}.
    """
    img_h, img_w = boxes2d_px_hw
    logits = outputs["logits"]
    B, N, _ = logits.shape
    p2s = p2 if isinstance(p2[0], (list, tuple)) else [p2] * B
    score, cls = logits.sigmoid().max(-1)
    dims = outputs["dims"]
    f_y = torch.tensor([float(p[5]) for p in p2s], device=logits.device)[:, None]
    box_h_px = outputs["boxes2d"][..., 3] * img_h
    d_geo = f_y * dims[..., 0] / box_h_px.clamp_min(1.0)  # == geometric_depth, batched
    d_map = None if mode == "rgb" else sample_depth_at(depth_map, ref_points)
    z, _ = depth_ensemble(outputs["d_reg"], d_geo, d_map, outputs["d_sigma"], mode)
    bin_idx = outputs["orient_bin"].argmax(-1)
    bin_res = outputs["orient_res"].gather(-1, bin_idx[..., None]).squeeze(-1)

    results = []
    for b in range(B):
        per = []
        for n in range(N):
            zz = float(z[b, n])
            u = float(outputs["center3d"][b, n, 0]) * img_w
            v = float(outputs["center3d"][b, n, 1]) * img_h
            x, y = backproject_center(u, v, zz, p2s[b])
            alpha = multibin_decode(int(bin_idx[b, n]), float(bin_res[b, n]), outputs["orient_bin"].shape[-1])
            ry = ry_from_alpha(alpha, x, zz)
            h, w, l = (float(t) for t in dims[b, n])
            per.append({"score": float(score[b, n]), "class": int(cls[b, n]),
                        "box": (h, w, l, x, y, zz, ry)})
        results.append(per)
    return results
