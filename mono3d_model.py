"""Assembled Mono3D DETR model: encoder -> depth predictor -> seeded decoder -> heads."""
import torch
import torch.nn as nn

from mono3d_decoder import DepthAwareDecoder
from mono3d_depth import DepthPredictor, sample_depth_at
from mono3d_encoder import TrainableEncoder
from mono3d_heads import Mono3DHeads, decode_boxes, depth_ensemble


class MonoDetr3D(nn.Module):
    def __init__(self, num_classes, mean_dims, feature_mode="fusion", num_queries=50,
                 decoder_layers=3, bins=80, orient_bins=12):
        super().__init__()
        self.feature_mode = feature_mode
        self.num_queries = num_queries
        self.encoder = TrainableEncoder()
        self.depth_predictor = DepthPredictor(bins=bins) if feature_mode != "rgb" else None
        self.decoder = DepthAwareDecoder(layers=decoder_layers, feature_mode=feature_mode,
                                         num_queries=num_queries)
        self.heads = Mono3DHeads(num_classes, bins=orient_bins, mean_dims=mean_dims)

    def forward(self, f0, proposals, p2, image_size, jitter=False):
        img_h, _ = image_size
        feat = self.encoder(f0)
        if self.depth_predictor is not None:
            logits, depth_map, depth_embed = self.depth_predictor(feat)
        else:
            logits = depth_map = depth_embed = None
        content, ref0, valid_mask = self.decoder.seed(proposals, feat, jitter)
        hiddens, refs = self.decoder.forward(content, ref0, feat, depth_embed)

        B = f0.shape[0]
        p2s = p2 if isinstance(p2[0], (list, tuple)) else [p2] * B
        f_y = torch.tensor([float(p[5]) for p in p2s], device=f0.device, dtype=f0.dtype)[:, None]
        aux = []
        for hidden, ref in zip(hiddens, refs):
            out = self.heads(hidden)
            box_h_px = out["boxes2d"][..., 3] * img_h
            d_geo = f_y * out["dims"][..., 0] / box_h_px.clamp_min(1.0)
            d_map = sample_depth_at(depth_map, ref) if depth_map is not None else None
            depth, _ = depth_ensemble(out["d_reg"], d_geo, d_map, out["d_sigma"], self.feature_mode)
            out["depth"] = depth
            out["ref_points"] = ref
            out["valid_mask"] = valid_mask
            aux.append(out)
        return {"aux": aux, "depth_logits": logits, "depth_map": depth_map}

    @torch.no_grad()
    def predict(self, f0, proposals, p2, image_size, score_thr=0.05, mode=None):
        mode = mode or self.feature_mode
        res = self.forward(f0, proposals, p2, image_size, jitter=False)
        last = res["aux"][-1]
        decoded = decode_boxes(last, last["ref_points"], tuple(image_size),
                               res["depth_map"], p2, mode)
        out = []
        for per in decoded:
            keep = [d for d in per if d["score"] >= score_thr]
            keep.sort(key=lambda d: d["score"], reverse=True)
            out.append(keep[:self.num_queries])
        return out
