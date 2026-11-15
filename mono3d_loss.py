"""Hungarian matching and set-prediction loss for the Mono3D DETR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mono3d_depth import soft_argmax_depth
from mono3d_targets import bin_value

DEFAULT_WEIGHTS = {
    "class": 2.0, "box": 5.0, "giou": 2.0, "center": 10.0, "depth": 1.0,
    "dim": 1.0, "orient": 1.0, "depthmap": 1.0,
}
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DEPTH_BINS = 80


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def pairwise_giou(a, b):
    """a:[N,4], b:[M,4] in xyxy -> [N,M] generalized IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(a[:, None, :2], b[None, :, :2])
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])
    enc = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou - (enc - union) / enc.clamp(min=1e-7)


def paired_giou(a, b):
    """Elementwise GIoU for matched pairs a,b:[K,4] xyxy -> [K]."""
    if a.shape[0] == 0:
        return a.new_zeros(0)
    return pairwise_giou(a, b).diagonal()


def _focal_cost(logits, labels):
    prob = logits.sigmoid()
    neg = (1 - FOCAL_ALPHA) * prob ** FOCAL_GAMMA * -(1 - prob + 1e-8).log()
    pos = FOCAL_ALPHA * (1 - prob) ** FOCAL_GAMMA * -(prob + 1e-8).log()
    return (pos - neg)[:, labels]


@torch.no_grad()
def hungarian_match(outputs_layer, targets, weights):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    res = []
    for b, t in enumerate(targets):
        g = t["labels"].shape[0]
        if g == 0:
            e = torch.empty(0, dtype=torch.long)
            res.append((e, e.clone()))
            continue
        cls = _focal_cost(outputs_layer["logits"][b].float(), t["labels"])
        pb, tb = outputs_layer["boxes2d"][b].float(), t["boxes2d"].float()
        c_box = torch.cdist(pb, tb, p=1)
        c_giou = 1 - pairwise_giou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(tb))
        c_ctr = torch.cdist(outputs_layer["center3d"][b].float(), t["center3d"].float(), p=1)
        c_dep = (outputs_layer["depth"][b].float()[:, None] - t["depth"].float()[None]).abs()
        cost = (w["class"] * cls + w["box"] * c_box + w["giou"] * c_giou
                + w["center"] * c_ctr + w["depth"] * c_dep)
        cost = torch.nan_to_num(cost.detach().cpu(), nan=1e6, posinf=1e6, neginf=-1e6)
        pi, gi = linear_sum_assignment(cost.numpy())
        res.append((torch.as_tensor(pi, dtype=torch.long), torch.as_tensor(gi, dtype=torch.long)))
    return res


def _sigmoid_focal(logits, targets):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t) ** FOCAL_GAMMA
    alpha_t = FOCAL_ALPHA * targets + (1 - FOCAL_ALPHA) * (1 - targets)
    return (alpha_t * loss).sum()


class SetCriterion(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def _layer_losses(self, layer, targets):
        matches = hungarian_match(layer, targets, self.weights)
        logits = layer["logits"]
        dev = logits.device
        num_gt = max(sum(int(t["labels"].shape[0]) for t in targets), 1)

        cls_t = torch.zeros_like(logits)
        picks = {k: [] for k in ("pb", "gb", "ctr", "dep", "sig", "dim", "ob", "or_")}
        gt = {k: [] for k in ("box", "ctr", "dep", "dim", "bin", "res")}
        for b, (pi, gi) in enumerate(matches):
            if pi.numel() == 0:
                continue
            pi, gi = pi.to(dev), gi.to(dev)
            t = targets[b]
            labels = t["labels"][gi]
            cls_t[b, pi, labels] = 1.0
            picks["pb"].append(layer["boxes2d"][b, pi])
            picks["ctr"].append(layer["center3d"][b, pi])
            picks["dep"].append(layer["depth"][b, pi])
            picks["sig"].append(layer["d_sigma"][b, pi])
            picks["dim"].append(layer["dims_all"][b, pi, labels])
            picks["ob"].append(layer["orient_bin"][b, pi])
            picks["or_"].append(layer["orient_res"][b, pi])
            gt["box"].append(t["boxes2d"][gi])
            gt["ctr"].append(t["center3d"][gi])
            gt["dep"].append(t["depth"][gi])
            gt["dim"].append(t["dims"][gi])
            gt["bin"].append(t["orient_bin"][gi])
            gt["res"].append(t["orient_res"][gi])

        zero = logits.new_zeros(())
        out = {"loss_class": _sigmoid_focal(logits, cls_t) / num_gt}
        if not picks["pb"]:
            for k in ("loss_box", "loss_giou", "loss_center", "loss_depth", "loss_dim", "loss_orient"):
                out[k] = zero
            return out
        cat = lambda xs: torch.cat(xs, 0)
        pb, gb = cat(picks["pb"]), cat(gt["box"])
        out["loss_box"] = F.l1_loss(pb, gb, reduction="sum") / num_gt
        out["loss_giou"] = (1 - paired_giou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb))).sum() / num_gt
        out["loss_center"] = F.l1_loss(cat(picks["ctr"]), cat(gt["ctr"]), reduction="sum") / num_gt
        sigma = cat(picks["sig"]).clamp(min=1e-6)
        err = (cat(picks["dep"]) - cat(gt["dep"])).abs()
        out["loss_depth"] = (math.sqrt(2.0) * err / sigma + sigma.log()).sum() / num_gt
        out["loss_dim"] = F.l1_loss(cat(picks["dim"]), cat(gt["dim"]), reduction="sum") / num_gt
        gbin = cat(gt["bin"])
        ce = F.cross_entropy(cat(picks["ob"]), gbin, reduction="sum")
        pres = cat(picks["or_"]).gather(1, gbin[:, None]).squeeze(1)
        gres = cat(gt["res"]).gather(1, gbin[:, None]).squeeze(1)
        out["loss_orient"] = (ce + F.smooth_l1_loss(pres, gres, reduction="sum")) / num_gt
        return out

    def _depthmap_loss(self, outputs, targets, dm_target, dm_mask, like):
        logits = outputs.get("depth_logits")
        if logits is None:
            return like.new_zeros(())
        if dm_target is None:
            dm_target = outputs.get("depth_map_target")
        if dm_mask is None:
            dm_mask = outputs.get("depth_map_mask")
        if dm_target is None and targets and "depth_map_target" in targets[0]:
            dm_target = torch.stack([t["depth_map_target"] for t in targets])
            dm_mask = torch.stack([t["depth_map_mask"] for t in targets])
        if dm_target is None or dm_mask is None:
            raise ValueError(
                "depth branch active but no depth_map_target/depth_map_mask provided; "
                "pass them to SetCriterion.forward")
        dm_target, dm_mask = dm_target.to(logits.device), dm_mask.to(logits.device)
        ce = F.cross_entropy(logits, dm_target, reduction="none")
        n = dm_mask.sum().clamp(min=1)
        loss_ce = (ce * dm_mask).sum() / n
        fg = dm_mask & (dm_target < DEPTH_BINS)
        values = torch.tensor([bin_value(i) for i in range(DEPTH_BINS)],
                              dtype=logits.dtype, device=logits.device)
        painted = values[dm_target.clamp(max=DEPTH_BINS - 1)]
        l1 = (soft_argmax_depth(logits) - painted).abs()
        loss_l1 = (l1 * fg).sum() / fg.sum().clamp(min=1)
        return loss_ce + loss_l1

    def forward(self, outputs, targets, depth_map_target=None, depth_map_mask=None):
        """depth_map_target/mask: batch tensors from collate; may instead live in outputs or targets[i]."""
        keys = ("loss_class", "loss_box", "loss_giou", "loss_center", "loss_depth", "loss_dim", "loss_orient")
        total_losses = None
        for layer in outputs["aux"]:
            ll = self._layer_losses(layer, targets)
            total_losses = ll if total_losses is None else {k: total_losses[k] + ll[k] for k in keys}
        total_losses["loss_depthmap"] = self._depthmap_loss(
            outputs, targets, depth_map_target, depth_map_mask, total_losses["loss_class"])
        w = self.weights
        wmap = {"loss_class": "class", "loss_box": "box", "loss_giou": "giou", "loss_center": "center",
                "loss_depth": "depth", "loss_dim": "dim", "loss_orient": "orient", "loss_depthmap": "depthmap"}
        total_losses["total"] = sum(w[v] * total_losses[k] for k, v in wmap.items())
        return total_losses
