"""Pure-logic KITTI 3D evaluation helpers (numpy + shapely only)."""
import numpy as np
from shapely.geometry import Polygon


def _bev_polygon(x, z, l, w, ry):
    """Rotated footprint: local x in [-l/2, l/2], local z in [-w/2, w/2]."""
    c, s = np.cos(ry), np.sin(ry)
    lx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    lz = np.array([w / 2, -w / 2, -w / 2, w / 2])
    # Same rotation as infer_3d.project_cuboid (rotation about the y axis).
    cx = c * lx + s * lz + x
    cz = -s * lx + c * lz + z
    return Polygon(np.stack([cx, cz], axis=1))


def _inter_area(a, b):
    """Intersection area of two BEV boxes (x, z, l, w, ry); 0.0 on any failure."""
    try:
        if min(a[2], a[3], b[2], b[3]) <= 0:
            return 0.0
        pa, pb = _bev_polygon(*a), _bev_polygon(*b)
        if not (pa.is_valid and pb.is_valid):
            return 0.0
        inter = pa.intersection(pb)
        if inter.is_empty or inter.geom_type not in ("Polygon", "MultiPolygon"):
            return 0.0
        area = float(inter.area)
        return area if np.isfinite(area) else 0.0
    except Exception:
        return 0.0


def object_difficulty(bbox_height_px, occlusion, truncation):
    """KITTI difficulty: 0 easy, 1 moderate, 2 hard, None if ignored (below Hard)."""
    if bbox_height_px >= 40 and occlusion <= 0 and truncation <= 0.15:
        return 0
    if bbox_height_px >= 25 and occlusion <= 1 and truncation <= 0.30:
        return 1
    if bbox_height_px >= 25 and occlusion <= 2 and truncation <= 0.50:
        return 2
    return None


def bev_iou(a, b):
    """BEV IoU of boxes (x, z, l, w, ry)."""
    try:
        inter = _inter_area(a, b)
        union = a[2] * a[3] + b[2] * b[3] - inter
        if inter <= 0 or union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0


def iou_3d(a, b):
    """3D IoU of boxes (h, w, l, x, y, z, ry); y is the box bottom."""
    try:
        ha, wa, la, xa, ya, za, ra = a
        hb, wb, lb, xb, yb, zb, rb = b
        if min(ha, hb) <= 0:
            return 0.0
        h_overlap = min(ya, yb) - max(ya - ha, yb - hb)
        if h_overlap <= 0:
            return 0.0
        inter = _inter_area((xa, za, la, wa, ra), (xb, zb, lb, wb, rb)) * h_overlap
        union = ha * wa * la + hb * wb * lb - inter
        if inter <= 0 or union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0


NEIGHBOURS = {"Car": {"Van"}, "Pedestrian": {"Person_sitting"}, "Cyclist": set()}


def match_image(preds, gts, iou_fn, iou_thr, difficulty, eval_class, neighbours):
    """Greedy KITTI matching within one image.

    Returns (scored_flags, num_valid_gt). Preds matching a neighbour-class GT
    or an ignored/too-hard eval-class GT are dropped (don't-care); unmatched
    preds are FP.
    """
    nbr = neighbours.get(eval_class, set())

    def is_valid(g):
        return (g["class"] == eval_class and g["difficulty"] is not None
                and g["difficulty"] <= difficulty)

    candidates = [i for i, g in enumerate(gts)
                  if g["class"] == eval_class or g["class"] in nbr]
    num_valid_gt = sum(1 for g in gts if is_valid(g))
    used = set()
    flags = []
    for pred in sorted(preds, key=lambda p: p["score"], reverse=True):
        best_valid, best_iou = None, iou_thr
        ignored = False
        for i in candidates:
            iou = iou_fn(pred["box"], gts[i]["box"])
            if iou <= iou_thr:
                continue
            if is_valid(gts[i]):
                if i not in used and iou > best_iou:
                    best_valid, best_iou = i, iou
            else:
                ignored = True  # neighbour / ignored GT: never consumed
        if best_valid is not None:
            used.add(best_valid)
            flags.append((pred["score"], True))
        elif not ignored:
            flags.append((pred["score"], False))
        # else don't-care: pred removed
    return flags, num_valid_gt


def ap40(scored_flags, num_valid_gt):
    """AP|R40: mean over recall targets k/40 (k=1..40, recall 0 excluded) of
    the monotone-interpolated precision (max precision at recall >= target).

    scored_flags: pooled (score, is_tp) over all images; sorted by score
    descending here. NaN if num_valid_gt == 0.
    """
    if num_valid_gt == 0:
        return float("nan")
    flags = sorted(scored_flags, key=lambda f: f[0], reverse=True)
    points = []  # (recall, precision) per operating point
    tp = fp = 0
    for _, is_tp in flags:
        if is_tp:
            tp += 1
        else:
            fp += 1
        points.append((tp / num_valid_gt, tp / (tp + fp)))
    total = 0.0
    for k in range(1, 41):
        target = k / 40 - 1e-9
        total += max((p for r, p in points if r >= target), default=0.0)
    return total / 40


_LEVELS = {"easy": 0, "moderate": 1, "hard": 2}


def _to_bev(box7):
    h, w, l, x, y, z, ry = box7
    return (x, z, l, w, ry)


def evaluate(images, classes, iou_thresholds, metrics):
    """result[class][metric][iou_thr][difficulty_name] = AP40.

    images: list of {"preds": [{score, box7, class}], "gts": [{box7, class,
    difficulty}]} with box7 = (h, w, l, x, y, z, ry).
    """
    result = {}
    for cls in classes:
        result[cls] = {}
        for metric in metrics:
            if metric == "3d":
                iou_fn, conv = iou_3d, (lambda b: b)
            elif metric == "bev":
                iou_fn, conv = bev_iou, _to_bev
            else:
                raise ValueError(f"unknown metric {metric}")
            per_img = [
                ([{"score": p["score"], "box": conv(p["box"])}
                  for p in im["preds"] if p["class"] == cls],
                 [{"box": conv(g["box"]), "class": g["class"],
                   "difficulty": g["difficulty"]} for g in im["gts"]])
                for im in images]
            result[cls][metric] = {}
            for thr in iou_thresholds[cls]:
                result[cls][metric][thr] = {}
                for name, level in _LEVELS.items():
                    flags, n = [], 0
                    for preds, gts in per_img:
                        f, ng = match_image(preds, gts, iou_fn, thr, level,
                                            cls, NEIGHBOURS)
                        flags.extend(f)
                        n += ng
                    result[cls][metric][thr][name] = ap40(flags, n)
    return result


_COLUMNS = (("easy", "Easy"), ("moderate", "Moderate"), ("hard", "Hard"))
_METRIC_TITLES = (("3d", "3D AP"), ("bev", "BEV AP"))


def _cell(value):
    return "nan" if value is None or value != value else f"{value:.2f}"


def format_table(result, feature_mode):
    """Markdown tables: a 3D AP block then a BEV AP block (class x IoU rows)."""
    blocks = []
    for key, title in _METRIC_TITLES:
        lines = [f"{title} ({feature_mode})", "",
                 "| Class | IoU | Easy | Moderate | Hard |",
                 "| --- | --- | --- | --- | --- |"]
        for cls, metrics in result.items():
            for thr, diffs in metrics.get(key, {}).items():
                cells = " | ".join(_cell(diffs.get(n)) for n, _ in _COLUMNS)
                lines.append(f"| {cls} | {float(thr):.2f} | {cells} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_comparison(results):
    """Markdown ablation table: one row per feature_mode."""
    header = ["Feature mode"]
    keys = []
    for key, title in _METRIC_TITLES:
        for cls, metrics in next(iter(results.values()), {}).items():
            for thr in metrics.get(key, {}):
                for name, label in _COLUMNS:
                    header.append(f"{title} {cls}@{float(thr):.2f} {label}")
                    keys.append((cls, key, thr, name))
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for mode, result in results.items():
        cells = []
        for cls, key, thr, name in keys:
            try:
                cells.append(_cell(result[cls][key][thr][name]))
            except KeyError:
                cells.append("nan")
        lines.append("| " + " | ".join([mode] + cells) + " |")
    return "\n".join(lines)
