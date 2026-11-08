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
