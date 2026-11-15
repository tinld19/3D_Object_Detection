"""Pure geometry and KITTI target helpers for depth-aware Mono3D DETR."""
import math


def _wrap_to_pi(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bin_value(index: int, depth_min: float = 2.0, depth_max: float = 60.0, bins: int = 80) -> float:
    """Inverse LID: lower depth edge of bin ``index`` (inverse of ``depth_bin``)."""
    delta = 2 * (depth_max - depth_min) / (bins * (bins + 1))
    return depth_min + delta * index * (index + 1) / 2


def rescale_p2(p2: list[float], orig_size: tuple[int, int], in_size: tuple[int, int]) -> list[float]:
    """Rescale a 12-value row-major P2 for a resize; sizes are (W, H)."""
    sx = in_size[0] / orig_size[0]
    sy = in_size[1] / orig_size[1]
    out = list(p2)
    out[0] *= sx
    out[2] *= sx
    out[5] *= sy
    out[6] *= sy
    return out


def geometric_depth(f_y: float, phys_height: float, box_height_px: float) -> float:
    return f_y * phys_height / max(box_height_px, 1.0)


def ray_angle(x: float, z: float) -> float:
    return math.atan2(x, z)


def alpha_from_ry(ry: float, x: float, z: float) -> float:
    return _wrap_to_pi(ry - ray_angle(x, z))


def ry_from_alpha(alpha: float, x: float, z: float) -> float:
    return _wrap_to_pi(alpha + ray_angle(x, z))


def multibin_target(alpha: float, bins: int = 12) -> tuple[int, float]:
    """Return (bin_index, residual) with residual relative to the bin center."""
    width = 2 * math.pi / bins
    alpha = _wrap_to_pi(alpha)
    index = min(bins - 1, int((alpha + math.pi) // width))
    center = -math.pi + (index + 0.5) * width
    return index, alpha - center


def multibin_decode(bin_index: int, residual: float, bins: int = 12) -> float:
    width = 2 * math.pi / bins
    return _wrap_to_pi(-math.pi + (bin_index + 0.5) * width + residual)


def project_center(x: float, y: float, z: float, p2: list[float]) -> tuple[float, float]:
    w = p2[8] * x + p2[9] * y + p2[10] * z + p2[11]
    u = (p2[0] * x + p2[1] * y + p2[2] * z + p2[3]) / w
    v = (p2[4] * x + p2[5] * y + p2[6] * z + p2[7]) / w
    return u, v


def backproject_center(u: float, v: float, z: float, p2: list[float]) -> tuple[float, float]:
    """Solve (x, y) from pixel (u, v) and depth z under P2."""
    # u*w = row0.[x,y,z,1], v*w = row1.[x,y,z,1], w = row2.[x,y,z,1]
    a11, a12 = p2[0] - u * p2[8], p2[1] - u * p2[9]
    a21, a22 = p2[4] - v * p2[8], p2[5] - v * p2[9]
    b1 = -(p2[2] - u * p2[10]) * z - (p2[3] - u * p2[11])
    b2 = -(p2[6] - v * p2[10]) * z - (p2[7] - v * p2[11])
    det = a11 * a22 - a12 * a21
    return (b1 * a22 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det
