"""分割指标：Dice / mIoU / Boundary F1（§15.2）。

Boundary F1 单独报：覆冰厚度的精度取决于边界，mIoU 对边界不敏感。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    inter = float((p & t).sum())
    return (2 * inter + eps) / (p.sum() + t.sum() + eps)


def iou_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    inter = float((p & t).sum())
    union = float((p | t).sum())
    return (inter + eps) / (union + eps)


def dice_multiclass(
    pred: np.ndarray, target: np.ndarray, classes: list[int], eps: float = 1e-6
) -> float:
    scores = [dice_binary(pred == c, target == c, eps) for c in classes]
    return float(np.mean(scores)) if scores else 0.0


def miou(pred: np.ndarray, target: np.ndarray, classes: list[int]) -> float:
    return float(np.mean([iou_binary(pred == c, target == c) for c in classes]))


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    er = binary_dilation(m) & ~m
    return er


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: float = 2.0) -> float:
    """容差 `tolerance_px` 内的边界 F1。"""
    bp = _boundary(pred)
    bt = _boundary(target)
    if not bp.any() and not bt.any():
        return 1.0
    if not bp.any() or not bt.any():
        return 0.0

    dt_t = distance_transform_edt(~bt)
    dt_p = distance_transform_edt(~bp)
    prec = float((dt_t[bp] <= tolerance_px).mean())
    rec = float((dt_p[bt] <= tolerance_px).mean())
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def coverage_from_mask(
    mask: np.ndarray, ice_classes: tuple[int, ...] = (2, 3), exposed_class: int = 1
) -> float | None:
    ice = int(np.isin(mask, ice_classes).sum())
    exp = int((mask == exposed_class).sum())
    if ice + exp == 0:
        return None
    return ice / (ice + exp)
