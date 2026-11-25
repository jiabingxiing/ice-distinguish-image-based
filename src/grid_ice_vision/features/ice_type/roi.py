"""终端归一化 ROI 裁切与视角偏移检查（纯函数，无文件 I/O）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizedRoi:
    """归一化坐标框 [0, 1]。"""

    monitor_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    asset_type: str = "unknown"
    view_version: str = "v1"

    def __post_init__(self) -> None:
        if not (0.0 <= self.x1 < self.x2 <= 1.0 and 0.0 <= self.y1 < self.y2 <= 1.0):
            raise ValueError(
                f"归一化 ROI 越界或非正面积（monitor={self.monitor_id}）: "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )


def crop_normalized_roi(
    image_bgr: np.ndarray,
    roi: NormalizedRoi,
    *,
    min_side: int = 16,
) -> np.ndarray:
    """按归一化坐标裁切 BGR 区域，自动钳位并保证最小边长。"""
    h, w = image_bgr.shape[:2]
    px1 = max(0, min(w - 1, round(roi.x1 * w)))
    py1 = max(0, min(h - 1, round(roi.y1 * h)))
    px2 = max(px1 + 1, min(w, round(roi.x2 * w)))
    py2 = max(py1 + 1, min(h, round(roi.y2 * h)))

    if px2 - px1 < min_side and w >= min_side:
        cx = (px1 + px2) // 2
        px1 = max(0, min(w - min_side, cx - min_side // 2))
        px2 = px1 + min_side
    if py2 - py1 < min_side and h >= min_side:
        cy = (py1 + py2) // 2
        py1 = max(0, min(h - min_side, cy - min_side // 2))
        py2 = py1 + min_side

    return np.ascontiguousarray(image_bgr[py1:py2, px1:px2])


def center_fallback_crop(image_bgr: np.ndarray, crop_ratio: float = 0.70) -> np.ndarray:
    """缺失终端 ROI 时的中心保护性裁切（聚焦杆塔/绝缘子常驻中心区）。"""
    h, w = image_bgr.shape[:2]
    ratio = float(np.clip(crop_ratio, 0.2, 1.0))
    ch = max(1, round(h * ratio))
    cw = max(1, round(w * ratio))
    top = (h - ch) // 2
    left = (w - cw) // 2
    return np.ascontiguousarray(image_bgr[top : top + ch, left : left + cw])


def compute_view_signature(roi_bgr: np.ndarray, size: int = 48) -> np.ndarray:
    """计算 ROI 零均值结构边缘指纹（用于检测同终端机位变焦/预置位偏移）。"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vec = mag.reshape(-1)
    vec = vec - float(vec.mean())
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def detect_view_drift(
    roi_bgr: np.ndarray,
    ref_signature: np.ndarray,
    *,
    corr_min: float = 0.35,
    size: int = 48,
) -> tuple[bool, float]:
    """比较当前 ROI 与参考视角边缘指纹相似度；低于 corr_min 视为视角偏移。"""
    cur = compute_view_signature(roi_bgr, size=size)
    ref = np.asarray(ref_signature, dtype=np.float32).reshape(-1)
    if cur.shape != ref.shape:
        raise ValueError(f"指纹维度不匹配: {cur.shape} vs {ref.shape}")
    score = float(np.clip(np.dot(cur, ref), 0.0, 1.0))
    return score < corr_min, round(score, 4)
