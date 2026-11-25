"""覆冰类型识别质量门控（复用 T0 质检算子，增加局部/ROI 清晰度与动态范围综合评分）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from grid_ice_vision.common.schemas import QualityInfo, QualityStatus
from grid_ice_vision.features.quality.gate import QualityThresholds, assess_quality


@dataclass(frozen=True)
class IceTypeQualityThresholds:
    """覆冰类型质量筛选阈值（全部参数通过配置注入，禁硬编码魔法数字）。"""

    tau_q: float = 0.25
    blur_weight: float = 0.45
    exposure_weight: float = 0.30
    contrast_weight: float = 0.25
    min_roi_side: int = 24
    min_roi_std: float = 4.0
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> IceTypeQualityThresholds:
        if not cfg:
            return cls()
        q = cfg.get("quality", cfg)
        return cls(
            tau_q=float(q.get("tau_q", 0.25)),
            blur_weight=float(q.get("blur_weight", 0.45)),
            exposure_weight=float(q.get("exposure_weight", 0.30)),
            contrast_weight=float(q.get("contrast_weight", 0.25)),
            min_roi_side=int(q.get("min_roi_side", 24)),
            min_roi_std=float(q.get("min_roi_std", 4.0)),
            quality_thresholds=QualityThresholds.from_config(cfg),
        )


@dataclass(frozen=True)
class IceTypeQuality:
    """单帧（含可选 ROI）质量评分结果。"""

    q_image: float
    usable: bool
    blur_score: float
    exposure_ratio: float
    contrast_score: float
    roi_readable: bool
    quality_info: QualityInfo
    reasons: list[str]


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def evaluate_ice_type_quality(
    image_bgr: np.ndarray,
    thresholds: IceTypeQualityThresholds | None = None,
    roi_image_bgr: np.ndarray | None = None,
) -> IceTypeQuality:
    """评估全图与可选 ROI 是否满足覆冰类型视觉判别最低质量要求。"""
    th = thresholds or IceTypeQualityThresholds()
    q_info = assess_quality(image_bgr, th.quality_thresholds)
    gray = _gray(image_bgr)

    over = float((gray > 250).mean())
    under = float((gray < 15).mean())
    exposure_ratio = max(over, under)
    exp_limit = max(th.quality_thresholds.exposure_unusable, 1e-6)
    exposure_score = float(np.clip(1.0 - exposure_ratio / exp_limit, 0.0, 1.0))

    p5, p95 = np.percentile(gray, [5, 95])
    dyn = float(p95 - p5) / 180.0
    std_s = float(gray.std()) / 50.0
    contrast_score = float(np.clip(0.6 * dyn + 0.4 * std_s, 0.0, 1.0))
    sharpness_score = float(np.clip(1.0 - q_info.blur_score, 0.0, 1.0))

    w_sum = max(th.blur_weight + th.exposure_weight + th.contrast_weight, 1e-6)
    q_image = (
        th.blur_weight * sharpness_score
        + th.exposure_weight * exposure_score
        + th.contrast_weight * contrast_score
    ) / w_sum
    q_image = float(np.clip(q_image, 0.0, 1.0))

    reasons = list(q_info.reasons)
    roi_readable = True
    if roi_image_bgr is not None:
        rh, rw = roi_image_bgr.shape[:2]
        roi_std = float(roi_image_bgr.std()) if roi_image_bgr.size else 0.0
        if rh < th.min_roi_side or rw < th.min_roi_side or roi_std < th.min_roi_std:
            roi_readable = False
            reasons.append(f"roi_unreadable(size={rw}x{rh},std={roi_std:.1f})")
            q_image = min(q_image, max(0.0, th.tau_q - 0.02))

    if q_info.status is QualityStatus.UNUSABLE:
        q_image = min(q_image, max(0.0, th.tau_q - 0.05))

    usable = q_info.status is not QualityStatus.UNUSABLE and q_image >= th.tau_q and roi_readable
    if not usable and "low_image_quality" not in reasons:
        reasons.append(f"q_below_threshold({q_image:.3f}<{th.tau_q:.3f})")

    return IceTypeQuality(
        q_image=round(q_image, 4),
        usable=usable,
        blur_score=q_info.blur_score,
        exposure_ratio=round(exposure_ratio, 4),
        contrast_score=round(contrast_score, 4),
        roi_readable=roi_readable,
        quality_info=q_info,
        reasons=reasons,
    )
