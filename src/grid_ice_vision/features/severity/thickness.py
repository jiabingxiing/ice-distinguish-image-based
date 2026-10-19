"""覆冰厚度几何解算。

纯函数模块：不 import 任何模型代码、不读写文件、不打印日志，
因此可以在没有任何权重、没有任何数据的情况下被完整单测（§10.6、§13.3）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from grid_ice_vision.common.constants import RHO_STANDARD

CalibSource = Literal["rigid_fitting", "ptz", "baseline_archive", "external"]

_SOURCE_PRIORITY: dict[str, int] = {
    "rigid_fitting": 0,
    "ptz": 1,
    "external": 1,
    "baseline_archive": 2,
}


@dataclass(frozen=True)
class ScaleReference:
    """尺度参考：某个已知物理尺寸的物体在图像中的像素宽度。"""

    d_ref_mm: float
    w_ref_px: float
    source: CalibSource

    def __post_init__(self) -> None:
        if self.d_ref_mm <= 0:
            raise ValueError(f"d_ref_mm 必须为正，得到 {self.d_ref_mm}")
        if self.w_ref_px <= 0:
            raise ValueError(f"w_ref_px 必须为正，得到 {self.w_ref_px}")

    @property
    def alpha(self) -> float:
        """尺度因子 mm/px。"""
        return self.d_ref_mm / self.w_ref_px


def fuse_scale_references(
    refs: list[ScaleReference],
    max_ratio: float = 1.20,
) -> tuple[float | None, CalibSource | None]:
    """多源尺度融合 + 一致性检验。

    三源差异过大（默认 > 20%）说明存在系统性偏差（典型：球机变焦后
    `W_0_px` 失效），此时返回 `(None, None)`，由上层降级为 `visual_only`
    并转人工 —— 宁可不给厚度，也不给错厚度。
    """
    if not refs:
        return None, None

    ordered = sorted(refs, key=lambda r: _SOURCE_PRIORITY[r.source])
    alphas = [r.alpha for r in ordered]

    if len(alphas) > 1 and max(alphas) / min(alphas) > max_ratio:
        return None, None

    return ordered[0].alpha, ordered[0].source


def radial_thickness_mm(
    w_ice_px: float,
    w0_px: float,
    alpha_mm_per_px: float,
) -> float:
    """单侧径向覆冰厚度（mm）。

    d_radial = 0.5 · (W_ice_px − W_0_px) · α
    """
    if alpha_mm_per_px <= 0:
        raise ValueError("alpha_mm_per_px 必须为正")
    if w_ice_px <= w0_px:
        return 0.0
    return 0.5 * (w_ice_px - w0_px) * alpha_mm_per_px


def equivalent_thickness_mm(
    d_radial_mm: float,
    d0_mm: float,
    rho_ice: float = RHO_STANDARD,
    rho_std: float = RHO_STANDARD,
) -> float:
    """折算到标准密度的等值覆冰厚度（mm）。

    基于单位长度覆冰质量守恒：

        rho_ice · (r_out² − r0²) = rho_std · (r_eq² − r0²)

    解出 r_eq 后减去 r0。密度只以比值出现，故与 kg/m³ 的量纲选择无关。
    """
    if d0_mm <= 0:
        raise ValueError("d0_mm 必须为正")
    if rho_ice <= 0 or rho_std <= 0:
        raise ValueError("密度必须为正")
    if d_radial_mm <= 0:
        return 0.0

    r0 = d0_mm / 2.0
    r_out = r0 + d_radial_mm
    r_eq_sq = (rho_ice / rho_std) * (r_out**2 - r0**2) + r0**2
    return math.sqrt(r_eq_sq) - r0


def ice_mass_per_meter(
    d_radial_mm: float,
    d0_mm: float,
    rho_ice: float = RHO_STANDARD,
) -> float:
    """单位长度覆冰质量（kg/m），用于与力学/称重传感器互校。"""
    if d0_mm <= 0:
        raise ValueError("d0_mm 必须为正")
    if d_radial_mm <= 0:
        return 0.0
    r0 = d0_mm / 2.0
    r_out = r0 + d_radial_mm
    area_m2 = math.pi * (r_out**2 - r0**2) * 1e-6  # mm² → m²
    return rho_ice * area_m2


def load_ratio(d_eq_mm: float | None, design_ice_mm: float | None) -> float | None:
    """载荷比 γ = d_eq / d_design。任一缺失则为 None（不猜）。"""
    if d_eq_mm is None or design_ice_mm is None or design_ice_mm <= 0:
        return None
    return d_eq_mm / design_ice_mm


def trimmed_mean_width(widths_px: np.ndarray, k: float = 1.5) -> float:
    """IQR 截尾均值：剔除局部遮挡、孤立冰柱造成的离群截面。"""
    widths = np.asarray(widths_px, dtype=float)
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        raise ValueError("无有效截面宽度")
    if widths.size < 4:
        return float(widths.mean())
    q1, q3 = np.percentile(widths, [25, 75])
    iqr = float(q3 - q1)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    kept = widths[(widths >= lo) & (widths <= hi)]
    return float(kept.mean() if kept.size else widths.mean())


def thickness_uncertainty_mm(
    w_ice_px: float,
    alpha_mm_per_px: float,
    sigma_w_px: float = 0.5,
    sigma_alpha_rel: float = 0.05,
) -> float:
    """厚度不确定度（一阶误差传播）。

    σ_d = 0.5 · sqrt( (σ_w·α)² + (W_ice·α·σ_α_rel)² )
    """
    if w_ice_px <= 0:
        return 0.0
    term_w = sigma_w_px * alpha_mm_per_px
    term_a = w_ice_px * alpha_mm_per_px * sigma_alpha_rel
    return 0.5 * math.sqrt(term_w**2 + term_a**2)


@dataclass(frozen=True)
class ThicknessEstimate:
    """一次厚度解算的完整结果，含中间量，便于审计与回放。"""

    w_ice_px: float
    w0_px: float
    alpha_mm_per_px: float
    d_radial_mm: float
    d_eq_mm: float
    rho_ice: float
    uncertainty_mm: float
    gamma: float | None = None


def estimate_thickness(
    w_ice_px: float,
    w0_px: float,
    alpha_mm_per_px: float,
    d0_mm: float,
    rho_ice: float = RHO_STANDARD,
    design_ice_mm: float | None = None,
    sigma_w_px: float = 0.5,
    sigma_alpha_rel: float = 0.05,
) -> ThicknessEstimate:
    """串起 §10.6.1 的 ⑧⑨⑩ 三步。"""
    d_radial = radial_thickness_mm(w_ice_px, w0_px, alpha_mm_per_px)
    d_eq = equivalent_thickness_mm(d_radial, d0_mm, rho_ice)
    return ThicknessEstimate(
        w_ice_px=w_ice_px,
        w0_px=w0_px,
        alpha_mm_per_px=alpha_mm_per_px,
        d_radial_mm=d_radial,
        d_eq_mm=d_eq,
        rho_ice=rho_ice,
        uncertainty_mm=thickness_uncertainty_mm(
            w_ice_px, alpha_mm_per_px, sigma_w_px, sigma_alpha_rel
        ),
        gamma=load_ratio(d_eq, design_ice_mm),
    )
