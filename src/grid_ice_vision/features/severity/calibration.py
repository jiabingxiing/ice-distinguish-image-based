"""三级动态标定与一致性检验（§10.6.4）。

L1 刚性金具（耐张线夹/防振锤/钢帽）→ 高可信
L2 PTZ 焦距修正 / 双目视差      → 中高
L3 晴天建档的裸导线宽度          → 中

一致性检验是必须的：球机变焦后 W_0_px 失效，若不检验会得到 2–3 倍的
系统性厚度偏差。宁可降级为 visual_only，也不输出错厚度。
"""

from __future__ import annotations

from dataclasses import dataclass

from grid_ice_vision.common.schemas import CalibInput, MeasurementStatus
from grid_ice_vision.features.severity.thickness import (
    CalibSource,
    ScaleReference,
    fuse_scale_references,
)


@dataclass(frozen=True)
class CalibrationOutcome:
    alpha_mm_per_px: float | None
    source: CalibSource | None
    status: MeasurementStatus
    requires_review: bool
    reason: str

    @property
    def measurable(self) -> bool:
        return self.alpha_mm_per_px is not None


def ptz_corrected_alpha(alpha_ref: float, zoom_ref: float, zoom_now: float) -> float:
    """PTZ 变倍修正：α_t = α_0 · (f_0 / f_t)。

    像素尺度与焦距成反比；焦距与 zoom 读数近似成正比（同一镜头内）。
    """
    if zoom_now <= 0 or zoom_ref <= 0:
        raise ValueError("zoom 必须为正")
    return alpha_ref * (zoom_ref / zoom_now)


def build_references(
    calib: CalibInput,
    *,
    ptz_zoom_now: float | None = None,
    ptz_zoom_ref: float | None = None,
) -> list[ScaleReference]:
    """把标定输入翻译成尺度参考列表。缺失的级别自动跳过。"""
    refs: list[ScaleReference] = []

    # L1 刚性金具
    if calib.d_ref_mm is not None and calib.w_ref_px is not None:
        refs.append(
            ScaleReference(d_ref_mm=calib.d_ref_mm, w_ref_px=calib.w_ref_px, source="rigid_fitting")
        )

    # L2 外部给定 α（双目/PTZ 解算结果），有 zoom 读数则做变倍修正
    if calib.alpha_mm_per_px is not None:
        alpha = calib.alpha_mm_per_px
        if ptz_zoom_now is not None and ptz_zoom_ref is not None:
            alpha = ptz_corrected_alpha(alpha, ptz_zoom_ref, ptz_zoom_now)
        refs.append(ScaleReference(d_ref_mm=alpha, w_ref_px=1.0, source="external"))

    # L3 晴天建档：用导线外径与裸导线像素宽度反算
    if calib.d0_mm is not None and calib.w0_px is not None:
        refs.append(
            ScaleReference(d_ref_mm=calib.d0_mm, w_ref_px=calib.w0_px, source="baseline_archive")
        )

    return refs


def resolve_calibration(
    calib: CalibInput,
    *,
    ptz_zoom_now: float | None = None,
    ptz_zoom_ref: float | None = None,
    max_ratio: float = 1.20,
) -> CalibrationOutcome:
    """多源融合 → (α, 可信度三态)。"""
    refs = build_references(calib, ptz_zoom_now=ptz_zoom_now, ptz_zoom_ref=ptz_zoom_ref)
    if not refs:
        return CalibrationOutcome(
            alpha_mm_per_px=None,
            source=None,
            status=MeasurementStatus.NOT_CALIBRATED,
            requires_review=False,
            reason="无任何尺度参考",
        )

    alpha, source = fuse_scale_references(refs, max_ratio=max_ratio)
    if alpha is None or source is None:
        alphas = [r.alpha for r in refs]
        spread = max(alphas) / min(alphas)
        return CalibrationOutcome(
            alpha_mm_per_px=None,
            source=None,
            status=MeasurementStatus.VISUAL_ONLY,
            requires_review=True,
            reason=f"多源尺度不一致：α 极差 {spread:.2f}× > {max_ratio:.2f}×，疑似变焦后档案失效",
        )

    if source == "baseline_archive":
        return CalibrationOutcome(
            alpha_mm_per_px=alpha,
            source=source,
            status=MeasurementStatus.VISUAL_ONLY,
            requires_review=False,
            reason="仅晴天建档，无实时刚性参考；厚度仅供参考",
        )

    return CalibrationOutcome(
        alpha_mm_per_px=alpha,
        source=source,
        status=MeasurementStatus.MEASURED,
        requires_review=False,
        reason=f"标定来源 {source}",
    )
