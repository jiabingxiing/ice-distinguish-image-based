"""三级标定与一致性检验单测（§10.6.4、§15.1）。"""

from __future__ import annotations

import pytest

from grid_ice_vision.common.schemas import CalibInput, MeasurementStatus
from grid_ice_vision.features.severity.calibration import (
    build_references,
    ptz_corrected_alpha,
    resolve_calibration,
)


def test_no_reference_is_not_calibrated() -> None:
    out = resolve_calibration(CalibInput())
    assert out.status is MeasurementStatus.NOT_CALIBRATED
    assert out.alpha_mm_per_px is None
    assert not out.measurable


def test_rigid_fitting_gives_measured() -> None:
    out = resolve_calibration(CalibInput(d_ref_mm=30.0, w_ref_px=200.0))
    assert out.status is MeasurementStatus.MEASURED
    assert out.source == "rigid_fitting"
    assert out.alpha_mm_per_px == pytest.approx(0.15)


def test_baseline_archive_is_visual_only() -> None:
    out = resolve_calibration(CalibInput(d0_mm=30.0, w0_px=200.0))
    assert out.status is MeasurementStatus.VISUAL_ONLY
    assert out.source == "baseline_archive"
    assert out.requires_review is False


def test_inconsistent_sources_downgrade() -> None:
    """刚性金具 1.5× 偏差 → 必须降级并转人工，不能给厚度。"""
    out = resolve_calibration(CalibInput(d_ref_mm=30.0, w_ref_px=200.0, d0_mm=30.0, w0_px=300.0))
    assert out.status is MeasurementStatus.VISUAL_ONLY
    assert out.requires_review is True
    assert out.alpha_mm_per_px is None
    assert "不一致" in out.reason


def test_external_alpha_with_ptz_correction() -> None:
    alpha0 = 0.15
    out = resolve_calibration(
        CalibInput(alpha_mm_per_px=alpha0), ptz_zoom_now=2.0, ptz_zoom_ref=1.0
    )
    assert out.alpha_mm_per_px == pytest.approx(alpha0 * 1.0 / 2.0)
    assert out.status is MeasurementStatus.MEASURED


def test_ptz_correction_formula() -> None:
    assert ptz_corrected_alpha(0.2, 1.0, 4.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ptz_corrected_alpha(0.2, 0.0, 1.0)


def test_build_references_skips_incomplete() -> None:
    refs = build_references(CalibInput(d0_mm=30.0))  # 缺 w0_px
    assert refs == []
