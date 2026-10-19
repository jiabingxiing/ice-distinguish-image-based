"""物理量算单测（§13.3、§15.1）。

这四条是**必须在没有任何真实数据时就能通过**的：恒等、零、单调、密度。
它们保证厚度公式的物理正确性，而不是拟合出来的数字。
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_ice_vision.common.constants import RHO_STANDARD
from grid_ice_vision.features.severity.thickness import (
    ScaleReference,
    equivalent_thickness_mm,
    estimate_thickness,
    fuse_scale_references,
    ice_mass_per_meter,
    load_ratio,
    radial_thickness_mm,
    thickness_uncertainty_mm,
    trimmed_mean_width,
)


def test_equivalent_thickness_identity() -> None:
    """密度等于标准密度时，等值厚度应等于径向厚度。"""
    assert equivalent_thickness_mm(6.0, 30.0, 900.0, 900.0) == pytest.approx(6.0)


def test_equivalent_thickness_zero() -> None:
    assert equivalent_thickness_mm(0.0, 30.0) == 0.0
    assert radial_thickness_mm(20.0, 20.0, 0.1) == 0.0
    assert radial_thickness_mm(18.0, 20.0, 0.1) == 0.0


def test_equivalent_thickness_monotonic() -> None:
    vals = [equivalent_thickness_mm(d, 30.0, 600.0) for d in (1, 2, 4, 8)]
    assert vals == sorted(vals)


def test_low_density_gives_smaller_equivalent() -> None:
    """雾凇密度低，折算到 900 后应更薄。"""
    assert equivalent_thickness_mm(10.0, 30.0, 400.0) < equivalent_thickness_mm(10.0, 30.0, 900.0)


def test_mass_roundtrip() -> None:
    """质量守恒：任意密度算出的质量 = 等值厚度 + 标准密度算出的质量。"""
    d, d0, rho = 10.0, 30.0, 600.0
    m = ice_mass_per_meter(d, d0, rho)
    d_eq = equivalent_thickness_mm(d, d0, rho)
    assert ice_mass_per_meter(d_eq, d0, RHO_STANDARD) == pytest.approx(m, rel=1e-9)


def test_mass_formula_against_hand_calculation() -> None:
    """r0=15mm, d=5mm, ρ=900 → 截面积 π(20²−15²)=549.8 mm² → 0.4948 kg/m。"""
    m = ice_mass_per_meter(5.0, 30.0, 900.0)
    assert m == pytest.approx(900 * np.pi * (400 - 225) * 1e-6, rel=1e-9)


def test_radial_thickness_formula() -> None:
    # w_ice=30px, w0=20px, α=0.15mm/px → 0.5*10*0.15 = 0.75mm
    assert radial_thickness_mm(30.0, 20.0, 0.15) == pytest.approx(0.75)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        equivalent_thickness_mm(5.0, 0.0)
    with pytest.raises(ValueError):
        radial_thickness_mm(30.0, 20.0, 0.0)


def test_scale_reference_alpha() -> None:
    ref = ScaleReference(d_ref_mm=30.0, w_ref_px=200.0, source="rigid_fitting")
    assert ref.alpha == pytest.approx(0.15)


def test_fuse_prefers_rigid_fitting() -> None:
    refs = [
        ScaleReference(30.0, 200.0, "baseline_archive"),
        ScaleReference(30.0, 205.0, "rigid_fitting"),
    ]
    alpha, source = fuse_scale_references(refs)
    assert source == "rigid_fitting"
    assert alpha == pytest.approx(30.0 / 205.0)


def test_fuse_rejects_inconsistent_sources() -> None:
    """1.5× 偏差必须被拦下（球机变焦后档案失效的典型表现）。"""
    refs = [
        ScaleReference(30.0, 200.0, "rigid_fitting"),
        ScaleReference(30.0, 300.0, "baseline_archive"),
    ]
    assert fuse_scale_references(refs) == (None, None)


def test_fuse_empty() -> None:
    assert fuse_scale_references([]) == (None, None)


def test_trimmed_mean_removes_outliers() -> None:
    widths = np.array([20.0, 20.2, 19.8, 20.1, 19.9, 45.0, 3.0])
    assert trimmed_mean_width(widths) == pytest.approx(20.0, abs=0.2)


def test_trimmed_mean_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        trimmed_mean_width(np.array([np.nan, np.nan]))


def test_load_ratio() -> None:
    assert load_ratio(10.0, 20.0) == pytest.approx(0.5)
    assert load_ratio(10.0, None) is None
    assert load_ratio(None, 20.0) is None


def test_estimate_thickness_bundle() -> None:
    est = estimate_thickness(
        w_ice_px=26.0, w0_px=20.0, alpha_mm_per_px=0.15, d0_mm=3.0, design_ice_mm=10.0
    )
    assert est.d_radial_mm == pytest.approx(0.45)
    assert est.d_eq_mm == pytest.approx(0.45)
    assert est.gamma == pytest.approx(0.045)
    assert est.uncertainty_mm > 0


def test_uncertainty_monotonic_in_width() -> None:
    assert thickness_uncertainty_mm(40.0, 0.15) > thickness_uncertainty_mm(20.0, 0.15)
