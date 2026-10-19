"""几何量算单测：骨架、法向、亚像素宽度（§10.6.3、§15.1）。"""

from __future__ import annotations

import numpy as np
import pytest

from grid_ice_vision.features.severity.geometry import (
    bridge_ratio,
    coverage_ratio,
    curvature,
    edge_probability,
    measure_widths,
    resample_polyline,
    skeleton_polyline,
    tangents_and_normals,
)


def _bar(
    h: int = 160, w: int = 320, top: int = 70, bottom: int = 90, x0: int = 20, x1: int = 300
) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[top:bottom, x0:x1] = 1
    return m


def test_horizontal_bar_width() -> None:
    m = _bar(bottom=90)  # 20 px 宽
    wm = measure_widths(m, spacing=4.0)
    assert wm.width_px == pytest.approx(20.0, abs=0.3)
    assert wm.valid_ratio > 0.8
    assert wm.n_sections > 30


def test_vertical_bar_width() -> None:
    m = _bar().T.copy()
    wm = measure_widths(m, spacing=4.0)
    assert wm.width_px == pytest.approx(20.0, abs=0.3)


def test_width_scales_linearly() -> None:
    widths = []
    for h in (10, 20, 40):
        m = _bar(top=80 - h // 2, bottom=80 + h // 2)
        widths.append(measure_widths(m, spacing=6.0).width_px)
    assert widths[0] == pytest.approx(10.0, abs=0.4)
    assert widths[1] == pytest.approx(20.0, abs=0.4)
    assert widths[2] == pytest.approx(40.0, abs=0.6)


def test_skeleton_polyline_is_ordered() -> None:
    m = _bar()
    pts = skeleton_polyline(m)
    assert pts.shape[1] == 2
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() <= np.sqrt(2) + 1e-6


def test_resample_spacing() -> None:
    pts = np.stack([np.arange(0, 101, 1.0), np.zeros(101)], axis=1)
    out = resample_polyline(pts, spacing=10.0)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(d, 10.0, atol=1e-6)


def test_normals_are_orthogonal_to_tangents() -> None:
    pts = np.stack([np.arange(0, 50, 1.0), np.zeros(50)], axis=1)
    tau, n = tangents_and_normals(pts)
    assert np.allclose((tau * n).sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)


def test_curvature_zero_on_straight_line() -> None:
    pts = np.stack([np.arange(0, 50, 1.0), np.zeros(50)], axis=1)
    assert np.abs(curvature(pts)).max() < 1e-6


def test_edge_probability_normalized() -> None:
    e = edge_probability(_bar())
    assert e.max() == pytest.approx(1.0)
    assert e.min() >= 0.0


def test_too_small_mask_raises() -> None:
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10, 10] = 1
    with pytest.raises(ValueError):
        measure_widths(m)


def test_coverage_ratio() -> None:
    assert coverage_ratio(30, 70) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        coverage_ratio(0, 0)


def test_bridge_ratio() -> None:
    assert bridge_ratio(20, 100) == pytest.approx(0.2)
    assert bridge_ratio(200, 100) == pytest.approx(1.0)
    assert bridge_ratio(10, 0) is None
