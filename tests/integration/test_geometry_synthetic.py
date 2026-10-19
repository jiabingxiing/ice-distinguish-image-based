"""合成圆筒真值上的几何精度测试（P3.4 DoD）。

硬门槛：像素宽度误差 ≤ 0.8 px；软目标 ≤ 0.5 px。
这条测试是「无现场数据也能验证物理链路」的核心证据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from grid_ice_vision.features.severity.geometry import measure_widths
from grid_ice_vision.features.severity.thickness import (
    equivalent_thickness_mm,
    radial_thickness_mm,
)
from synthesize_ice import IceShapeSpec, expand_cross_section, make_synthetic_scene


@pytest.mark.parametrize("t", [3.0, 6.0, 10.0, 14.0])
def test_cylinder_width_accuracy(t: float) -> None:
    _img, mask, _truth = make_synthetic_scene(360, 720, t, "cylinder", seed=int(t))
    bare = (mask == 1).astype(np.uint8)
    iced = ((mask == 1) | (mask == 2)).astype(np.uint8)
    w0 = measure_widths(bare, spacing=4.0).width_px
    w_ice = measure_widths(iced, spacing=4.0).width_px
    assert abs(w_ice - (w0 + 2 * t)) <= 0.5


@pytest.mark.parametrize("shape", ["c2", "c7", "c11"])
def test_crescent_shapes_measurable(shape: str) -> None:
    _img, mask, _truth = make_synthetic_scene(360, 720, 6.0, shape, seed=7)
    iced = ((mask == 1) | (mask == 2)).astype(np.uint8)
    wm = measure_widths(iced, spacing=4.0)
    assert wm.width_px > 0
    assert wm.valid_ratio > 0.5


def test_thickness_recovers_ground_truth() -> None:
    """给定 α 与裸导线宽度，反算厚度应等于注入的 t（误差 < 0.1 mm）。"""
    t = 8.0
    _img, mask, _truth = make_synthetic_scene(360, 720, t, "cylinder", seed=11)
    bare = (mask == 1).astype(np.uint8)
    iced = ((mask == 1) | (mask == 2)).astype(np.uint8)
    w0 = measure_widths(bare, spacing=4.0).width_px
    w_ice = measure_widths(iced, spacing=4.0).width_px
    alpha = 30.0 / w0
    d_radial = radial_thickness_mm(w_ice, w0, alpha)
    # 以像素为单位断言：厚度误差 = 宽度误差/2，门槛 0.3 px
    assert abs(d_radial / alpha - t) <= 0.3
    assert equivalent_thickness_mm(d_radial, 30.0, 900.0) == pytest.approx(d_radial, abs=1e-9)


def test_expand_cross_section_zero_thickness() -> None:
    """t=0 时扩出的区域应与原掩码基本重合（允许 1 px 栅格误差）。"""
    m = np.zeros((80, 200), dtype=np.uint8)
    m[34:46, 20:180] = 1
    out = expand_cross_section(m, IceShapeSpec(thickness_px=0.0))
    inter = float((out & m).sum())
    union = float((out | m).sum())
    assert inter / union > 0.85


def test_expand_cross_section_rejects_unknown_shape() -> None:
    m = np.zeros((40, 80), dtype=np.uint8)
    m[18:22, 10:70] = 1
    with pytest.raises(ValueError):
        expand_cross_section(m, IceShapeSpec(shape="hexagon"))
