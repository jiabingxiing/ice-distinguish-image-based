"""几何量算的性能与退化保护门禁（计划 §7.2 CPU P95 ≤ 180 ms）。

背景（实测，2026-11-25）
------------------------
`measure_widths` 曾对每个截面各调一次 `map_coordinates(order=3)`，而 scipy 默认
每次都会对**整幅图**重做 `spline_filter`。400×400 竖向设备、95 个截面时
单次测量 **427 ms**，直接击穿 180 ms 端到端预算。预滤波一次后降至 **7.8 ms**
（55×），且宽度数值逐位一致。

同时，碎片化掩码（如随机权重的分割输出）会生成 8.8×10⁴ px 的骨架，
BFS 退化到 **~1 s**。因此加了 `MAX_SKEL_PX` 上限，超限抛 `ValueError`
由上层降级为 `visual_only`。

本文件把这些结论固化为回归门禁：**任何人重新引入逐截面 spline 滤波，
或移除退化保护，都会在这里失败。**
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from grid_ice_vision.features.severity.geometry import (
    MAX_SKEL_PX,
    measure_widths,
    skeleton_polyline,
)


def rod(h: int = 400, w: int = 400, half: int = 12) -> np.ndarray:
    """竖向柱体掩码，模拟绝缘子串/导线。"""
    m = np.zeros((h, w), np.uint8)
    cx = w // 2
    m[:, cx - half : cx + half] = 1
    return m


# ---------------------------------------------------------------- 性能门禁


@pytest.mark.parametrize(
    ("h", "w", "half", "budget_ms"),
    [
        (400, 400, 12, 120.0),
        (512, 512, 10, 160.0),
        (480, 640, 14, 170.0),
    ],
)
def test_measure_widths_within_budget(h: int, w: int, half: int, budget_ms: float) -> None:
    """真实形态设备必须在预算内完成（预滤波优化后的实测 38–97 ms）。"""
    m = rod(h, w, half)
    measure_widths(m, spacing=4.0)  # 预热
    t0 = time.perf_counter()
    measure_widths(m, spacing=4.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < budget_ms, (
        f"{h}x{w} 宽度测量耗时 {elapsed_ms:.1f} ms 超出预算 {budget_ms} ms —— "
        "检查是否重新引入了逐截面 spline_filter"
    )


def test_measure_widths_width_is_exact_on_clean_rod() -> None:
    """性能优化不得改变数值：干净柱体的宽度应精确等于几何宽度。"""
    m = rod(400, 400, 12)
    wm = measure_widths(m, spacing=4.0)
    assert wm.width_px == pytest.approx(24.0, abs=0.5)


def test_measure_widths_valid_ratio_high_on_clean_rod() -> None:
    m = rod(400, 400, 12)
    assert measure_widths(m, spacing=4.0).valid_ratio > 0.9


# ---------------------------------------------------------------- 退化保护


def test_fragmented_mask_raises_fast() -> None:
    """碎片化掩码必须快速失败，而不是卡住 1 s。"""
    rng = np.random.default_rng(1)
    m = (rng.integers(0, 4, (400, 400)) >= 1).astype(np.uint8)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="碎片化"):
        measure_widths(m, spacing=4.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, f"退化输入耗时 {elapsed_ms:.1f} ms，保护未生效"


def test_skeleton_limit_is_exported() -> None:
    """上限常量必须可被配置层与测试引用。"""
    assert isinstance(MAX_SKEL_PX, int)
    assert 5_000 <= MAX_SKEL_PX <= 50_000


def test_skeleton_polyline_accepts_long_thin_line() -> None:
    """上限不能误伤长导线：3,000 px 的细长线段应正常通过。"""
    m = np.zeros((200, 3000), np.uint8)
    m[98:102, :] = 1
    pts = skeleton_polyline(m.astype(bool))
    assert len(pts) > 100
    assert len(pts) <= MAX_SKEL_PX


def test_tiny_mask_still_raises_min_pixels() -> None:
    with pytest.raises(ValueError, match="过少"):
        skeleton_polyline(np.zeros((10, 10), bool))
