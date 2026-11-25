"""绝缘子 Pa / 桥接率 / 偏冰分布单测（计划 §3.4.4）。

纯函数模块，不依赖任何权重与数据。四条不可删原则（`AGENTS.md` §5）在本文件
的映射：
- **恒等**：同一掩码作基准 → Pa 必为 0；
- **零**：无冰掩码 → Pa 反映纯设备面积，覆盖率为 0，分布为 none；
- **单调**：冰面积增加 → Pa 单调增加、覆盖率单调增加；
- **密度**：桥接率分母正确、左右/上下分布符合构造。
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_ice_vision.features.severity.insulator_pa import (
    IDX_BACKGROUND,
    IDX_BRIDGE,
    IDX_EXPOSED,
    IDX_ICE,
    SCUT_H_MAP,
    SCUT_V_MAP,
    InsulatorPaConfig,
    area_change_rate,
    compute_insulator_pa,
    horizontal_skew,
    vertical_spread,
)


def make_mask(
    h: int = 64,
    w: int = 48,
    band: tuple[int, int] | None = None,
    kind: str = "ice",
    base_width: int = 0,
) -> np.ndarray:
    """构造**竖向绝缘子**掩码（严谨版）。

    参数
    ----
    band : (y0, y1) 覆冰纵向范围；None 表示不覆冰。
    kind : `ice`（附冰，类别 2）或 `bridge`（桥接冰凌，类别 3）。
    base_width : 无冰期设备半宽。为 0 时设备占满整幅宽度，此时**设备无法变宽**，
                 Pa 恒为 0，不能用来测单调性；测面积增长时须给出
                 0 < base_width < w//2，由 `device_width` 控制当前帧设备宽度。

    说明：覆冰附着在设备表面，使**设备轮廓变宽**。因此「设备轮廓面积」随冰层
    增厚而增加，这正是 `Pa` 的物理含义。
    """
    m = np.full((h, w), IDX_BACKGROUND, dtype=np.uint8)
    cx = w // 2
    # 无冰基准：以中线为中心、半宽 base_width 的柱体
    if base_width > 0:
        m[:, cx - base_width : cx + base_width] = IDX_EXPOSED
    else:
        m[:, :] = IDX_EXPOSED
    if band is None:
        return m
    y0, y1 = band
    val = IDX_ICE if kind == "ice" else IDX_BRIDGE
    m[y0:y1, :] = np.where(m[y0:y1, :] == IDX_EXPOSED, val, val)
    return m


def growing_mask(
    h: int = 64, w: int = 48, ice_thickness: int = 4, base_width: int = 6
) -> np.ndarray:
    """模拟「冰层增厚导致设备轮廓变宽」：轮廓半宽 = base_width + ice_thickness。"""
    m = np.full((h, w), IDX_BACKGROUND, dtype=np.uint8)
    cx = w // 2
    half = base_width + ice_thickness
    m[:, cx - half : cx + half] = IDX_ICE
    return m


def clean_mask(h: int = 64, w: int = 48, base_width: int = 6) -> np.ndarray:
    """无冰基准：半宽 base_width 的裸露设备柱体。"""
    m = np.full((h, w), IDX_BACKGROUND, dtype=np.uint8)
    cx = w // 2
    m[:, cx - base_width : cx + base_width] = IDX_EXPOSED
    return m


# ---------------------------------------------------------------- 恒等


def test_pa_identity_is_zero() -> None:
    """同一掩码作基准，Pa 必须恰为 0。"""
    m = growing_mask(ice_thickness=6)
    r = compute_insulator_pa(m, baseline_mask=m)
    assert r.pa_pct == pytest.approx(0.0)


def test_area_change_rate_identity() -> None:
    assert area_change_rate(500, 500) == pytest.approx(0.0)


# ---------------------------------------------------------------- 零


def test_no_ice_mask() -> None:
    """全裸露设备：覆盖率 0、桥接率 0、分布 none、无基准时无 Pa。"""
    m = clean_mask()
    r = compute_insulator_pa(m)
    assert r.coverage == pytest.approx(0.0)
    assert r.bridge_ratio == pytest.approx(0.0)
    assert r.vertical == "none"
    assert r.horizontal == "none"
    assert r.pa_pct is None
    assert r.area_ice_px == 0


def test_zero_baseline_returns_none() -> None:
    """基准面积为 0 时无法定义相对变化，必须返回 None 而非 inf。"""
    assert area_change_rate(100, 0) is None
    assert area_change_rate(0, 0) is None


def test_negative_pa_when_ice_melted() -> None:
    """融冰：当前轮廓面积小于基准 → Pa 为负。"""
    base = growing_mask(ice_thickness=8)
    cur = growing_mask(ice_thickness=2)
    r = compute_insulator_pa(cur, baseline_mask=base)
    assert r.pa_pct is not None
    assert r.pa_pct < 0
    # (10−4)/... 半宽 6+2=8 vs 6+8=14 → 面积比 8/14
    assert r.pa_pct == pytest.approx((8 / 14 - 1.0) * 100.0, rel=1e-6)


# ---------------------------------------------------------------- 单调


def test_pa_monotonic_in_ice_area() -> None:
    """冰层增厚 → 设备轮廓变宽 → Pa 严格递增。"""
    base = clean_mask()
    vals = []
    for t in (1, 2, 4, 6, 8):
        cur = growing_mask(ice_thickness=t)
        r = compute_insulator_pa(cur, baseline_mask=base)
        assert r.pa_pct is not None
        vals.append(r.pa_pct)
    assert vals == sorted(vals)
    assert len(set(vals)) == len(vals)


def test_coverage_monotonic_in_ice_area() -> None:
    """设备宽度固定、冰带纵向扩展 → 覆盖率单调不减。"""
    covs = []
    for y1 in (8, 20, 32, 48, 64):
        m = make_mask(band=(0, y1), kind="ice")
        covs.append(compute_insulator_pa(m).coverage)
    assert all(c is not None for c in covs)
    assert covs == sorted(covs)


# ---------------------------------------------------------------- 密度


def test_bridge_ratio_denominator() -> None:
    """桥接率 = 桥接像素 / (裸露设备 + 附冰)，桥接像素不重复计入分母。"""
    m = np.full((64, 48), IDX_BACKGROUND, dtype=np.uint8)
    m[0:20, :] = IDX_ICE          # 20 行附冰
    m[20:40, :] = IDX_BRIDGE      # 20 行桥接
    m[40:64, :] = IDX_EXPOSED     # 24 行裸露
    r = compute_insulator_pa(m)
    expected = (20 * 48) / (20 * 48 + 24 * 48)
    assert r.bridge_ratio == pytest.approx(expected, rel=1e-9)


def test_bridge_ratio_capped_at_one() -> None:
    m = np.full((64, 48), IDX_BACKGROUND, dtype=np.uint8)
    m[0, :] = IDX_EXPOSED
    m[1:64, :] = IDX_BRIDGE
    r = compute_insulator_pa(m)
    assert r.bridge_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------- 纵向分布


def test_vertical_none_for_tiny_ice() -> None:
    """冰像素低于 min_ice_px 视为无冰（抑制单像素噪声）。"""
    m = clean_mask()
    m[0, 0] = IDX_ICE
    assert vertical_spread(m, InsulatorPaConfig(min_ice_px=8)) == "none"


def test_vertical_all_when_uniform() -> None:
    m = make_mask(band=(0, 60), kind="ice")
    assert vertical_spread(m) == "all"


def test_vertical_top() -> None:
    """上端覆冰：只有上 1/3 有冰，且与其余条带密度差足够大。"""
    m = clean_mask()
    m[0:18, :] = IDX_ICE
    assert vertical_spread(m) == "top"


def test_vertical_bottom() -> None:
    m = clean_mask()
    m[46:64, :] = IDX_ICE
    assert vertical_spread(m) == "bottom"


def test_vertical_middle() -> None:
    m = clean_mask()
    m[24:40, :] = IDX_ICE
    assert vertical_spread(m) == "middle"


def test_vertical_map_matches_scut_vocabulary() -> None:
    """SCUT `InsulatorShapeV` 五种取值必须全部有映射。"""
    assert set(SCUT_V_MAP) == {"全部覆冰", "上端覆冰", "中部覆冰", "下端覆冰", "无覆冰"}
    assert set(SCUT_V_MAP.values()) == {"all", "top", "middle", "bottom", "none"}


# ---------------------------------------------------------------- 横向分布


def test_horizontal_none_for_tiny_ice() -> None:
    m = clean_mask()
    m[0, 0] = IDX_ICE
    assert horizontal_skew(m) == "none"


def test_horizontal_uniform() -> None:
    m = make_mask(band=(0, 30), kind="ice")
    assert horizontal_skew(m) == "uniform"


def test_horizontal_left_heavy() -> None:
    m = clean_mask()
    m[:, 0:20] = IDX_ICE
    assert horizontal_skew(m) == "left_heavy"


def test_horizontal_right_heavy() -> None:
    m = clean_mask()
    m[:, 28:] = IDX_ICE
    assert horizontal_skew(m) == "right_heavy"


def test_horizontal_map_matches_scut_vocabulary() -> None:
    assert set(SCUT_H_MAP) == {
        "两侧覆冰较均匀",
        "覆冰左侧重右侧轻",
        "覆冰左侧轻右侧重",
        "无覆冰",
    }
    assert set(SCUT_H_MAP.values()) == {"uniform", "left_heavy", "right_heavy", "none"}


# ---------------------------------------------------------------- 可靠性门限


def test_reliable_flag_uses_min_asset_px() -> None:
    small = np.zeros((10, 10), dtype=np.uint8)
    small[:] = IDX_EXPOSED
    r = compute_insulator_pa(small, cfg=InsulatorPaConfig(min_asset_px=200))
    assert r.reliable is False
    assert r.area_asset_px == 100


def test_shape_mismatch_raises() -> None:
    a = make_mask()
    b = make_mask(h=32, w=48)
    with pytest.raises(ValueError, match="尺寸"):
        compute_insulator_pa(a, baseline_mask=b)


def test_non_2d_mask_raises() -> None:
    with pytest.raises(ValueError, match="二维"):
        compute_insulator_pa(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------- 配置


def test_config_from_yaml_section() -> None:
    cfg = {"insulator_pa": {"band_all_min": 0.5, "uniform_tol": 0.2, "min_asset_px": 50}}
    c = InsulatorPaConfig.from_config(cfg)
    assert c.band_all_min == 0.5
    assert c.uniform_tol == 0.2
    assert c.min_asset_px == 50


def test_config_defaults_when_absent() -> None:
    assert InsulatorPaConfig.from_config(None) == InsulatorPaConfig()
    assert InsulatorPaConfig.from_config({}) == InsulatorPaConfig()