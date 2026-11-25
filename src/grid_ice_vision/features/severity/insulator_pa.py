"""绝缘子面积变化率 Pa、桥接率与横/纵向偏冰分布（计划 §3.4.4 第 1 条）。

纯函数模块：不读写文件、不解析配置、不打印日志、不 import 模型代码，
因此可在无权重、无数据时完整单测。

对齐 SCUT 标注体系
------------------
`InsulatorShapeV`（纵向）取值：`全部覆冰` / `上端覆冰` / `中部覆冰` / `下端覆冰` / `无覆冰`
`InsulatorShapeH`（横向）取值：`两侧覆冰较均匀` / `覆冰左侧重右侧轻` / `覆冰左侧轻右侧重` / `无覆冰`

本模块输出的 `VerticalSpread` / `HorizontalSkew` 与上述中文标签一一对应，
便于直接用 SCUT 的 7,296 条标签做离线校验（见 `scripts/eval_insulator_pa.py`）。

为什么需要 Pa
-------------
技术报告证明绝缘子在低清图（占 93.2%）下比导线更可靠：导线仅 5–12 px 宽，
骨架法向测量易断裂；绝缘子串尺寸大、伞裙间隙清晰。因此 `Pa` 是 Track A 的
主指标，`d_eq`（Track B）在缺标定时必须为 null。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# 掩码类别索引，与 `common/constants.py::SEG_CLASSES` 保持一致
IDX_BACKGROUND = 0
IDX_EXPOSED = 1
IDX_ICE = 2
IDX_BRIDGE = 3

VerticalSpread = Literal["all", "top", "middle", "bottom", "none"]
HorizontalSkew = Literal["uniform", "left_heavy", "right_heavy", "none"]

SCUT_V_MAP: dict[str, VerticalSpread] = {
    "全部覆冰": "all",
    "上端覆冰": "top",
    "中部覆冰": "middle",
    "下端覆冰": "bottom",
    "无覆冰": "none",
}
"""SCUT `InsulatorShapeV` 中文标签 → 本模块枚举。"""

SCUT_H_MAP: dict[str, HorizontalSkew] = {
    "两侧覆冰较均匀": "uniform",
    "覆冰左侧重右侧轻": "left_heavy",
    "覆冰左侧轻右侧重": "right_heavy",
    "无覆冰": "none",
}
"""SCUT `InsulatorShapeH` 中文标签 → 本模块枚举。"""


@dataclass(frozen=True)
class InsulatorPaConfig:
    """阈值。默认值与 `configs/labels/severity.yaml` 的 `insulator_pa` 段一致。"""

    #: 三个纵向条带都达到该冰密度时判为「全部覆冰」
    band_all_min: float = 0.60
    #: 条带间最大密度差低于该值时，视为均匀分布（判「全部覆冰」而非 argmax）
    band_contrast_min: float = 0.25
    #: 横向左右冰量占比差值低于该值时判「均匀」
    uniform_tol: float = 0.15
    #: 少于该像素数的冰视为无冰（抑制单像素噪声）
    min_ice_px: int = 8
    #: Pa 的置信门限：有效设备像素少于该值时不出 Pa
    min_asset_px: int = 200

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> InsulatorPaConfig:
        sec = (cfg or {}).get("insulator_pa", {}) or {}
        return cls(
            band_all_min=float(sec.get("band_all_min", 0.60)),
            band_contrast_min=float(sec.get("band_contrast_min", 0.25)),
            uniform_tol=float(sec.get("uniform_tol", 0.15)),
            min_ice_px=int(sec.get("min_ice_px", 8)),
            min_asset_px=int(sec.get("min_asset_px", 200)),
        )


@dataclass(frozen=True)
class InsulatorPaResult:
    """一次 Pa 解算的完整结果，含全部中间量，便于审计与回放。"""

    pa_pct: float | None
    """面积变化率 Pa = (A_ice − A_nor) / A_nor，以百分数表示；无基准时为 None。"""

    coverage: float | None
    """有效覆冰覆盖率 = (N_ice + N_bridge) / (N_exposed + N_ice + N_bridge)。"""

    bridge_ratio: float | None
    """伞裙间隙桥接率 η_bridge；设备像素不足时为 None。"""

    vertical: VerticalSpread
    horizontal: HorizontalSkew

    area_ice_px: int
    area_bridge_px: int
    area_exposed_px: int
    area_asset_px: int
    """当前帧有效设备像素 = exposed + ice + bridge。"""

    baseline_asset_px: int | None
    """基准帧有效设备像素；无基准时为 None。"""

    reliable: bool
    """有效设备像素是否达到 `min_asset_px`。False 时调用方应置 requires_review。"""


def _object_mask(mask: np.ndarray) -> np.ndarray:
    """设备轮廓 = 裸露设备 + 附冰 + 桥接冰（不含背景）。

    这是**轮廓面积**口径：覆冰附着在设备表面，轮廓随冰层增厚而变大。
    `Pa` 比较的就是这个轮廓面积相对无冰基准的增长。
    """
    m = np.asarray(mask)
    return (m >= IDX_EXPOSED) & (m <= IDX_BRIDGE)


def _ice_mask(mask: np.ndarray) -> np.ndarray:
    """全部冰像素 = 附冰 + 桥接冰（覆盖率分子用）。"""
    m = np.asarray(mask)
    return (m == IDX_ICE) | (m == IDX_BRIDGE)


def _accretion_mask(mask: np.ndarray) -> np.ndarray:
    """仅附冰（不含桥接冰）。桥接率分母用，避免桥接像素被重复计数。"""
    return np.asarray(mask) == IDX_ICE


def area_change_rate(area_ice_px: int, area_baseline_px: int) -> float | None:
    """面积变化率 Pa。

    定义与计划 §3.4.4 一致：``Pa = (A_ice − A_nor) / A_nor``。
    基准面积为 0 时返回 None（无法定义相对变化，不猜）。

    返回**小数**（非百分数）；调用方按需 ×100。
    """
    if area_baseline_px <= 0:
        return None
    return (float(area_ice_px) - float(area_baseline_px)) / float(area_baseline_px)


def _row_bands(obj: np.ndarray, n: int) -> list[slice]:
    """按设备本体的行范围等分为 n 个横向条带（自上而下）。"""
    rows = np.where(obj.any(axis=1))[0]
    if rows.size == 0:
        return []
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    if bottom - top < n:
        # 设备过薄：退化为单条带，避免产生空条带
        return [slice(top, bottom)]
    edges = np.linspace(top, bottom, n + 1).round().astype(int)
    return [slice(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(n)]


def vertical_spread(mask: np.ndarray, cfg: InsulatorPaConfig | None = None) -> VerticalSpread:
    """纵向偏冰分布：`all` / `top` / `middle` / `bottom` / `none`。

    判据（按顺序）：
    1. 冰像素总数 < `min_ice_px` → `none`；
    2. 每个条带的冰密度都 ≥ `band_all_min` → `all`；
    3. 条带间最大密度差 < `band_contrast_min` → `all`（均匀薄冰，不是局部）；
    4. 否则取冰密度最高的条带。
    """
    c = cfg or InsulatorPaConfig()
    m = np.asarray(mask)
    obj = _object_mask(m)
    ice = _ice_mask(m)

    if int(ice.sum()) < c.min_ice_px:
        return "none"

    bands = _row_bands(obj, 3)
    if not bands:
        return "none"
    if len(bands) == 1:
        return "all"

    density: list[float] = []
    for b in bands:
        obj_b = int(obj[b].sum())
        density.append(0.0 if obj_b == 0 else float(ice[b].sum()) / float(obj_b))

    if min(density) >= c.band_all_min:
        return "all"
    if max(density) - min(density) < c.band_contrast_min:
        return "all"

    names: list[VerticalSpread] = ["top", "middle", "bottom"]
    return names[int(np.argmax(density))]


def horizontal_skew(mask: np.ndarray, cfg: InsulatorPaConfig | None = None) -> HorizontalSkew:
    """横向偏冰分布：`uniform` / `left_heavy` / `right_heavy` / `none`。

    以设备本体水平中位线为界分为左右两半，比较两侧冰像素占比。
    占比差 |right−left| / (left+right) ≤ `uniform_tol` 判均匀。
    """
    c = cfg or InsulatorPaConfig()
    m = np.asarray(mask)
    obj = _object_mask(m)
    ice = _ice_mask(m)

    if int(ice.sum()) < c.min_ice_px:
        return "none"

    cols = np.where(obj.any(axis=0))[0]
    if cols.size < 2:
        return "uniform"
    mid = round((int(cols[0]) + int(cols[-1]) + 1) / 2.0)

    left = int(ice[:, :mid].sum())
    right = int(ice[:, mid:].sum())
    total = left + right
    if total <= 0:
        return "none"

    imbalance = (right - left) / total
    if abs(imbalance) <= c.uniform_tol:
        return "uniform"
    return "right_heavy" if imbalance > 0 else "left_heavy"


def compute_insulator_pa(
    mask: np.ndarray,
    baseline_mask: np.ndarray | None = None,
    cfg: InsulatorPaConfig | None = None,
) -> InsulatorPaResult:
    """计算绝缘子 Track A 全套指标。

    `mask` 与 `baseline_mask` 均为 0..3 类别掩码（同 ROI 尺寸）。
    基准掩码应来自**同 monitorId、同机位、无冰期**的历史帧；比对前须已由
    `application/baseline_matcher.py` 完成配准与尺寸归一。
    """
    c = cfg or InsulatorPaConfig()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"掩码必须是二维，得到 shape={m.shape}")

    ice = _ice_mask(m)
    exposed = np.asarray(m) == IDX_EXPOSED
    bridge = np.asarray(m) == IDX_BRIDGE
    obj = _object_mask(m)

    n_ice = int(ice.sum())
    n_accretion = int(_accretion_mask(m).sum())
    n_bridge = int(bridge.sum())
    n_exposed = int(exposed.sum())
    # A_ice 与基准 A_nor 必须同口径：都取**设备轮廓像素**
    # （exposed + ice + bridge），否则有桥接冰时两者不可比、Pa 失真。
    n_object = int(obj.sum())

    coverage: float | None = (
        float(n_ice + n_bridge) / float(n_object) if n_object > 0 else None
    )
    """有效覆冰覆盖率 = (N_ice + N_bridge) / N_object（计划 §3.4.4 的定义）。"""

    bridge_ratio: float | None = None
    # 分母用「裸露设备 + 附冰」，不含桥接冰本身 —— 否则桥接像素被重复计入。
    gap = n_exposed + n_accretion
    if gap > 0:
        bridge_ratio = float(min(n_bridge / gap, 1.0))

    base_asset: int | None = None
    pa: float | None = None
    if baseline_mask is not None:
        bm = np.asarray(baseline_mask)
        if bm.shape != m.shape:
            raise ValueError(f"基准掩码尺寸 {bm.shape} 与当前掩码 {m.shape} 不一致")
        base_asset = int(_object_mask(bm).sum())
        pa = area_change_rate(n_object, base_asset)

    return InsulatorPaResult(
        pa_pct=None if pa is None else pa * 100.0,
        coverage=coverage,
        bridge_ratio=bridge_ratio,
        vertical=vertical_spread(m, c),
        horizontal=horizontal_skew(m, c),
        area_ice_px=n_ice,
        area_bridge_px=n_bridge,
        area_exposed_px=n_exposed,
        area_asset_px=n_object,
        baseline_asset_px=base_asset,
        reliable=n_object >= c.min_asset_px,
    )