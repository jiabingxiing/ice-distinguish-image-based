"""全局常量。

只放**物理与协议层**常量；业务阈值一律放 configs/，禁止在此硬编码。
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------- 物理常量
RHO_STANDARD: Final[float] = 900.0
"""标准覆冰密度 kg/m^3（电力行业等值冰厚折算基准）。"""

RHO_BY_ICE_TYPE: Final[dict[str, float]] = {
    "glaze": 900.0,
    "rime": 350.0,
    "mixed": 600.0,
    "none": 900.0,
    "unknown": 900.0,
}
"""各类覆冰的典型密度 kg/m^3（用于密度折算，区间中值）。"""

# ---------------------------------------------------------------- 协议常量
SCHEMA_VERSION: Final[str] = "1.0.0"
DEFAULT_NOTE: Final[str] = "visual_only; freezing-rain cannot be confirmed from image alone"

# 天气 8 类（顺序即标签索引，禁止随意调整）
WEATHER_LABELS: Final[tuple[str, ...]] = (
    "clear",
    "rain",
    "snow",
    "hail",
    "fog",
    "frost",
    "rime",
    "glaze_evidence",
)

INTENSITY_LEVELS: Final[tuple[str, ...]] = ("none", "light", "moderate", "heavy")

ICE_TYPES: Final[tuple[str, ...]] = ("none", "glaze", "rime", "mixed", "unknown")

VISUAL_GRADES: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3", "G4")

ALARM_LEVELS: Final[tuple[str, ...]] = ("I", "II", "III", "IV")

# 分割 4 类
SEG_CLASSES: Final[tuple[str, ...]] = (
    "background",
    "exposed_asset",
    "ice_accretion",
    "icicle_bridge",
)

ASSET_CLASSES: Final[tuple[str, ...]] = (
    "conductor",
    "insulator_string",
    "tower",
    "reference_fitting",
)
