"""预警规则引擎（§10.7.3）。

三条硬约束：
1. `measurement_status != measured` 时**一律**返回 None —— 不给没标定的厚度定级；
2. 升降级都要连续帧确认（升 3 帧、降 5 帧），避免告警闪烁；
3. 阈值来自 `configs/rules/alarm.yaml`，由业务方按本地区规程校准，代码里不写死。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grid_ice_vision.common.constants import ALARM_LEVELS
from grid_ice_vision.common.schemas import AlarmLevel, MeasurementStatus


@dataclass(frozen=True)
class AlarmRules:
    version: str = "1.0.0"
    require_measured: bool = True
    escalate_frames: int = 3
    deescalate_frames: int = 5
    # 每级三个判据的区间（满足任一即命中该级）
    d_eq_ranges: dict[str, tuple[float, float | None]] = field(default_factory=dict)
    gamma_ranges: dict[str, tuple[float, float | None]] = field(default_factory=dict)
    bridge_ranges: dict[str, tuple[float, float | None]] = field(default_factory=dict)
    review_levels: tuple[str, ...] = ("III", "IV")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> AlarmRules:
        if not cfg:
            return cls()
        levels = cfg.get("levels", {}) or {}
        d_eq: dict[str, tuple[float, float | None]] = {}
        gamma: dict[str, tuple[float, float | None]] = {}
        bridge: dict[str, tuple[float, float | None]] = {}
        for name, spec in levels.items():
            for key, target in (("d_eq_mm", d_eq), ("gamma", gamma), ("bridge", bridge)):
                rng = (spec or {}).get(key)
                if isinstance(rng, dict):
                    target[str(name)] = (float(rng.get("min", 0.0)), rng.get("max"))
        deb = cfg.get("debounce", {}) or {}
        return cls(
            version=str(cfg.get("version", "1.0.0")),
            require_measured=bool(cfg.get("require_measured", True)),
            escalate_frames=int(deb.get("escalate_frames", 3)),
            deescalate_frames=int(deb.get("deescalate_frames", 5)),
            d_eq_ranges=d_eq,
            gamma_ranges=gamma,
            bridge_ranges=bridge,
        )


def _in_range(value: float | None, rng: tuple[float, float | None]) -> bool:
    if value is None:
        return False
    lo, hi = rng
    if value < lo:
        return False
    return hi is None or value < hi


def classify_level(
    rules: AlarmRules,
    d_eq_mm: float | None,
    gamma: float | None = None,
    bridge: float | None = None,
    measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
) -> AlarmLevel | None:
    """按「满足任一判据」定级，取最高命中等级。"""
    if rules.require_measured and measurement_status is not MeasurementStatus.MEASURED:
        return None
    if d_eq_mm is None:
        return None

    hit: AlarmLevel | None = None
    for name in ALARM_LEVELS:
        matched = (
            _in_range(d_eq_mm, rules.d_eq_ranges.get(name, (0.0, None)))
            or _in_range(gamma, rules.gamma_ranges.get(name, (0.0, None)))
            or _in_range(bridge, rules.bridge_ranges.get(name, (0.0, None)))
        )
        if matched:
            hit = AlarmLevel(name)
    return hit


def _rank(level: AlarmLevel | None) -> int:
    return -1 if level is None else ALARM_LEVELS.index(level.value)


@dataclass
class _TrackState:
    level: AlarmLevel | None = None
    pending: AlarmLevel | None = None
    count: int = 0


class AlarmStateMachine:
    """逐 (site, camera, roi) 维护去抖状态。"""

    def __init__(self, rules: AlarmRules | None = None) -> None:
        self.rules = rules or AlarmRules()
        self._tracks: dict[str, _TrackState] = {}

    def update(
        self,
        key: str,
        candidate: AlarmLevel | None,
        measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
    ) -> AlarmLevel | None:
        if self.rules.require_measured and measurement_status is not MeasurementStatus.MEASURED:
            self._tracks.pop(key, None)
            return None
        if candidate is None:
            return self._tracks.get(key, _TrackState()).level

        st = self._tracks.setdefault(key, _TrackState(level=candidate, pending=candidate, count=0))
        if st.level is None:
            st.level = candidate
            st.pending = candidate
            st.count = 0
            return st.level

        if candidate is st.level:
            st.pending = candidate
            st.count = 0
            return st.level

        if st.pending is not candidate:
            st.pending = candidate
            st.count = 0
        st.count += 1

        need = (
            self.rules.escalate_frames
            if _rank(candidate) > _rank(st.level)
            else self.rules.deescalate_frames
        )
        if st.count >= need:
            st.level = candidate
            st.count = 0
        return st.level

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._tracks.clear()
        else:
            self._tracks.pop(key, None)


def requires_review(level: AlarmLevel | None, rules: AlarmRules) -> bool:
    return level is not None and level.value in rules.review_levels
