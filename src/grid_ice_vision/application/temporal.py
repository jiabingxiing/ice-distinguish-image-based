"""时序滤波：2 状态 EKF + 物理增长率约束（§10.7.2）。

单帧在暴风雪中会被飞掠前景雪花、镜头水珠、导线风振污染，产生 5–20 mm 的
瞬时跳变；而真实覆冰生长是小时级慢变过程。用「增长率上限 + 新息门控」
即可剔除绝大部分尖峰，且不需要任何额外传感器。

变步长（M3 新增）
-----------------
南网监控的实际采样间隔为 **1–6 小时**，且并不固定：同一终端可能因通信中断、
夜间停采、PTZ 巡检而出现不规则间隔。原实现固定 `dt_minutes=15`，会把真实的
6 小时生长按 15 分钟外推，导致增长率被系统性高估 **24 倍**，或在新息门控下
被整段丢弃。

现支持两种用法：
1. `update(z)` —— 沿用固定 `dt_minutes`（向后兼容，单测与历史回放不受影响）；
2. `update(z, dt_minutes=Δ)` —— 按真实间隔动态计算状态转移与过程噪声。
   `Δ` 由调用方从相邻两帧的 `Image_Time` 求得，本模块只做数值计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: 变步长下限：低于该值视为同一时刻重复上报，避免 F=I 造成奇异
MIN_DT_MINUTES = 1.0
#: 变步长上限：超过该值视为序列断裂，调用方应先 `reset()` 再重新初始化
MAX_DT_MINUTES = 24 * 60.0


@dataclass(frozen=True)
class EKFConfig:
    dt_minutes: float = 15.0
    growth_rate_max_mm_per_h: float = 3.0
    innovation_gate_sigma: float = 3.0
    process_noise_d: float = 0.5
    process_noise_rate: float = 0.2
    measurement_noise_mm: float = 1.5
    init_variance: float = 4.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> EKFConfig:
        if not cfg:
            return cls()
        e = cfg.get("ekf", cfg)
        pn = e.get("process_noise", {}) or {}
        return cls(
            dt_minutes=float(e.get("dt_minutes", 15.0)),
            growth_rate_max_mm_per_h=float(e.get("growth_rate_max_mm_per_h", 3.0)),
            innovation_gate_sigma=float(e.get("innovation_gate_sigma", 3.0)),
            process_noise_d=float(pn.get("d", 0.5)),
            process_noise_rate=float(pn.get("rate", 0.2)),
            measurement_noise_mm=float(e.get("measurement_noise_mm", 1.5)),
            init_variance=float(e.get("init_variance", 4.0)),
        )


@dataclass
class EKFState:
    d_eq_mm: float
    growth_mm_per_h: float
    updated: bool
    outlier_rejected: bool
    rate_clamped: bool
    dt_minutes: float = 0.0
    """本步实际使用的步长；0 表示未推进（首帧或纯预测）。"""


class ThicknessEKF:
    """状态 x = [d, ḋ]ᵀ，观测 z = d_eq_measured。"""

    def __init__(self, cfg: EKFConfig | None = None) -> None:
        self.cfg = cfg or EKFConfig()
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None

    def reset(self) -> None:
        self._x = None
        self._P = None

    @property
    def initialized(self) -> bool:
        return self._x is not None

    @staticmethod
    def _effective_dt(dt_minutes: float | None, fallback: float) -> float:
        """校验并夹取步长；`None` 用配置默认值（向后兼容固定步长调用）。"""
        dt = fallback if dt_minutes is None else float(dt_minutes)
        if not np.isfinite(dt):
            raise ValueError(f"dt_minutes 必须有限，得到 {dt}")
        return float(np.clip(dt, MIN_DT_MINUTES, MAX_DT_MINUTES))

    def _transition(self, dt_minutes: float) -> np.ndarray:
        dt_h = dt_minutes / 60.0
        return np.array([[1.0, dt_h], [0.0, 1.0]])

    def _process_noise_cov(self, dt_minutes: float) -> np.ndarray:
        dt_h = max(dt_minutes / 60.0, 1e-3)
        return np.diag(
            [
                (self.cfg.process_noise_d * dt_h) ** 2,
                (self.cfg.process_noise_rate * dt_h) ** 2,
            ]
        )

    def update(self, z: float | None, dt_minutes: float | None = None) -> EKFState:
        """推进一拍。

        `dt_minutes` 为距上一帧的真实间隔（小时级采样时由 Image_Time 求得）；
        传 `None` 则用 `cfg.dt_minutes`。
        """
        cfg = self.cfg
        dt = self._effective_dt(dt_minutes, cfg.dt_minutes)

        if self._x is None or self._P is None:
            if z is None:
                return EKFState(0.0, 0.0, False, False, False, 0.0)
            self._x = np.array([float(z), 0.0])
            self._P = np.eye(2) * cfg.init_variance
            return EKFState(float(z), 0.0, True, False, False, 0.0)

        F = self._transition(dt)
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + self._process_noise_cov(dt)

        if z is None:
            self._x, self._P = x_pred, P_pred
            return EKFState(float(x_pred[0]), float(x_pred[1]), False, False, False, dt)

        H = np.array([[1.0, 0.0]])
        R = np.array([[cfg.measurement_noise_mm**2]])
        # numpy 2.x 下 1×1 数组不能用 float() 直接转换，统一取标量
        y = float(z) - float((H @ x_pred).item())
        S = float((H @ P_pred @ H.T + R).item())
        gate = cfg.innovation_gate_sigma * np.sqrt(max(S, 1e-9))
        if abs(y) > gate:
            # 离群：只预测不更新，等待下一帧确认（真实生长不会一步跳 10 mm）
            self._x, self._P = x_pred, P_pred
            return EKFState(float(x_pred[0]), float(x_pred[1]), False, True, False, dt)

        K = P_pred @ H.T / S
        self._x = x_pred + (K.flatten() * y)
        self._P = (np.eye(2) - K @ H) @ P_pred

        rate_max = cfg.growth_rate_max_mm_per_h
        clamped = False
        if abs(self._x[1]) > rate_max:
            self._x[1] = float(np.clip(self._x[1], -rate_max, rate_max))
            clamped = True
        if self._x[0] < 0:
            self._x[0] = 0.0

        return EKFState(float(self._x[0]), float(self._x[1]), True, False, clamped, dt)

    def replay(
        self, history: list[float], timestamps: list[str] | None = None
    ) -> EKFState:
        """把历史序列重放进 EKF，返回最后一帧状态。

        `timestamps` 为与 `history` 等长的 ISO 时间串；给出时按真实间隔推进，
        否则沿用固定 `dt_minutes`（兼容旧调用）。
        """
        if timestamps is not None and len(timestamps) != len(history):
            raise ValueError(
                f"timestamps 长度 {len(timestamps)} 与 history 长度 {len(history)} 不一致"
            )
        dts = _intervals_minutes(timestamps) if timestamps else [None] * len(history)

        state = EKFState(0.0, 0.0, False, False, False, 0.0)
        for z, dt in zip(history, dts, strict=True):
            state = self.update(z, dt_minutes=dt)
        return state


def _intervals_minutes(timestamps: list[str]) -> list[float | None]:
    """由时间戳序列求相邻间隔（分钟）。首帧为 None（初始化不推进）。"""
    from datetime import datetime

    out: list[float | None] = []
    prev: datetime | None = None
    for ts in timestamps:
        try:
            cur = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            out.append(None)
            prev = None
            continue
        if prev is None:
            out.append(None)
        else:
            out.append(abs((cur - prev).total_seconds()) / 60.0)
        prev = cur
    return out
