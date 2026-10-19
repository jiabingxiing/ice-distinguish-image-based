"""时序滤波：2 状态 EKF + 物理增长率约束（§10.7.2）。

单帧在暴风雪中会被飞掠前景雪花、镜头水珠、导线风振污染，产生 5–20 mm 的
瞬时跳变；而真实覆冰生长是小时级慢变过程。用「增长率上限 + 新息门控」
即可剔除绝大部分尖峰，且不需要任何额外传感器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    def _transition(self) -> np.ndarray:
        dt_h = self.cfg.dt_minutes / 60.0
        return np.array([[1.0, dt_h], [0.0, 1.0]])

    def _process_noise_cov(self) -> np.ndarray:
        dt_h = max(self.cfg.dt_minutes / 60.0, 1e-3)
        return np.diag(
            [
                (self.cfg.process_noise_d * dt_h) ** 2,
                (self.cfg.process_noise_rate * dt_h) ** 2,
            ]
        )

    def update(self, z: float | None) -> EKFState:
        cfg = self.cfg
        if self._x is None or self._P is None:
            if z is None:
                return EKFState(0.0, 0.0, False, False, False)
            self._x = np.array([float(z), 0.0])
            self._P = np.eye(2) * cfg.init_variance
            return EKFState(float(z), 0.0, True, False, False)

        F = self._transition()
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + self._process_noise_cov()

        if z is None:
            self._x, self._P = x_pred, P_pred
            return EKFState(float(x_pred[0]), float(x_pred[1]), False, False, False)

        H = np.array([[1.0, 0.0]])
        R = np.array([[cfg.measurement_noise_mm**2]])
        # numpy 2.x 下 1×1 数组不能用 float() 直接转换，统一取标量
        y = float(z) - float((H @ x_pred).item())
        S = float((H @ P_pred @ H.T + R).item())
        gate = cfg.innovation_gate_sigma * np.sqrt(max(S, 1e-9))
        if abs(y) > gate:
            # 离群：只预测不更新，等待下一帧确认（真实生长不会一步跳 10 mm）
            self._x, self._P = x_pred, P_pred
            return EKFState(float(x_pred[0]), float(x_pred[1]), False, True, False)

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

        return EKFState(float(self._x[0]), float(self._x[1]), True, False, clamped)

    def replay(self, history: list[float]) -> EKFState:
        """把历史序列重放进 EKF，返回最后一帧状态。"""
        state = EKFState(0.0, 0.0, False, False, False)
        for z in history:
            state = self.update(z)
        return state
