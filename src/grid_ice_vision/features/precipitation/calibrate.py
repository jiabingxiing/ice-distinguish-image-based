"""温度标定与 ECE（§10.2.4）。

选温度标定而非 MC-Dropout：单卡训练成本低、推理零开销，对分类任务足够。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bce(logits: np.ndarray, targets: np.ndarray, t: float) -> float:
    z = logits / max(t, 1e-6)
    # 数值稳定版 BCE
    return float(np.mean(np.logaddexp(0.0, z) - targets * z))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """ECE = Σ (n_b/N)·|acc_b − conf_b|，用于二值/多标签。"""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()
    if p.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = (y[m] > 0.5).mean()
        ece += (m.sum() / p.size) * abs(acc - conf)
    return float(ece)


@dataclass
class TemperatureScaler:
    """单参数温度标定。`temperature=1.0` 表示未标定。"""

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> TemperatureScaler:
        """在验证集上最小化 BCE，T 在 log 空间搜索（数值更稳）。"""
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if z.size == 0:
            return self
        res = minimize_scalar(
            lambda log_t: _bce(z, y, float(np.exp(log_t))),
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
        )
        self.temperature = float(np.exp(res.x)) if res.success else 1.0
        self.fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.transform(logits))

    def to_dict(self) -> dict[str, float | bool]:
        return {"temperature": self.temperature, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: dict[str, float | bool]) -> TemperatureScaler:
        return cls(
            temperature=float(d.get("temperature", 1.0)), fitted=bool(d.get("fitted", False))
        )
