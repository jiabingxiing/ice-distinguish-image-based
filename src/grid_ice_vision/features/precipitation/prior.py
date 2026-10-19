"""气象物理先验（§10.2.3）。

不做硬规则「T<0 才可能是 glaze」—— 传感器误差、太阳辐射、设备表面温度
与气温差都会打破它。这里只把历史统计做成**软先验**，以 `lam` 加权叠加到 logits。
`lam=0` 时完全退化为纯视觉模型。

⚠️ 内置表格是**占位值**，仅表达「低温更可能有雾凇/雨凇」这一单调趋势。
上线前必须用本地区历史气象记录重估，并写入 `configs/model/precip_prior.yaml`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from grid_ice_vision.common.constants import WEATHER_LABELS

DEFAULT_TEMP_BINS: tuple[float, ...] = (-30.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 40.0)

# 每行对应一个温度区间 [bin_i, bin_{i+1})，列顺序同 WEATHER_LABELS
DEFAULT_TABLE: tuple[tuple[float, ...], ...] = (
    (0.05, 0.00, 0.25, 0.00, 0.05, 0.15, 0.40, 0.10),  # < -10℃
    (0.05, 0.00, 0.20, 0.00, 0.05, 0.10, 0.45, 0.15),  # -10 ~ -5
    (0.05, 0.02, 0.15, 0.00, 0.08, 0.05, 0.35, 0.30),  # -5 ~ -2
    (0.05, 0.10, 0.10, 0.00, 0.10, 0.02, 0.25, 0.38),  # -2 ~ 0
    (0.05, 0.30, 0.05, 0.00, 0.12, 0.01, 0.15, 0.32),  # 0 ~ 2
    (0.10, 0.45, 0.02, 0.00, 0.10, 0.00, 0.08, 0.15),  # 2 ~ 5
    (0.25, 0.50, 0.00, 0.01, 0.08, 0.00, 0.02, 0.04),  # 5 ~ 10
    (0.35, 0.45, 0.00, 0.01, 0.05, 0.00, 0.01, 0.02),  # >= 10
)


@dataclass(frozen=True)
class TemperaturePrior:
    bins: tuple[float, ...] = DEFAULT_TEMP_BINS
    table: tuple[tuple[float, ...], ...] = DEFAULT_TABLE

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> TemperaturePrior:
        if not cfg:
            return cls()
        prior = cfg.get("prior", cfg)
        bins = prior.get("temp_bins")
        table = prior.get("table")
        if not bins or not table:
            return cls()
        return cls(
            bins=tuple(float(b) for b in bins),
            table=tuple(tuple(float(v) for v in row) for row in table),
        )

    def bin_index(self, t_amb_c: float) -> int:
        idx = int(np.searchsorted(np.asarray(self.bins[1:-1]), float(t_amb_c), side="right"))
        return min(idx, len(self.table) - 1)

    def log_prior(self, t_amb_c: float) -> np.ndarray:
        probs = np.asarray(self.table[self.bin_index(t_amb_c)], dtype=np.float64)
        probs = np.clip(probs, 1e-6, 1.0)
        return np.log(probs)

    def apply(self, logits: Tensor, t_amb_c: float | None, lam: float = 0.3) -> Tensor:
        """logits ← logits + λ·log P(class | T)。`t_amb_c=None` 或 `lam=0` 时原样返回。"""
        if t_amb_c is None or lam <= 0:
            return logits
        prior = torch.as_tensor(self.log_prior(t_amb_c), dtype=logits.dtype, device=logits.device)
        return logits + lam * prior.unsqueeze(0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "temp_bins": list(self.bins),
            "labels": list(WEATHER_LABELS),
            "table": [list(r) for r in self.table],
        }
