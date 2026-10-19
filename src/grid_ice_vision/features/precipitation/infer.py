"""T1 推理：概率 → 拒识 → 契约对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from grid_ice_vision.common.constants import DEFAULT_NOTE, INTENSITY_LEVELS, WEATHER_LABELS
from grid_ice_vision.common.schemas import Intensity, PrecipResult, QualityStatus
from grid_ice_vision.features.precipitation.calibrate import TemperatureScaler
from grid_ice_vision.features.precipitation.model import PrecipitationNet
from grid_ice_vision.features.precipitation.prior import TemperaturePrior


@dataclass(frozen=True)
class RejectPolicy:
    tau_reject: float = 0.45
    h_max: float = 1.8

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> RejectPolicy:
        if not cfg:
            return cls()
        r = cfg.get("reject", cfg)
        return cls(tau_reject=float(r.get("tau_reject", 0.45)), h_max=float(r.get("h_max", 1.8)))


@dataclass
class PrecipPrediction:
    labels: dict[str, float] = field(default_factory=dict)
    intensity: Intensity = Intensity.UNKNOWN
    intensity_conf: float = 0.0
    rejected: bool = False
    entropy: float = 0.0
    met_used: bool = False

    def to_result(self) -> PrecipResult:
        note = DEFAULT_NOTE
        if self.rejected:
            note = f"rejected_low_confidence(entropy={self.entropy:.2f}); {DEFAULT_NOTE}"
        return PrecipResult(
            labels=self.labels,
            intensity=self.intensity,
            intensity_conf=self.intensity_conf,
            meteorological_input_used=self.met_used,
            note=note,
        )


def binary_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def coral_to_level(intensity_logits: Tensor) -> tuple[int, float]:
    """CORAL 推理：level = Σ 1[σ(z_k) > 0.5]；置信度取最小决策裕度。"""
    probs = torch.sigmoid(intensity_logits).detach().cpu().numpy().ravel()
    level = int((probs > 0.5).sum())
    margin = float(np.min(np.abs(probs - 0.5)) * 2.0)
    return min(level, len(INTENSITY_LEVELS) - 1), max(0.0, min(margin, 1.0))


class PrecipInferencer:
    """封装模型、先验、标定与拒识规则。"""

    def __init__(
        self,
        model: PrecipitationNet,
        scaler: TemperatureScaler | None = None,
        prior: TemperaturePrior | None = None,
        policy: RejectPolicy | None = None,
        lam: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.scaler = scaler or TemperatureScaler()
        self.prior = prior or TemperaturePrior()
        self.policy = policy or RejectPolicy()
        self.lam = lam
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image_tensor: Tensor,
        t_amb_c: float | None = None,
        met: Tensor | None = None,
        quality_status: QualityStatus = QualityStatus.USABLE,
    ) -> PrecipPrediction:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        out = self.model(image_tensor.to(self.device), met)
        logits = self.prior.apply(out["weather_logits"], t_amb_c, self.lam)
        probs = self.scaler.predict_proba(logits.detach().cpu().numpy())[0]
        labels = {name: round(float(p), 4) for name, p in zip(WEATHER_LABELS, probs, strict=True)}

        level, conf = coral_to_level(out["intensity_logits"])
        ent = binary_entropy(probs)
        rejected = False
        if (
            quality_status is QualityStatus.UNUSABLE
            or float(probs.max()) < self.policy.tau_reject
            or ent > self.policy.h_max
        ):
            rejected = True

        return PrecipPrediction(
            labels=labels,
            intensity=Intensity.UNKNOWN if rejected else Intensity(INTENSITY_LEVELS[level]),
            intensity_conf=0.0 if rejected else round(conf, 4),
            rejected=rejected,
            entropy=round(ent, 4),
            met_used=met is not None and t_amb_c is not None,
        )
