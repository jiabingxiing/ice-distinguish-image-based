"""覆冰类型单帧推理：质检 + 预处理 + 特征提取/双视角融合 + 温度校准 + 三重拒识。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.common.schemas import IceType, IcingResult
from grid_ice_vision.data.transforms.common import letterbox, to_chw
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    RejectionThresholds,
    apply_rejection_gate,
)
from grid_ice_vision.features.ice_type.classifier import SklearnClassifierAdapter
from grid_ice_vision.features.ice_type.quality import (
    IceTypeQuality,
    IceTypeQualityThresholds,
    evaluate_ice_type_quality,
)
from grid_ice_vision.features.ice_type.roi import (
    NormalizedRoi,
    center_fallback_crop,
    crop_normalized_roi,
    detect_view_drift,
)


@dataclass(frozen=True)
class IceTypeInferConfig:
    """推理配置快照（由 configs/model/ice_type_mnv3s.yaml 注入）。"""

    image_size: int = 224
    roi_size: int = 224
    pad_value: int = 114
    view_mode: Literal["full", "roi", "dual"] = "full"
    model_name: str = "mobilenet_v3_small_ice_type"
    model_version: str = "v1"
    drift_corr_min: float = 0.35
    quality_thresholds: IceTypeQualityThresholds = field(default_factory=IceTypeQualityThresholds)
    rejection_thresholds: RejectionThresholds = field(default_factory=RejectionThresholds)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> IceTypeInferConfig:
        if not cfg:
            return cls()
        m = cfg.get("model", {}) or {}
        p = cfg.get("preprocess", {}) or {}
        q = cfg.get("quality", {}) or {}
        vm = str(m.get("view_mode", "full"))
        view_mode: Literal["full", "roi", "dual"] = (
            vm if vm in ("full", "roi", "dual") else "full"  # type: ignore[assignment]
        )
        return cls(
            image_size=int(p.get("image_size", 224)),
            roi_size=int(p.get("roi_size", 224)),
            pad_value=int(p.get("pad_value", 114)),
            view_mode=view_mode,
            model_name=str(m.get("backbone", "mobilenetv3_small_100")),
            model_version=str(m.get("version", "v1")),
            drift_corr_min=float(q.get("drift_corr_min", 0.35)),
            quality_thresholds=IceTypeQualityThresholds.from_config(cfg),
            rejection_thresholds=RejectionThresholds.from_config(cfg),
        )


@dataclass(frozen=True)
class IceTypePrediction:
    """四分类推理结果对象，可映射为契约 `IcingResult`。"""

    ice_type: IceType
    ice_type_conf: float
    ice_type_source: Literal["model", "unavailable"]
    class_probabilities: dict[str, float]
    requires_review: bool
    review_reason: str | None
    quality: IceTypeQuality
    roi_used: bool
    view_drifted: bool
    inference_ms: float

    def to_icing_result(
        self,
        *,
        present: bool | None = True,
        confidence: float | None = None,
        coverage_ratio: float | None = None,
        bridge_ratio: float | None = None,
    ) -> IcingResult:
        return IcingResult(
            present=present,
            confidence=self.ice_type_conf if confidence is None else confidence,
            coverage_ratio=coverage_ratio,
            ice_type=self.ice_type,
            ice_type_conf=self.ice_type_conf,
            ice_type_source=self.ice_type_source,
            class_probabilities=dict(self.class_probabilities),
            requires_review=self.requires_review,
            review_reason=self.review_reason,
            bridge_ratio=bridge_ratio,
        )


class IceTypeTorchModel(nn.Module):
    """端到端可导出推理模型：IceTypeBackbone + PyTorch 分类头 + 可选温度缩放。"""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        view_mode: Literal["full", "roi", "dual"] = "full",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.view_mode = view_mode
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, x: Tensor, roi: Tensor | None = None) -> Tensor:
        feats = (
            self.backbone(x, roi)
            if self.view_mode == "dual" and roi is not None
            else self.backbone(x)
        )
        logits = self.head(feats)
        return logits / self.temperature


class IceTypeInferencer:
    """纯算法覆冰类型推理器（不读文件、不解析 YAML、不写日志）。"""

    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        head: nn.Module | SklearnClassifierAdapter | None = None,
        logits_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None = None,
        scaler: MultiClassTemperatureScaler | None = None,
        cfg: IceTypeInferConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or IceTypeInferConfig()
        self.device = device
        self.backbone = backbone.to(device).eval() if backbone is not None else None
        self.head = head.to(device).eval() if isinstance(head, nn.Module) else head
        self.logits_fn = logits_fn
        self.scaler = scaler or MultiClassTemperatureScaler()
        if self.logits_fn is None and (self.backbone is None or self.head is None):
            raise ValueError("必须提供 (backbone + head) 或 logits_fn")

    @torch.no_grad()
    def _compute_logits(self, full_chw: np.ndarray, roi_chw: np.ndarray | None) -> np.ndarray:
        if self.logits_fn is not None:
            x_full = full_chw[None, ...].astype(np.float32)
            x_roi = roi_chw[None, ...].astype(np.float32) if roi_chw is not None else None
            return np.asarray(self.logits_fn(x_full, x_roi), dtype=np.float64)

        assert self.backbone is not None and self.head is not None
        t_full = torch.from_numpy(full_chw[None, ...]).to(self.device)
        if self.cfg.view_mode == "dual" and roi_chw is not None:
            t_roi = torch.from_numpy(roi_chw[None, ...]).to(self.device)
            feats = self.backbone(t_full, t_roi)
        elif self.cfg.view_mode == "roi" and roi_chw is not None:
            t_roi = torch.from_numpy(roi_chw[None, ...]).to(self.device)
            feats = self.backbone(t_roi)
        else:
            feats = self.backbone(t_full)

        if isinstance(self.head, SklearnClassifierAdapter):
            return self.head.predict_logits(feats.cpu().numpy())
        assert isinstance(self.head, nn.Module)
        return self.head(feats).cpu().numpy().astype(np.float64)

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        roi: NormalizedRoi | None = None,
        ref_signature: np.ndarray | None = None,
    ) -> IceTypePrediction:
        t0 = time.perf_counter()

        roi_crop: np.ndarray | None = None
        roi_used = False
        view_drifted = False
        if roi is not None:
            roi_crop = crop_normalized_roi(image_bgr, roi)
            roi_used = True
            if ref_signature is not None:
                view_drifted, _ = detect_view_drift(
                    roi_crop, ref_signature, corr_min=self.cfg.drift_corr_min
                )
        elif self.cfg.view_mode in ("roi", "dual"):
            roi_crop = center_fallback_crop(image_bgr)

        q = evaluate_ice_type_quality(
            image_bgr,
            thresholds=self.cfg.quality_thresholds,
            roi_image_bgr=roi_crop if roi_used else None,
        )

        # 若图像质量不合格或视角严重偏移，直接触发质量拒识（避免给不可信概率）
        if not q.usable:
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return IceTypePrediction(
                ice_type=IceType.UNKNOWN,
                ice_type_conf=0.0,
                ice_type_source="unavailable",
                class_probabilities={},
                requires_review=True,
                review_reason="low_image_quality",
                quality=q,
                roi_used=roi_used,
                view_drifted=view_drifted,
                inference_ms=ms,
            )

        full_lb = letterbox(image_bgr, self.cfg.image_size, pad_value=self.cfg.pad_value)
        full_chw = to_chw(full_lb, normalize=True)
        roi_chw: np.ndarray | None = None
        if roi_crop is not None:
            roi_lb = letterbox(roi_crop, self.cfg.roi_size, pad_value=self.cfg.pad_value)
            roi_chw = to_chw(roi_lb, normalize=True)

        logits = self._compute_logits(full_chw, roi_chw)
        probs = self.scaler.predict_proba(logits)[0]
        class_probs = {
            name: round(float(p), 4) for name, p in zip(ICE_TYPE_FOUR_CLASSES, probs, strict=True)
        }

        rej_mask, reasons = apply_rejection_gate(
            probs[None, :],
            np.array([q.q_image], dtype=np.float64),
            self.cfg.rejection_thresholds,
        )
        rejected = bool(rej_mask[0])
        reason = reasons[0] if rejected and reasons[0] else None
        if view_drifted and reason is None:
            rejected = True
            reason = "roi_view_drift"

        top_idx = int(np.argmax(probs))
        top_conf = round(float(probs[top_idx]), 4)
        pred_type = IceType.UNKNOWN if rejected else IceType(ICE_TYPE_FOUR_CLASSES[top_idx])

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return IceTypePrediction(
            ice_type=pred_type,
            ice_type_conf=top_conf,
            ice_type_source="model",
            class_probabilities=class_probs,
            requires_review=rejected,
            review_reason=reason,
            quality=q,
            roi_used=roi_used,
            view_drifted=view_drifted,
            inference_ms=ms,
        )
