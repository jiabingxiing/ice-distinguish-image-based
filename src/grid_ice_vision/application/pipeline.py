"""全链路编排（§7.1 的 T0–T5）。

编排原则：
- 天气分支与覆冰分支**互不阻塞**：没检测到资产时仍然输出天气结论；
- 任一层失败都走显式降级路径（§9.3），不抛异常给调用方；
- 输出的每一个数值都必须能追溯到 manifest / 配置 / 模型版本。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grid_ice_vision.application.decision import (
    AlarmRules,
    AlarmStateMachine,
    classify_level,
    requires_review,
)
from grid_ice_vision.application.evidence import draw_heatmap, draw_mask_overlay
from grid_ice_vision.application.temporal import EKFConfig, ThicknessEKF
from grid_ice_vision.common.config import config_hash, get
from grid_ice_vision.common.constants import RHO_BY_ICE_TYPE
from grid_ice_vision.common.logging import get_logger
from grid_ice_vision.common.schemas import (
    Asset,
    AssetClass,
    CalibInput,
    IcingResult,
    InferenceResult,
    Intensity,
    MeasurementStatus,
    Provenance,
    QualityInfo,
    QualityStatus,
    SeverityResult,
    SourceInfo,
    VisualGrade,
)
from grid_ice_vision.features.detection.infer import DetectorInferConfig, DetectorInferencer
from grid_ice_vision.features.detection.model import DetectionNet
from grid_ice_vision.features.ice.infer import IceSegInferencer, SegInferConfig
from grid_ice_vision.features.ice.model import UNetLite
from grid_ice_vision.features.precipitation.calibrate import TemperatureScaler
from grid_ice_vision.features.precipitation.infer import (
    PrecipInferencer,
    PrecipPrediction,
    RejectPolicy,
)
from grid_ice_vision.features.precipitation.model import PrecipitationNet
from grid_ice_vision.features.precipitation.prior import TemperaturePrior
from grid_ice_vision.features.quality.gate import (
    LensIcingTracker,
    QualityThresholds,
    assess_quality,
)
from grid_ice_vision.features.severity.calibration import resolve_calibration
from grid_ice_vision.features.severity.geometry import measure_widths
from grid_ice_vision.features.severity.grading import SeverityThresholds, grade_from_coverage
from grid_ice_vision.features.severity.thickness import estimate_thickness

log = get_logger("pipeline")

_PIPELINE_VERSION = "0.1.0"


@dataclass
class PipelineConfig:
    device: str = "cpu"
    backbone: str = "mobilenetv4_conv_small"
    pretrained: bool = False
    weights: dict[str, str | None] = field(default_factory=dict)
    evidence_dir: str | None = "artifacts/evidence"
    seg_input_size: int = 512
    min_coverage: float = 0.03
    lam: float = 0.3
    quality: dict[str, Any] = field(default_factory=dict)
    precip: dict[str, Any] = field(default_factory=dict)
    detector: dict[str, Any] = field(default_factory=dict)
    seg: dict[str, Any] = field(default_factory=dict)
    alarm: dict[str, Any] = field(default_factory=dict)
    severity: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> PipelineConfig:
        return cls(
            device=str(cfg.get("device", "cpu")),
            backbone=str(get(cfg, "model.backbone", "mobilenetv4_conv_small")),
            pretrained=bool(get(cfg, "model.pretrained", False)),
            weights=dict(cfg.get("weights", {}) or {}),
            evidence_dir=cfg.get("evidence_dir", "artifacts/evidence"),
            seg_input_size=int(get(cfg, "infer.seg.input_size", 512)),
            min_coverage=float(get(cfg, "infer.seg.min_coverage", 0.03)),
            lam=float(get(cfg, "infer.precip.lam", 0.3)),
            quality=dict(cfg.get("quality", {}) or {}),
            precip=dict(cfg.get("precip", {}) or {}),
            detector=dict(cfg.get("detector", {}) or {}),
            seg=dict(cfg.get("seg", {}) or {}),
            alarm=dict(cfg.get("alarm", {}) or cfg.get("rules", {}) or {}),
            severity=dict(cfg.get("severity", {}) or {}),
            raw=cfg,
        )


class InferencePipeline:
    """把 T0–T5 串成一次调用。"""

    def __init__(
        self,
        precip_model: PrecipitationNet,
        detector_model: DetectionNet,
        seg_model: UNetLite,
        cfg: PipelineConfig,
        scaler: TemperatureScaler | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.precip = PrecipInferencer(
            precip_model,
            scaler=scaler,
            prior=TemperaturePrior.from_config(cfg.raw),
            policy=RejectPolicy.from_config(cfg.precip or cfg.raw),
            lam=cfg.lam,
            device=cfg.device,
        )
        self.detector = DetectorInferencer(
            detector_model,
            DetectorInferConfig.from_config(cfg.detector or cfg.raw),
            device=cfg.device,
        )
        self.seg = IceSegInferencer(
            seg_model, SegInferConfig.from_config(cfg.seg or cfg.raw), device=cfg.device
        )
        self.quality_th = QualityThresholds.from_config(cfg.quality or cfg.raw)
        self.alarm_rules = AlarmRules.from_config(cfg.alarm or cfg.raw)
        self.severity_th = SeverityThresholds.from_config(cfg.severity or cfg.raw)
        self.ekf_cfg = EKFConfig.from_config(cfg.alarm or cfg.raw)
        self.lens_tracker = LensIcingTracker()
        self.alarms = AlarmStateMachine(self.alarm_rules)
        self._cfg_hash = config_hash(cfg.raw)

    # ------------------------------------------------------------------ 构建
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        device: str = "cpu",
        load_weights: bool = True,
    ) -> InferencePipeline:
        pc = PipelineConfig.from_dict({**cfg, "device": device})
        backbone = pc.backbone
        precip_model = PrecipitationNet(
            backbone_name=backbone,
            pretrained=pc.pretrained,
            use_film=bool(get(cfg, "model.precip.use_film", True)),
        )
        det_model = DetectionNet(
            backbone_name=backbone,
            pretrained=pc.pretrained,
            reg_max=int(get(cfg, "model.detector.reg_max", 8)),
        )
        seg_model = UNetLite(backbone_name=backbone, pretrained=pc.pretrained)
        if load_weights:
            _maybe_load(precip_model, pc.weights.get("precip"), device)
            _maybe_load(det_model, pc.weights.get("detector"), device)
            _maybe_load(seg_model, pc.weights.get("seg"), device)
        scaler = TemperatureScaler()
        cal_path = Path(pc.weights.get("calibration") or "configs/model/precip_calibration.json")
        if cal_path.exists():
            import json

            scaler = TemperatureScaler.from_dict(json.loads(cal_path.read_text("utf-8")))
        return cls(precip_model, det_model, seg_model, pc, scaler)

    # ------------------------------------------------------------------ 推理
    def run(
        self,
        image: np.ndarray,
        image_id: str,
        source: SourceInfo | None = None,
        calib: CalibInput | None = None,
        history_d_eq_mm: list[float] | None = None,
        write_evidence: bool = True,
    ) -> InferenceResult:
        t_start = time.perf_counter()
        calib = calib or CalibInput()
        camera_id = source.camera_id if source else None
        quality = assess_quality(
            image, self.quality_th, camera_id=camera_id, tracker=self.lens_tracker
        )

        # 降级路径 1：图像不可用 → 终止推理，下游一律不给结论
        if quality.status is QualityStatus.UNUSABLE:
            return self._degraded_result(image_id, source, quality, t_start)

        met = _met_tensor(source)
        t_amb = source.weather_sensor.t_amb_c if source and source.weather_sensor else None
        pred: PrecipPrediction = self.precip.predict(
            _to_tensor(image), t_amb_c=t_amb, met=met, quality_status=quality.status
        )

        assets = self.detector.predict(image)
        primary = _pick_primary(assets)
        if primary is None:
            # 降级路径 2：无资产 → 只出天气结论
            return self._build_result(
                image_id=image_id,
                source=source,
                quality=quality,
                pred=pred,
                assets=assets,
                icing=IcingResult(present=None, confidence=0.0),
                severity=_null_severity(quality),
                evidence={},
                t_start=t_start,
            )

        seg = self.seg.predict(
            image, primary.bbox, is_insulator=primary.cls is AssetClass.INSULATOR_STRING
        )
        coverage = seg.coverage
        grade = (
            grade_from_coverage(coverage, self.severity_th)
            if coverage is not None
            else VisualGrade.G0
        )

        evidence: dict[str, str] = {}
        if write_evidence and self.cfg.evidence_dir:
            evidence = self._write_evidence(image, image_id, seg, grade)

        severity = self._severity(
            image=image,
            seg_mask=seg.mask,
            coverage=coverage,
            grade=grade,
            calib=calib,
            source=source,
            ice_type=None,
            quality=quality,
            history=history_d_eq_mm or [],
            roi_id=primary.roi_id,
        )

        icing = IcingResult(
            present=seg.ice_present,
            confidence=seg.confidence,
            coverage_ratio=None if coverage is None else round(coverage, 4),
            ice_type=severity_ice_type(severity),
            ice_type_conf=0.0,
            bridge_ratio=None if seg.bridge is None else round(seg.bridge, 4),
            mask_rle=seg.mask_rle(),
        )

        return self._build_result(
            image_id=image_id,
            source=source,
            quality=quality,
            pred=pred,
            assets=assets,
            icing=icing,
            severity=severity,
            evidence=evidence,
            t_start=t_start,
        )

    # ------------------------------------------------------------------ 内部
    def _severity(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        coverage: float | None,
        grade: VisualGrade,
        calib: CalibInput,
        source: SourceInfo | None,
        ice_type: str | None,
        quality: QualityInfo,
        history: list[float],
        roi_id: str,
    ) -> SeverityResult:
        outcome = resolve_calibration(
            calib,
            ptz_zoom_now=source.ptz.zoom if source and source.ptz else None,
            ptz_zoom_ref=None,
        )
        review = outcome.requires_review or quality.status is QualityStatus.DEGRADED

        if outcome.alpha_mm_per_px is None or calib.d0_mm is None:
            return SeverityResult(
                visual_grade=grade,
                alarm_level=None,
                measurement_status=outcome.status,
                calibration_source=outcome.source,
                requires_review=review,
            )

        alpha = outcome.alpha_mm_per_px
        w0_px = calib.w0_px if calib.w0_px is not None else calib.d0_mm / alpha

        obj_mask = (seg_mask >= 1).astype(np.uint8)  # 冰 + 裸露设备 = 当前设备轮廓
        try:
            wm = measure_widths(obj_mask, spacing=4.0)
        except ValueError as exc:
            log.warning("宽度测量失败，降级为 visual_only：%s", exc)
            return SeverityResult(
                visual_grade=grade,
                alarm_level=None,
                measurement_status=MeasurementStatus.VISUAL_ONLY,
                calibration_source=outcome.source,
                alpha_mm_per_px=alpha,
                requires_review=True,
            )

        rho = RHO_BY_ICE_TYPE.get(ice_type or "unknown", 900.0)
        est = estimate_thickness(
            w_ice_px=wm.width_px,
            w0_px=w0_px,
            alpha_mm_per_px=alpha,
            d0_mm=calib.d0_mm,
            rho_ice=rho,
            design_ice_mm=calib.design_ice_mm,
        )

        ekf = ThicknessEKF(self.ekf_cfg)
        if history:
            ekf.replay(history)
        state = ekf.update(est.d_eq_mm)

        candidate = classify_level(
            self.alarm_rules,
            d_eq_mm=state.d_eq_mm,
            gamma=est.gamma,
            bridge=None,
            measurement_status=outcome.status,
        )
        key = (
            f"{source.site_id if source else 'NA'}|{source.camera_id if source else 'NA'}|{roi_id}"
        )
        level = self.alarms.update(key, candidate, outcome.status)

        status = outcome.status
        if status is MeasurementStatus.MEASURED and calib.design_ice_mm is None:
            # 无设计冰厚 → 无法算载荷比，但等值厚度仍可信
            pass

        return SeverityResult(
            visual_grade=grade,
            alarm_level=level,
            d_radial_mm=round(est.d_radial_mm, 2),
            d_eq_mm=round(state.d_eq_mm, 2),
            gamma=None if est.gamma is None else round(est.gamma, 4),
            measurement_status=status,
            calibration_source=outcome.source,
            alpha_mm_per_px=round(alpha, 5),
            uncertainty_mm=round(est.uncertainty_mm, 2),
            requires_review=review or requires_review(level, self.alarm_rules),
        )

    def _write_evidence(
        self, image: np.ndarray, image_id: str, seg: Any, grade: VisualGrade
    ) -> dict[str, str]:
        out_dir = Path(self.cfg.evidence_dir or "artifacts/evidence")
        paths: dict[str, str] = {}
        try:
            paths["mask_overlay"] = draw_mask_overlay(
                image, seg.mask, seg.roi_bbox, out_dir / f"{image_id}_mask.png"
            )
            paths["heatmap"] = draw_heatmap(
                seg.probs[2] + seg.probs[3], out_dir / f"{image_id}_prob.png"
            )
        except Exception as exc:
            log.warning("证据图生成失败：%s", exc)
        return paths

    def _degraded_result(
        self, image_id: str, source: SourceInfo | None, quality: QualityInfo, t_start: float
    ) -> InferenceResult:
        return self._build_result(
            image_id=image_id,
            source=source,
            quality=quality,
            pred=PrecipPrediction(
                labels={}, intensity=Intensity.UNKNOWN, rejected=True, entropy=0.0
            ),
            assets=[],
            icing=IcingResult(present=None, confidence=0.0),
            severity=_null_severity(quality),
            evidence={},
            t_start=t_start,
        )

    def _build_result(
        self,
        image_id: str,
        source: SourceInfo | None,
        quality: QualityInfo,
        pred: PrecipPrediction,
        assets: list[Asset],
        icing: IcingResult,
        severity: SeverityResult,
        evidence: dict[str, str],
        t_start: float,
    ) -> InferenceResult:
        return InferenceResult(
            image_id=image_id,
            source=source,
            quality=quality,
            precipitation=pred.to_result(),
            assets=assets,
            icing=icing,
            severity=severity,
            evidence=evidence,
            provenance=Provenance(
                model_version={
                    "pipeline": _PIPELINE_VERSION,
                    "precip": self.precip.model.backbone.spec.name,
                    "detector": self.detector.model.backbone.spec.name,
                    "seg": self.seg.model.backbone.spec.name,
                },
                config_hash=self._cfg_hash[:16],
                inference_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                rule_version=self.alarm_rules.version,
            ),
        )


def severity_ice_type(severity: SeverityResult) -> Any:
    """类型头未接入时，用「有无覆冰」给出保守取值。"""
    from grid_ice_vision.common.schemas import IceType

    if severity.visual_grade is VisualGrade.G0:
        return IceType.NONE
    return IceType.UNKNOWN


def _null_severity(quality: QualityInfo) -> SeverityResult:
    return SeverityResult(
        visual_grade=VisualGrade.G0,
        alarm_level=None,
        measurement_status=MeasurementStatus.NOT_CALIBRATED,
        requires_review=quality.status is not QualityStatus.USABLE,
    )


def _pick_primary(assets: list[Asset]) -> Asset | None:
    """优先导线，其次绝缘子串；同类取最高分。"""
    order = {AssetClass.CONDUCTOR: 0, AssetClass.INSULATOR_STRING: 1}
    candidates = [a for a in assets if a.cls in order]
    if not candidates:
        return None
    return min(candidates, key=lambda a: (order[a.cls], -a.score))


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _met_tensor(source: SourceInfo | None) -> torch.Tensor | None:
    if source is None or source.weather_sensor is None:
        return None
    ws = source.weather_sensor
    return torch.tensor([[ws.t_amb_c, ws.rh, ws.v_wind_ms, ws.p_atm_hpa]], dtype=torch.float32)


def _maybe_load(model: torch.nn.Module, path: str | None, device: str) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        log.warning("权重不存在，使用随机初始化：%s", p)
        return
    state = torch.load(p, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning("权重不完全匹配：missing=%d unexpected=%d", len(missing), len(unexpected))
