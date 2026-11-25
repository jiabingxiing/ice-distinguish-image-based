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
from typing import TYPE_CHECKING, Any

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
from grid_ice_vision.common.logging import get_logger
from grid_ice_vision.common.schemas import (
    Asset,
    AssetClass,
    CalibInput,
    IceType,
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
from grid_ice_vision.features.severity.ice_type import (
    IceTypeSource,
    RhoPolicy,
    ice_type_confidence,
    resolve_ice_type,
)

if TYPE_CHECKING:
    from grid_ice_vision.deployment.ice_type_runtime import OnnxIceTypeService
from grid_ice_vision.features.severity.insulator_pa import (
    InsulatorPaConfig,
    InsulatorPaResult,
    compute_insulator_pa,
)
from grid_ice_vision.features.severity.thickness import estimate_thickness

log = get_logger("pipeline")

_PIPELINE_VERSION = "0.2.0"


def _round(value: float | None, ndigits: int) -> float | None:
    """None 安全的四舍五入：None 保持 None（不把缺失伪装成 0）。"""
    return None if value is None else round(float(value), ndigits)


def _step_dt_minutes(
    history_timestamps: list[str] | None, source: SourceInfo | None
) -> float | None:
    """求「最近一次历史帧 → 当前帧」的真实间隔（分钟）。

    没有时间戳或解析失败时返回 None，由 EKF 退回配置默认步长。
    跨度过大（> 24 h）也返回 None：那说明序列已断裂，应视为新起点而非慢生长。
    """
    if not history_timestamps or source is None or not source.timestamp:
        return None
    from datetime import datetime

    try:
        prev = datetime.fromisoformat(str(history_timestamps[-1]).replace("Z", "+00:00"))
        cur = datetime.fromisoformat(str(source.timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    minutes = abs((cur - prev).total_seconds()) / 60.0
    if minutes <= 0 or minutes > 24 * 60:
        return None
    return minutes


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
        ice_type_service: OnnxIceTypeService | None = None,
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
        self.ice_type_service = ice_type_service
        self.quality_th = QualityThresholds.from_config(cfg.quality or cfg.raw)
        self.alarm_rules = AlarmRules.from_config(cfg.alarm or cfg.raw)
        self.severity_th = SeverityThresholds.from_config(cfg.severity or cfg.raw)
        self.rho = RhoPolicy.from_config(cfg.raw)
        self.ekf_cfg = EKFConfig.from_config(cfg.alarm or cfg.raw)
        self.pa_cfg = InsulatorPaConfig.from_config(cfg.severity or cfg.raw)
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

        ice_service: OnnxIceTypeService | None = None
        rho_policy = RhoPolicy.from_config(cfg)
        it_path = pc.weights.get("ice_type")
        if load_weights and it_path and rho_policy.mode in ("shadow", "model"):
            from grid_ice_vision.common import paths
            from grid_ice_vision.deployment.ice_type_runtime import load_onnx_ice_type_inferencer

            p_onnx = Path(it_path) if Path(it_path).is_absolute() else paths.PROJ / it_path
            if p_onnx.exists():
                ice_service = load_onnx_ice_type_inferencer(
                    p_onnx,
                    roi_csv=pc.weights.get("ice_type_roi_csv"),
                )
            else:
                log.warning("配置了 ice_type 权重但文件不存在，退回 none_rule：%s", p_onnx)

        return cls(precip_model, det_model, seg_model, pc, scaler, ice_type_service=ice_service)

    # ------------------------------------------------------------------ 推理
    def run(
        self,
        image: np.ndarray,
        image_id: str,
        source: SourceInfo | None = None,
        calib: CalibInput | None = None,
        history_d_eq_mm: list[float] | None = None,
        write_evidence: bool = True,
        history_timestamps: list[str] | None = None,
        baseline_mask: np.ndarray | None = None,
        baseline_iou: float | None = None,
        baseline_corr: float | None = None,
    ) -> InferenceResult:
        """执行 T0–T5 全链路。

        参数
        ----
        history_timestamps : 与 `history_d_eq_mm` 等长的 ISO 时间串。给出时 EKF 按
            真实间隔（南网 1–6 h）推进，消除原固定 15 min 造成的增长率失真。
        baseline_mask : 同 `monitorId`、已配准的无冰基准掩码（0..3）。给出时才能算 `Pa`。
        baseline_iou / baseline_corr : 基准匹配的 IoU 与 ECC 相关系数，用于审计。
        """
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

        # 类型必须在厚度之前确定：密度折算要用它（shadow/model 启用 ONNX 分类 + 拒识保护）
        ice_type: IceType
        ice_src: IceTypeSource
        ice_type, ice_src = resolve_ice_type(grade)
        ice_type_conf = ice_type_confidence(ice_src)
        ice_class_probs: dict[str, float] = {}
        ice_review = False
        ice_reason: str | None = None

        g0_confirmed_no_ice = (
            self.rho.mode == "model"
            and grade is VisualGrade.G0
            and bool(self.cfg.weights.get("seg"))
        )
        if (
            self.ice_type_service is not None
            and self.rho.mode in ("shadow", "model")
            and not g0_confirmed_no_ice
        ):
            it_pred = self.ice_type_service.predict_bgr(image, monitor_id=camera_id)
            ice_type = it_pred.ice_type
            ice_type_conf = it_pred.ice_type_conf
            ice_src = it_pred.ice_type_source
            ice_class_probs = dict(it_pred.class_probabilities)
            ice_review = it_pred.requires_review
            ice_reason = it_pred.review_reason

        severity = self._severity(
            image=image,
            seg_mask=seg.mask,
            coverage=coverage,
            grade=grade,
            calib=calib,
            source=source,
            ice_type=ice_type,
            quality=quality,
            history=history_d_eq_mm or [],
            roi_id=primary.roi_id,
            history_timestamps=history_timestamps,
            baseline_mask=baseline_mask,
            baseline_iou=baseline_iou,
            baseline_corr=baseline_corr,
            is_insulator=primary.cls is AssetClass.INSULATOR_STRING,
            ice_type_review=ice_review,
        )

        icing = IcingResult(
            present=seg.ice_present,
            confidence=seg.confidence,
            coverage_ratio=None if coverage is None else round(coverage, 4),
            ice_type=ice_type,
            ice_type_conf=ice_type_conf,
            ice_type_source=ice_src,
            class_probabilities=ice_class_probs,
            requires_review=ice_review,
            review_reason=ice_reason,
            # 优先用 Track A 的桥接率（绝缘子），否则退回分割头给出的值
            bridge_ratio=(
                severity.bridge_ratio
                if severity.bridge_ratio is not None
                else (None if seg.bridge is None else round(seg.bridge, 4))
            ),
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
        ice_type: IceType | None,
        quality: QualityInfo,
        history: list[float],
        roi_id: str,
        history_timestamps: list[str] | None = None,
        baseline_mask: np.ndarray | None = None,
        baseline_iou: float | None = None,
        baseline_corr: float | None = None,
        is_insulator: bool = False,
        ice_type_review: bool = False,
    ) -> SeverityResult:
        outcome = resolve_calibration(
            calib,
            ptz_zoom_now=source.ptz.zoom if source and source.ptz else None,
            ptz_zoom_ref=None,
        )
        review = (
            outcome.requires_review or quality.status is QualityStatus.DEGRADED or ice_type_review
        )

        # ---- Track A：绝缘子视觉轨（不依赖标定，永远先算）----
        pa: InsulatorPaResult | None = None
        if is_insulator:
            pa = compute_insulator_pa(seg_mask, baseline_mask=baseline_mask, cfg=self.pa_cfg)
            if not pa.reliable:
                review = True

        if outcome.alpha_mm_per_px is None or calib.d0_mm is None:
            return SeverityResult(
                visual_grade=grade,
                alarm_level=None,
                measurement_status=outcome.status,
                calibration_source=outcome.source,
                requires_review=review,
                insulator_pa_pct=None if pa is None else _round(pa.pa_pct, 2),
                insulator_dist_v=None if pa is None else pa.vertical,
                insulator_dist_h=None if pa is None else pa.horizontal,
                bridge_ratio=None if pa is None else _round(pa.bridge_ratio, 4),
                baseline_matched_iou=baseline_iou,
                baseline_ecc_corr=baseline_corr,
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
                insulator_pa_pct=None if pa is None else _round(pa.pa_pct, 2),
                insulator_dist_v=None if pa is None else pa.vertical,
                insulator_dist_h=None if pa is None else pa.horizontal,
                bridge_ratio=None if pa is None else _round(pa.bridge_ratio, 4),
                baseline_matched_iou=baseline_iou,
                baseline_ecc_corr=baseline_corr,
            )

        rho, rho_src = self.rho.resolve(ice_type, requires_review=ice_type_review)
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
            # 变步长：有真实时间戳时按实际间隔回放（南网 1–6 h 采样）
            ekf.replay(history, timestamps=history_timestamps)
        # 本步 dt 取「最近一次历史时间戳 → 当前帧时间戳」
        step_dt = _step_dt_minutes(history_timestamps, source)
        state = ekf.update(est.d_eq_mm, dt_minutes=step_dt)

        # 桥接率：优先用绝缘子 Track A 的真实值，取代原先硬编码的 None
        bridge = None if pa is None else pa.bridge_ratio
        candidate = classify_level(
            self.alarm_rules,
            d_eq_mm=state.d_eq_mm,
            gamma=est.gamma,
            bridge=bridge,
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
            rho_kg_m3=round(rho, 1),
            rho_source=rho_src,
            requires_review=review or requires_review(level, self.alarm_rules),
            insulator_pa_pct=None if pa is None else _round(pa.pa_pct, 2),
            insulator_dist_v=None if pa is None else pa.vertical,
            insulator_dist_h=None if pa is None else pa.horizontal,
            bridge_ratio=None if pa is None else _round(pa.bridge_ratio, 4),
            baseline_matched_iou=baseline_iou,
            baseline_ecc_corr=baseline_corr,
            dt_minutes=None if state.dt_minutes <= 0 else round(state.dt_minutes, 1),
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
        model_ver = {
            "pipeline": _PIPELINE_VERSION,
            "precip": self.precip.model.backbone.spec.name,
            "detector": self.detector.model.backbone.spec.name,
            "seg": self.seg.model.backbone.spec.name,
        }
        if self.ice_type_service is not None:
            model_ver["ice_type"] = self.ice_type_service.model_version
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
                model_version=model_ver,
                config_hash=self._cfg_hash[:16],
                inference_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                rule_version=self.alarm_rules.version,
            ),
        )


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
