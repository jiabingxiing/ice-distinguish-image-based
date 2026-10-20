"""统一数据契约（Schema v1）。

设计要点（见 IMPLEMENTATION_PLAN §7.2）：
- `measurement_status` 三态是核心：未标定不得输出厚度与预警；
- 天气结论与覆冰结论**物理隔离**，不得互相推导；
- 所有可选字段给默认值，新增字段不升主版本。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grid_ice_vision.common.constants import SCHEMA_VERSION


class QualityStatus(StrEnum):
    USABLE = "usable"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    VISUAL_ONLY = "visual_only"
    NOT_CALIBRATED = "not_calibrated"


class Exposure(StrEnum):
    UNDER = "under"
    NORMAL = "normal"
    OVER = "over"


class IceType(StrEnum):
    NONE = "none"
    GLAZE = "glaze"
    RIME = "rime"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class VisualGrade(StrEnum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"


class AlarmLevel(StrEnum):
    I = "I"  # noqa: E741 - 预警等级用罗马数字，属领域命名
    II = "II"
    III = "III"
    IV = "IV"


class AssetClass(StrEnum):
    CONDUCTOR = "conductor"
    INSULATOR_STRING = "insulator_string"
    TOWER = "tower"
    REFERENCE_FITTING = "reference_fitting"


class Intensity(StrEnum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------- 输入
class PTZ(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zoom: float = 1.0
    pan: float = 0.0
    tilt: float = 0.0


class WeatherSensor(BaseModel):
    """现场气象传感器读数；缺失时整体为 None，不做任何插补。"""

    model_config = ConfigDict(extra="forbid")

    t_amb_c: float
    rh: float = Field(ge=0.0, le=100.0)
    v_wind_ms: float = Field(ge=0.0)
    p_atm_hpa: float = 1013.0


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: str
    camera_id: str
    asset_id: str = ""
    timestamp: str
    ptz: PTZ | None = None
    weather_sensor: WeatherSensor | None = None


class CalibInput(BaseModel):
    """标定输入。任何一项缺失都不影响推理，只影响 measurement_status。"""

    model_config = ConfigDict(extra="forbid")

    d0_mm: float | None = Field(default=None, gt=0.0, description="导线外径 / 绝缘子盘径")
    design_ice_mm: float | None = Field(default=None, gt=0.0, description="设计冰厚")
    w0_px: float | None = Field(default=None, gt=0.0, description="晴天建档的裸导线像素宽度")
    d_ref_mm: float | None = Field(default=None, gt=0.0, description="刚性金具已知物理尺寸")
    w_ref_px: float | None = Field(default=None, gt=0.0, description="刚性金具像素宽度")
    alpha_mm_per_px: float | None = Field(default=None, gt=0.0, description="外部给定的尺度因子")


# --------------------------------------------------------------------- 输出
class QualityInfo(BaseModel):
    status: QualityStatus
    blur_score: float = Field(ge=0.0, le=1.0)
    exposure: Exposure = Exposure.NORMAL
    occlusion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    lens_icing_suspected: bool = False
    reasons: list[str] = Field(default_factory=list)


class PrecipResult(BaseModel):
    """视觉相态：多标签概率，**不代表**气象确认结论。"""

    labels: dict[str, float] = Field(default_factory=dict)
    intensity: Intensity = Intensity.UNKNOWN
    intensity_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    meteorological_input_used: bool = False
    note: str = "visual_only"

    @field_validator("labels")
    @classmethod
    def _check_labels(cls, v: dict[str, float]) -> dict[str, float]:
        for k, p in v.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"标签概率越界: {k}={p}")
        return v


class Asset(BaseModel):
    cls: AssetClass
    bbox: tuple[float, float, float, float]
    score: float = Field(ge=0.0, le=1.0)
    roi_id: str
    reference_width_px: float | None = None

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox 非法: {v}")
        return v


class MaskRLE(BaseModel):
    size: tuple[int, int]
    counts: str


class IcingResult(BaseModel):
    present: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    ice_type: IceType = IceType.UNKNOWN
    ice_type_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    # 类型是怎么来的：规则 / 启发式 / 模型 / 无（不猜）
    ice_type_source: Literal["none_rule", "heuristic", "model", "unavailable"] = "unavailable"
    bridge_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_rle: MaskRLE | None = None


class SeverityResult(BaseModel):
    visual_grade: VisualGrade = VisualGrade.G0
    alarm_level: AlarmLevel | None = None
    d_radial_mm: float | None = None
    d_eq_mm: float | None = None
    gamma: float | None = None
    measurement_status: MeasurementStatus = MeasurementStatus.NOT_CALIBRATED
    calibration_source: Literal["rigid_fitting", "ptz", "baseline_archive", "external"] | None = (
        None
    )
    alpha_mm_per_px: float | None = None
    uncertainty_mm: float | None = None
    # 密度折算必须可追溯：用了哪个 ρ、依据是什么，否则事后无法解释厚度
    rho_kg_m3: float | None = Field(default=None, gt=0.0)
    rho_source: Literal["standard", "config_prior", "ice_type_model"] = "standard"
    requires_review: bool = False

    @model_validator(mode="after")
    def _enforce_measured_gate(self) -> Self:
        """硬约束：未标定不得输出厚度与预警等级。

        用 raise 而非 assert，避免 `python -O` 下被剥离。
        """
        if self.measurement_status is MeasurementStatus.NOT_CALIBRATED:
            if self.d_eq_mm is not None:
                raise ValueError("not_calibrated 时不得输出 d_eq_mm")
            if self.d_radial_mm is not None:
                raise ValueError("not_calibrated 时不得输出 d_radial_mm")
            if self.alarm_level is not None:
                raise ValueError("not_calibrated 时不得输出 alarm_level")
        if (
            self.alarm_level is not None
            and self.measurement_status is not MeasurementStatus.MEASURED
        ):
            raise ValueError("alarm_level 仅在 measurement_status=measured 时允许")
        return self


class Provenance(BaseModel):
    model_version: dict[str, str] = Field(default_factory=dict)
    config_hash: str = ""
    inference_ms: float = 0.0
    rule_version: str = ""


class InferenceResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    image_id: str
    source: SourceInfo | None = None
    quality: QualityInfo
    precipitation: PrecipResult
    assets: list[Asset] = Field(default_factory=list)
    icing: IcingResult = Field(default_factory=IcingResult)
    severity: SeverityResult = Field(default_factory=SeverityResult)
    evidence: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class InferenceRequest(BaseModel):
    """服务端请求体：图像以 base64 或 multipart 提供，元数据随请求下发。"""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    source: SourceInfo
    calib: CalibInput = Field(default_factory=CalibInput)
    history_d_eq_mm: list[float] = Field(
        default_factory=list, description="同机位历史厚度序列（升序时间）"
    )
