"""契约单测：三态门禁、bbox、标签范围（§7.2、§15.1）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grid_ice_vision.common.schemas import (
    Asset,
    AssetClass,
    CalibInput,
    IcingResult,
    InferenceResult,
    MeasurementStatus,
    PrecipResult,
    QualityInfo,
    QualityStatus,
    SeverityResult,
    SourceInfo,
    VisualGrade,
)


def test_not_calibrated_forbids_thickness() -> None:
    with pytest.raises(ValidationError):
        SeverityResult(measurement_status=MeasurementStatus.NOT_CALIBRATED, d_eq_mm=5.0)


def test_not_calibrated_forbids_alarm() -> None:
    with pytest.raises(ValidationError):
        SeverityResult(measurement_status=MeasurementStatus.NOT_CALIBRATED, alarm_level="II")


def test_visual_only_forbids_alarm() -> None:
    with pytest.raises(ValidationError):
        SeverityResult(measurement_status=MeasurementStatus.VISUAL_ONLY, alarm_level="I")


def test_measured_allows_alarm() -> None:
    s = SeverityResult(
        measurement_status=MeasurementStatus.MEASURED,
        d_eq_mm=12.0,
        alarm_level="III",
        visual_grade=VisualGrade.G3,
    )
    assert s.alarm_level == "III"


def test_visual_only_allows_radial_thickness() -> None:
    s = SeverityResult(measurement_status=MeasurementStatus.VISUAL_ONLY, d_radial_mm=1.2)
    assert s.d_eq_mm is None


def test_asset_bbox_validation() -> None:
    with pytest.raises(ValidationError):
        Asset(cls=AssetClass.CONDUCTOR, bbox=(10, 10, 5, 5), score=0.9, roi_id="r0")


def test_asset_score_range() -> None:
    with pytest.raises(ValidationError):
        Asset(cls=AssetClass.CONDUCTOR, bbox=(0, 0, 10, 10), score=1.5, roi_id="r0")


def test_precip_labels_range() -> None:
    with pytest.raises(ValidationError):
        PrecipResult(labels={"rain": 1.4})


def test_precip_defaults_are_visual_only() -> None:
    p = PrecipResult()
    assert p.note == "visual_only"
    assert p.meteorological_input_used is False


def test_icing_defaults_unknown() -> None:
    i = IcingResult()
    assert i.present is None
    assert i.ice_type.value == "unknown"


def test_calib_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        CalibInput(d0_mm=0.0)


def test_full_result_serialization() -> None:
    result = InferenceResult(
        image_id="S03_C07_20260115T083012",
        source=SourceInfo(site_id="S03", camera_id="C07", timestamp="2026-01-15T08:30:12+08:00"),
        quality=QualityInfo(status=QualityStatus.USABLE, blur_score=0.1),
        precipitation=PrecipResult(labels={"rain": 0.9}),
        icing=IcingResult(present=True, confidence=0.9, coverage_ratio=0.24),
        severity=SeverityResult(measurement_status=MeasurementStatus.MEASURED, d_eq_mm=7.1),
    )
    d = result.to_json_dict()
    assert d["schema_version"] == "1.0.0"
    assert d["severity"]["d_eq_mm"] == 7.1
    assert d["icing"]["ice_type"] == "unknown"


def test_schema_version_present() -> None:
    assert InferenceResult.model_fields["schema_version"].default == "1.0.0"
