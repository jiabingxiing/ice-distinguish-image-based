"""全链路契约测试（§15.1 集成层）。

无权重、无真实数据也能跑：用随机初始化的模型 + 打桩的检测/分割输出，
验证的是**编排与契约**，不是精度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from grid_ice_vision.application.pipeline import InferencePipeline
from grid_ice_vision.common.config import load_config
from grid_ice_vision.common.schemas import (
    Asset,
    AssetClass,
    CalibInput,
    MeasurementStatus,
    QualityStatus,
    SeverityResult,
    SourceInfo,
)
from grid_ice_vision.features.ice.infer import SegPrediction, decode_rle, encode_rle


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> InferencePipeline:
    cfg = load_config("configs/model/pipeline.yaml")
    cfg["infer"]["detector"]["tile"] = 320
    cfg["infer"]["detector"]["score_thr"] = 0.99
    cfg["evidence_dir"] = str(tmp_path_factory.mktemp("evidence"))
    return InferencePipeline.from_config(cfg, device="cpu")


def _scene(h: int = 320, w: int = 480) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(70, 190, (h, w, 3), dtype=np.uint8)
    return img


def _fake_asset() -> list[Asset]:
    return [
        Asset(cls=AssetClass.CONDUCTOR, bbox=(40.0, 120.0, 440.0, 200.0), score=0.95, roi_id="r0")
    ]


def _fake_seg(h: int = 80, w: int = 400) -> SegPrediction:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[36:44, :] = 1  # 裸露导线 8 px
    mask[32:36, :] = 2  # 上侧冰 4 px
    mask[44:48, :] = 2  # 下侧冰 4 px
    probs = np.zeros((4, h, w), dtype=np.float32)
    for c in range(4):
        probs[c] = (mask == c).astype(np.float32) * 0.8 + 0.05
    ice = int(((mask == 2) | (mask == 3)).sum())
    exp = int((mask == 1).sum())
    return SegPrediction(
        mask=mask,
        probs=probs,
        coverage=ice / (ice + exp),
        ice_present=True,
        confidence=0.88,
        bridge=None,
        roi_bbox=(40, 120, 440, 200),
    )


def _run(pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch, calib: CalibInput) -> Any:
    monkeypatch.setattr(pipeline.detector, "predict", lambda img: _fake_asset())
    monkeypatch.setattr(pipeline.seg, "predict", lambda *a, **k: _fake_seg())
    src = SourceInfo(site_id="S01", camera_id="C01", timestamp="2026-01-15T08:30:12+08:00")
    return pipeline.run(_scene(), "S01_C01_20260115T083012", src, calib, write_evidence=False)


def test_measured_path_produces_thickness(
    pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(
        pipeline,
        monkeypatch,
        CalibInput(d0_mm=30.0, w0_px=8.0, d_ref_mm=30.0, w_ref_px=8.0, design_ice_mm=20.0),
    )
    assert result.quality.status is not QualityStatus.UNUSABLE
    assert result.icing.present is True
    assert result.icing.coverage_ratio == pytest.approx(0.5, abs=0.02)
    assert result.severity.measurement_status is MeasurementStatus.MEASURED
    assert result.severity.d_eq_mm is not None and result.severity.d_eq_mm > 0
    assert result.severity.visual_grade.value == "G4"
    assert result.severity.alpha_mm_per_px == pytest.approx(30.0 / 8.0, rel=1e-3)


def test_not_calibrated_forbids_thickness(
    pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(pipeline, monkeypatch, CalibInput())
    assert result.severity.measurement_status is MeasurementStatus.NOT_CALIBRATED
    assert result.severity.d_eq_mm is None
    assert result.severity.d_radial_mm is None
    assert result.severity.alarm_level is None
    # 覆盖率与等级仍然可用
    assert result.icing.coverage_ratio is not None


def test_unusable_image_short_circuits(pipeline: InferencePipeline) -> None:
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    src = SourceInfo(site_id="S01", camera_id="C01", timestamp="2026-01-15T08:30:12+08:00")
    result = pipeline.run(
        dark, "dark", src, CalibInput(d0_mm=30.0, w0_px=8.0), write_evidence=False
    )
    assert result.quality.status is QualityStatus.UNUSABLE
    assert result.icing.present is None
    assert result.severity.d_eq_mm is None
    assert result.precipitation.intensity.value == "unknown"
    assert "rejected" in result.precipitation.note


def test_no_asset_keeps_weather_only(
    pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.detector, "predict", lambda img: [])
    src = SourceInfo(site_id="S01", camera_id="C01", timestamp="2026-01-15T08:30:12+08:00")
    result = pipeline.run(_scene(), "noasset", src, CalibInput(), write_evidence=False)
    assert result.assets == []
    assert result.icing.present is None
    assert result.precipitation.labels  # 天气结论仍输出


def test_result_json_roundtrip(
    pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(
        pipeline, monkeypatch, CalibInput(d0_mm=30.0, w0_px=8.0, d_ref_mm=30.0, w_ref_px=8.0)
    )
    payload = result.to_json_dict()
    assert payload["schema_version"] == "1.0.0"
    assert payload["provenance"]["rule_version"] == "1.0.0"
    assert payload["provenance"]["inference_ms"] >= 0
    assert isinstance(payload["icing"]["mask_rle"]["counts"], str)
    assert set(payload) >= {
        "quality",
        "precipitation",
        "assets",
        "icing",
        "severity",
        "evidence",
        "provenance",
    }


def test_rle_roundtrip() -> None:
    rng = np.random.default_rng(0)
    mask = (rng.random((37, 53)) > 0.6).astype(np.uint8)
    rle = encode_rle(mask)
    assert np.array_equal(decode_rle(rle), mask)


def test_evidence_images_written(
    tmp_path: Path, pipeline: InferencePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.detector, "predict", lambda img: _fake_asset())
    monkeypatch.setattr(pipeline.seg, "predict", lambda *a, **k: _fake_seg())
    pipeline.cfg.evidence_dir = str(tmp_path)
    src = SourceInfo(site_id="S01", camera_id="C01", timestamp="2026-01-15T08:30:12+08:00")
    result = pipeline.run(
        _scene(), "evid", src, CalibInput(d0_mm=30.0, w0_px=8.0), write_evidence=True
    )
    assert "mask_overlay" in result.evidence
    assert Path(result.evidence["mask_overlay"]).exists()
    assert Path(result.evidence["heatmap"]).exists()


def test_severity_guard_still_holds() -> None:
    with pytest.raises(ValueError):
        SeverityResult(measurement_status=MeasurementStatus.NOT_CALIBRATED, d_eq_mm=1.0)
