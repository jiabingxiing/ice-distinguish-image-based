"""分级阈值、质量门控、温度标定单测。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from grid_ice_vision.common.schemas import QualityStatus, VisualGrade
from grid_ice_vision.features.precipitation.calibrate import (
    TemperatureScaler,
    expected_calibration_error,
)
from grid_ice_vision.features.precipitation.prior import TemperaturePrior
from grid_ice_vision.features.quality.gate import (
    LensIcingTracker,
    QualityThresholds,
    assess_quality,
    normalized_blur_score,
)
from grid_ice_vision.features.severity.grading import (
    SeverityThresholds,
    coverage_upper_bound,
    grade_description,
    grade_from_coverage,
)


# ------------------------------------------------------------------ 分级
@pytest.mark.parametrize(
    ("cov", "grade"),
    [
        (0.0, "G0"),
        (0.02, "G0"),
        (0.03, "G1"),
        (0.14, "G1"),
        (0.15, "G2"),
        (0.29, "G2"),
        (0.30, "G3"),
        (0.49, "G3"),
        (0.50, "G4"),
        (0.95, "G4"),
    ],
)
def test_grade_boundaries(cov: float, grade: str) -> None:
    assert grade_from_coverage(cov) == VisualGrade(grade)


def test_grade_from_config() -> None:
    th = SeverityThresholds.from_config(
        {
            "grades": {
                "G0": {"max_coverage": 0.05},
                "G1": {"max_coverage": 0.25},
                "G2": {"max_coverage": None},
            }
        }
    )
    assert th.grade(0.04) == VisualGrade.G0
    assert th.grade(0.20) == VisualGrade.G1
    assert th.grade(0.90) == VisualGrade.G2


def test_grade_config_empty_falls_back() -> None:
    assert SeverityThresholds.from_config({}).grade(0.02) == VisualGrade.G0


def test_grade_description_and_bounds() -> None:
    assert grade_description(VisualGrade.G3)
    assert coverage_upper_bound(VisualGrade.G0) == pytest.approx(0.03)
    assert coverage_upper_bound(VisualGrade.G4) is None


# ------------------------------------------------------------------ 质量门控
def _textured(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_sharp_image_usable() -> None:
    q = assess_quality(_textured())
    assert q.status in (QualityStatus.USABLE, QualityStatus.DEGRADED)
    assert q.blur_score < 0.6


def test_blurred_image_flagged() -> None:
    img = cv2.GaussianBlur(_textured(), (0, 0), 8.0)
    q = assess_quality(img)
    assert q.blur_score > 0.6
    assert any("blur" in r for r in q.reasons)


def test_night_image_unusable() -> None:
    dark = np.full((120, 160, 3), 8, dtype=np.uint8)
    q = assess_quality(dark)
    assert q.status is QualityStatus.UNUSABLE
    assert any("night" in r for r in q.reasons)


def test_overexposed_flagged() -> None:
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    q = assess_quality(img)
    assert q.exposure.value == "over"
    assert q.status is QualityStatus.UNUSABLE


def test_blur_score_monotonic() -> None:
    sharp = normalized_blur_score(cv2.cvtColor(_textured(), cv2.COLOR_BGR2GRAY))
    blurred = normalized_blur_score(
        cv2.GaussianBlur(cv2.cvtColor(_textured(), cv2.COLOR_BGR2GRAY), (0, 0), 6.0)
    )
    assert blurred > sharp


def test_lens_icing_tracker_triggers() -> None:
    tr = LensIcingTracker(frames=3, window=5, energy_floor=0.05)
    energy = np.ones((6, 8), dtype=np.float32)
    energy[2, 3] = 0.01
    assert tr.update("C01", energy) is False
    assert tr.update("C01", energy) is False
    assert tr.update("C01", energy) is True


def test_thresholds_from_config() -> None:
    th = QualityThresholds.from_config({"quality": {"blur_degraded": 0.5, "grid": [3, 4]}})
    assert th.blur_degraded == 0.5
    assert th.grid == (3, 4)


# ------------------------------------------------------------------ 校准
def test_temperature_scaler_identity_when_unfitted() -> None:
    sc = TemperatureScaler()
    logits = np.array([[2.0, -1.0]])
    assert np.allclose(sc.transform(logits), logits)
    assert sc.temperature == 1.0


def test_temperature_scaler_reduces_overconfidence() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 4, size=(500, 1))
    targets = (rng.random((500, 1)) < 0.5).astype(float)
    sc = TemperatureScaler().fit(logits, targets)
    assert sc.fitted
    assert sc.temperature > 1.0  # 过自信 → T>1 降温
    ece_before = expected_calibration_error(1 / (1 + np.exp(-logits)), targets)
    ece_after = expected_calibration_error(sc.predict_proba(logits), targets)
    assert ece_after <= ece_before + 1e-6


def test_ece_perfect_calibration() -> None:
    probs = np.full(100, 0.7)
    targets = np.zeros(100)
    targets[:70] = 1.0
    assert expected_calibration_error(probs, targets) == pytest.approx(0.0, abs=0.02)


def test_prior_shifts_logits_toward_cold_classes() -> None:
    import torch

    prior = TemperaturePrior()
    logits = torch.zeros(1, 8)
    cold = prior.apply(logits, -15.0, lam=1.0)[0].numpy()
    warm = prior.apply(logits, 25.0, lam=1.0)[0].numpy()
    assert cold[6] > warm[6]  # rime
    assert warm[1] > cold[1]  # rain


def test_prior_noop_when_no_temperature() -> None:
    import torch

    prior = TemperaturePrior()
    logits = torch.zeros(1, 8)
    assert torch.equal(prior.apply(logits, None, 1.0), logits)
    assert torch.equal(prior.apply(logits, -5.0, 0.0), logits)


def test_prior_bin_index_bounds() -> None:
    prior = TemperaturePrior()
    assert prior.bin_index(-100) == 0
    assert prior.bin_index(100) == len(prior.table) - 1
