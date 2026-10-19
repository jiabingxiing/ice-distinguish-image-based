"""EKF 与预警规则单测（§10.7、§15.1）。"""

from __future__ import annotations

import pytest

from grid_ice_vision.application.decision import (
    AlarmRules,
    AlarmStateMachine,
    classify_level,
    requires_review,
)
from grid_ice_vision.application.temporal import EKFConfig, ThicknessEKF
from grid_ice_vision.common.schemas import AlarmLevel, MeasurementStatus


# ------------------------------------------------------------------ EKF
def test_ekf_initializes_from_first_measurement() -> None:
    ekf = ThicknessEKF()
    st = ekf.update(6.0)
    assert st.updated
    assert st.d_eq_mm == pytest.approx(6.0)
    assert st.growth_mm_per_h == 0.0


def test_ekf_rejects_single_frame_spike() -> None:
    """注入 +10 mm 单帧尖峰，滤波输出抖动应 < 2 mm。"""
    ekf = ThicknessEKF(EKFConfig(measurement_noise_mm=1.5))
    for _ in range(4):
        ekf.update(6.0)
    st = ekf.update(16.0)  # 飞掠雪花造成的尖峰
    assert st.outlier_rejected
    assert abs(st.d_eq_mm - 6.0) < 2.0


def test_ekf_tracks_slow_growth() -> None:
    ekf = ThicknessEKF(EKFConfig(growth_rate_max_mm_per_h=3.0, dt_minutes=60))
    for i in range(6):
        st = ekf.update(5.0 + 1.0 * i)
    assert 8.0 < st.d_eq_mm < 12.0


def test_ekf_clamps_unphysical_rate() -> None:
    ekf = ThicknessEKF(
        EKFConfig(growth_rate_max_mm_per_h=1.0, dt_minutes=15, innovation_gate_sigma=10.0)
    )
    ekf.update(1.0)
    st = ekf.update(2.0)
    assert abs(st.growth_mm_per_h) <= 1.0 + 1e-9


def test_ekf_missing_measurement_only_predicts() -> None:
    ekf = ThicknessEKF()
    ekf.update(6.0)
    st = ekf.update(None)
    assert st.updated is False
    assert st.d_eq_mm == pytest.approx(6.0, abs=0.5)


def test_ekf_replay_history() -> None:
    ekf = ThicknessEKF()
    st = ekf.replay([5.0, 5.2, 5.4, 5.6])
    assert st.d_eq_mm > 5.0


# ------------------------------------------------------------------ 预警规则
def _rules() -> AlarmRules:
    return AlarmRules(
        d_eq_ranges={
            "I": (0.001, 5.0),
            "II": (5.0, 10.0),
            "III": (10.0, 20.0),
            "IV": (20.0, None),
        },
        gamma_ranges={"I": (0.0, 0.3), "II": (0.3, 0.5), "III": (0.5, 0.8), "IV": (0.8, None)},
        bridge_ranges={"I": (0.0, 0.2), "II": (0.2, 0.5), "III": (0.5, 0.8), "IV": (0.8, None)},
    )


@pytest.mark.parametrize(
    ("d_eq", "expected"),
    [(1.0, "I"), (4.99, "I"), (5.0, "II"), (9.9, "II"), (10.0, "III"), (20.0, "IV"), (50.0, "IV")],
)
def test_classify_level_by_thickness(d_eq: float, expected: str) -> None:
    assert classify_level(_rules(), d_eq) == AlarmLevel(expected)


def test_classify_uses_highest_hit() -> None:
    """d_eq=3mm 命中 I，但 bridge=0.6 命中 III → 取最高。"""
    assert classify_level(_rules(), 3.0, bridge=0.6) == AlarmLevel("III")


def test_classify_blocked_when_not_measured() -> None:
    assert classify_level(_rules(), 25.0, measurement_status=MeasurementStatus.VISUAL_ONLY) is None
    assert (
        classify_level(_rules(), 25.0, measurement_status=MeasurementStatus.NOT_CALIBRATED) is None
    )


def test_classify_none_thickness() -> None:
    assert classify_level(_rules(), None) is None


def test_debounce_escalation() -> None:
    sm = AlarmStateMachine(_rules())
    assert sm.update("k", AlarmLevel("I")) == AlarmLevel("I")
    assert sm.update("k", AlarmLevel("II")) == AlarmLevel("I")  # 第 1 帧
    assert sm.update("k", AlarmLevel("II")) == AlarmLevel("I")  # 第 2 帧
    assert sm.update("k", AlarmLevel("II")) == AlarmLevel("II")  # 第 3 帧升级


def test_debounce_deescalation_needs_more_frames() -> None:
    sm = AlarmStateMachine(_rules())
    sm.update("k", AlarmLevel("III"))
    for _ in range(4):
        assert sm.update("k", AlarmLevel("II")) == AlarmLevel("III")
    assert sm.update("k", AlarmLevel("II")) == AlarmLevel("II")


def test_debounce_resets_on_flip() -> None:
    sm = AlarmStateMachine(_rules())
    sm.update("k", AlarmLevel("I"))
    sm.update("k", AlarmLevel("II"))
    sm.update("k", AlarmLevel("III"))  # 计数被打断
    assert sm.update("k", AlarmLevel("II")) == AlarmLevel("I")


def test_state_machine_clears_when_not_measured() -> None:
    sm = AlarmStateMachine(_rules())
    sm.update("k", AlarmLevel("III"))
    assert sm.update("k", AlarmLevel("III"), MeasurementStatus.VISUAL_ONLY) is None


def test_requires_review_for_high_levels() -> None:
    rules = _rules()
    assert requires_review(AlarmLevel("III"), rules) is True
    assert requires_review(AlarmLevel("IV"), rules) is True
    assert requires_review(AlarmLevel("I"), rules) is False
    assert requires_review(None, rules) is False
