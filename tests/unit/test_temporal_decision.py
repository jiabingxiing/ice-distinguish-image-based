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


# ---------------------------------------------------------------- 变步长 EKF
def test_ekf_variable_dt_is_recorded() -> None:
    """传入的真实间隔必须回写到状态，供上层审计。"""
    ekf = ThicknessEKF(EKFConfig())
    ekf.update(5.0)
    st = ekf.update(5.2, dt_minutes=180.0)
    assert st.dt_minutes == pytest.approx(180.0)


def test_ekf_first_frame_dt_is_zero() -> None:
    ekf = ThicknessEKF()
    st = ekf.update(5.0, dt_minutes=240.0)
    assert st.dt_minutes == 0.0  # 初始化不推进时间


def test_ekf_none_dt_falls_back_to_config() -> None:
    """向后兼容：不传 dt 时用 cfg.dt_minutes。"""
    ekf = ThicknessEKF(EKFConfig(dt_minutes=15.0))
    ekf.update(5.0)
    st = ekf.update(5.1)
    assert st.dt_minutes == pytest.approx(15.0)


def test_ekf_dt_is_clamped() -> None:
    """过小/过大的间隔被夹取，避免退化的 F 矩阵。"""
    from grid_ice_vision.application.temporal import MAX_DT_MINUTES, MIN_DT_MINUTES

    ekf = ThicknessEKF()
    ekf.update(5.0)
    assert ekf.update(5.1, dt_minutes=0.0).dt_minutes == pytest.approx(MIN_DT_MINUTES)
    assert ekf.update(5.2, dt_minutes=1e6).dt_minutes == pytest.approx(MAX_DT_MINUTES)


def test_ekf_rejects_non_finite_dt() -> None:
    ekf = ThicknessEKF()
    ekf.update(5.0)
    with pytest.raises(ValueError, match="有限"):
        ekf.update(5.1, dt_minutes=float("nan"))


def test_ekf_variable_dt_estimates_growth_rate_correctly() -> None:
    """真实 2 小时长 1 mm ⇒ 增长率收敛到 0.5 mm/h（6 步后实测 0.499）。"""
    ekf = ThicknessEKF(EKFConfig(growth_rate_max_mm_per_h=50.0, measurement_noise_mm=0.2))
    st = None
    for i in range(7):
        st = ekf.update(5.0 + 1.0 * i, dt_minutes=120.0 if i else None)
    assert st is not None
    assert st.growth_mm_per_h == pytest.approx(0.5, abs=0.08)


def test_ekf_fixed_dt_overestimates_growth_by_order() -> None:
    """核心回归：同样每步 +1 mm，误用固定 15 min 会把增长率高估近 8 倍。

    实测 15 min → 3.915 mm/h，120 min → 0.499 mm/h（比值 7.8）。
    这正是原 `dt_minutes=15` 硬编码在 1–6 h 真实采样下的失真量。
    """
    def run(dt: float) -> float:
        ekf = ThicknessEKF(EKFConfig(growth_rate_max_mm_per_h=50.0, measurement_noise_mm=0.2))
        st = None
        for i in range(7):
            st = ekf.update(5.0 + 1.0 * i, dt_minutes=dt if i else None)
        assert st is not None
        return st.growth_mm_per_h

    rate_fixed = run(15.0)
    rate_true = run(120.0)
    assert rate_fixed > rate_true * 5.0
    assert rate_fixed == pytest.approx(3.915, abs=0.2)
    assert rate_true == pytest.approx(0.499, abs=0.05)


def test_ekf_growth_rate_scales_inverse_with_dt() -> None:
    """同一观测斜率下，增长率应与 Δt 成反比（物理一致性）。"""
    def run(dt: float) -> float:
        ekf = ThicknessEKF(EKFConfig(growth_rate_max_mm_per_h=50.0, measurement_noise_mm=0.2))
        st = None
        for i in range(7):
            st = ekf.update(5.0 + 1.0 * i, dt_minutes=dt if i else None)
        assert st is not None
        return st.growth_mm_per_h

    r120, r360 = run(120.0), run(360.0)
    assert r120 / r360 == pytest.approx(3.0, abs=0.3)


def test_ekf_replay_with_timestamps() -> None:
    """带时间戳回放：按真实间隔推进，末态与逐帧调用一致。"""
    ekf = ThicknessEKF(EKFConfig(measurement_noise_mm=0.3, growth_rate_max_mm_per_h=10.0))
    ts = ["2026-01-01T00:00:00", "2026-01-01T03:00:00", "2026-01-01T06:00:00"]
    st = ekf.replay([5.0, 5.6, 6.4], timestamps=ts)
    assert st.updated
    assert st.d_eq_mm > 5.5


def test_ekf_replay_length_mismatch_raises() -> None:
    ekf = ThicknessEKF()
    with pytest.raises(ValueError, match="长度"):
        ekf.replay([5.0, 6.0], timestamps=["2026-01-01T00:00:00"])


def test_ekf_replay_without_timestamps_is_backward_compatible() -> None:
    """旧调用 `replay(history)` 必须仍可用。"""
    ekf = ThicknessEKF()
    st = ekf.replay([5.0, 5.2, 5.4])
    assert st.updated


def test_ekf_replay_bad_timestamp_degrades_to_none_dt() -> None:
    """无法解析的时间戳退化为固定步长，不抛异常打断整段回放。"""
    ekf = ThicknessEKF()
    st = ekf.replay([5.0, 5.1], timestamps=["not-a-time", "also-bad"])
    assert st.updated
