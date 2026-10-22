"""覆冰类型与密度策略单测（A8）。

这里同时**记录一个风险**：若接入随机权重的类型头，rime(ρ=350) 会让
等值冰厚缩水约 58%。这不是假设，是可算出来的数。
"""

from __future__ import annotations

import pytest

from grid_ice_vision.common.schemas import IceType, VisualGrade
from grid_ice_vision.features.severity.ice_type import (
    RhoPolicy,
    ice_type_confidence,
    resolve_ice_type,
)
from grid_ice_vision.features.severity.thickness import equivalent_thickness_mm


def test_g0_is_none_with_rule_source() -> None:
    t, src = resolve_ice_type(VisualGrade.G0)
    assert t is IceType.NONE
    assert src == "none_rule"
    assert ice_type_confidence(src) == 1.0


@pytest.mark.parametrize("grade", ["G1", "G2", "G3", "G4"])
def test_nonzero_grade_never_guesses(grade: str) -> None:
    t, src = resolve_ice_type(VisualGrade(grade))
    assert t is IceType.UNKNOWN
    assert src == "unavailable"
    assert ice_type_confidence(src) == 0.0


def test_unknown_and_none_use_standard_density() -> None:
    pol = RhoPolicy()
    assert pol.resolve(IceType.UNKNOWN) == (900.0, "standard")
    assert pol.resolve(IceType.NONE) == (900.0, "standard")
    assert pol.resolve(None) == (900.0, "standard")


def test_known_type_uses_config_prior() -> None:
    pol = RhoPolicy()
    rho, src = pol.resolve(IceType.RIME)
    assert rho == 350.0
    assert src == "config_prior"


def test_from_config_overrides() -> None:
    pol = RhoPolicy.from_config(
        {
            "severity": {
                "rho": {"default": 880, "by_type": {"rime": 400}},
                "ice_type": {"mode": "none_rule"},
            }
        }
    )
    assert pol.resolve(IceType.RIME)[0] == 400.0
    assert pol.resolve(IceType.UNKNOWN)[0] == 880.0


def test_unimplemented_mode_is_rejected() -> None:
    """heuristic/model 未验证前不允许打开，避免"用阈值猜"冒充能力。"""
    with pytest.raises(ValueError, match="尚未实现"):
        RhoPolicy(mode="heuristic")
    with pytest.raises(ValueError, match="尚未实现"):
        RhoPolicy(mode="model")


def test_nonpositive_default_rejected() -> None:
    with pytest.raises(ValueError, match="密度必须为正"):
        RhoPolicy(default=0.0)


def test_rime_density_would_shrink_equivalent_thickness() -> None:
    """d0=30mm、径向 5mm：ρ 900 → 5.00mm；ρ 350 → 2.12mm（-57.6%）。"""
    assert equivalent_thickness_mm(5.0, 30.0, 900.0) == pytest.approx(5.00, abs=0.01)
    assert equivalent_thickness_mm(5.0, 30.0, 350.0) == pytest.approx(2.12, abs=0.01)


def test_ice_type_net_is_not_wired_into_pipeline() -> None:
    """防回归：没有 ice_type 标注前，pipeline 不得引用 IceTypeGradeNet。"""
    import inspect

    import grid_ice_vision.application.pipeline as pipeline_mod

    assert "IceTypeGradeNet" not in inspect.getsource(pipeline_mod)


def test_pipeline_config_keeps_ice_type_off() -> None:
    """防回归：配置文件层面也必须保持关闭。

    只查源码不够——改一行 YAML 就能把 `mode` 改成 `model` 而绕过上面的检查；
    这里从**真实配置**构造策略，等价于给生产配置加一道门禁（ADR-0003 G4）。
    """
    from grid_ice_vision.common.config import get, load_config

    cfg = load_config("configs/model/pipeline.yaml")
    assert get(cfg, "severity.ice_type.mode") == "none_rule"
    pol = RhoPolicy.from_config(cfg)
    assert pol.mode == "none_rule"


def test_no_path_yields_model_rho_source() -> None:
    """`ice_type_model` 只能由已验证的类型头产生；现阶段出现即为缺陷。"""
    pol = RhoPolicy()
    for t in (IceType.NONE, IceType.UNKNOWN, IceType.GLAZE, IceType.RIME, IceType.MIXED):
        rho, src = pol.resolve(t)
        assert src != "ice_type_model"
        assert rho > 0
