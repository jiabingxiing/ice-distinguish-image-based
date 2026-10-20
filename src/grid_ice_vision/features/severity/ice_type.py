"""覆冰类型与密度折算策略（§10.5、§10.6）。

为什么不接 `IceTypeGradeNet`：没有 `ice_type` 标注数据时，随机权重的类型头
约 2/5 概率输出 `rime`（ρ=350），等值冰厚会缩水约 58%（d0=30 mm、径向 5 mm：
5.00 → 2.12 mm），并直接跨级改变预警；而契约里若没有字段记录"密度是猜的"，
事后无法审计。

因此当前只做两件事：
1. `resolve_ice_type`：G0 → none，其余 → unknown（**不猜**）；
2. `RhoPolicy`：把"用了哪个密度、为什么"写进结果，为将来接模型留好接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from grid_ice_vision.common.constants import RHO_BY_ICE_TYPE, RHO_STANDARD
from grid_ice_vision.common.schemas import IceType, VisualGrade

IceTypeSource = Literal["none_rule", "heuristic", "model", "unavailable"]
RhoSource = Literal["standard", "config_prior", "ice_type_model"]

_VALID_MODES = ("none_rule",)


@dataclass(frozen=True)
class RhoPolicy:
    """覆冰密度取值策略。"""

    by_type: dict[str, float] = field(default_factory=lambda: dict(RHO_BY_ICE_TYPE))
    default: float = RHO_STANDARD
    mode: str = "none_rule"

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"ice_type.mode={self.mode!r} 尚未实现/未验证。"
                f"当前仅支持 {_VALID_MODES}；启用 heuristic 前需 ≥200 个带类型的 ROI 验证。"
            )
        if self.default <= 0:
            raise ValueError("default 密度必须为正")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> RhoPolicy:
        """从 `configs/model/pipeline.yaml` 的 `severity.rho` / `severity.ice_type` 读取。"""
        sec = (cfg or {}).get("severity", {}) or {}
        rho = sec.get("rho", {}) or {}
        by = {str(k): float(v) for k, v in (rho.get("by_type") or {}).items()}
        mode = str((sec.get("ice_type") or {}).get("mode", "none_rule"))
        return cls(
            by_type=by or dict(RHO_BY_ICE_TYPE),
            default=float(rho.get("default", RHO_STANDARD)),
            mode=mode,
        )

    def resolve(self, ice_type: IceType | str | None) -> tuple[float, RhoSource]:
        """返回 (ρ, 来源)。`none`/`unknown`/缺失 一律用标准密度。"""
        key = str(ice_type or "unknown")
        if key in ("unknown", "none"):
            return self.default, "standard"
        return self.by_type.get(key, self.default), "config_prior"


def resolve_ice_type(grade: VisualGrade) -> tuple[IceType, IceTypeSource]:
    """由视觉等级给出覆冰类型。**没有标注数据时不做任何猜测。**"""
    if grade is VisualGrade.G0:
        return IceType.NONE, "none_rule"
    return IceType.UNKNOWN, "unavailable"


def ice_type_confidence(source: IceTypeSource) -> float:
    """只有确定性的规则结论才给高置信度；其余为 0（表示"未知"而非"很确信"）。"""
    return 1.0 if source == "none_rule" else 0.0
