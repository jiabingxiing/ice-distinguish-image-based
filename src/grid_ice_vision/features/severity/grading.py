"""覆盖率 → 视觉等级 G0–G4（§8.5）。

G 级是**项目内部指标**，不得等同于任何电力行业规程等级。
阈值来自 `configs/labels/severity.yaml`，这里只给与配置一致的兜底默认值。
"""

from __future__ import annotations

from dataclasses import dataclass

from grid_ice_vision.common.schemas import VisualGrade

_DEFAULT_BOUNDS: tuple[tuple[float, VisualGrade], ...] = (
    (0.03, VisualGrade.G0),
    (0.15, VisualGrade.G1),
    (0.30, VisualGrade.G2),
    (0.50, VisualGrade.G3),
    (float("inf"), VisualGrade.G4),
)

_GRADE_TEXT: dict[VisualGrade, str] = {
    VisualGrade.G0: "轮廓清晰，无明显增厚",
    VisualGrade.G1: "边缘轻微增厚，导线轮廓基本保持",
    VisualGrade.G2: "轮廓明显改变，可见冰壳/短冰柱",
    VisualGrade.G3: "形态显著畸变，可见下垂/偏心/桥接",
    VisualGrade.G4: "大面积冰壳，跨导线桥接，或伴随明显弧垂",
}


@dataclass(frozen=True)
class SeverityThresholds:
    """G 级阈值表，按覆盖率升序排列。"""

    bounds: tuple[tuple[float, VisualGrade], ...] = _DEFAULT_BOUNDS

    @classmethod
    def from_config(cls, cfg: dict[str, object]) -> SeverityThresholds:
        """从 `configs/labels/severity.yaml` 的 `grades` 段构造。

        期望结构：`grades: {G0: {max_coverage: 0.03}, ...}`
        """
        grades = cfg.get("grades")
        if not isinstance(grades, dict):
            return cls()
        pairs: list[tuple[float, VisualGrade]] = []
        for name, spec in grades.items():
            if not isinstance(spec, dict):
                continue
            upper = spec.get("max_coverage")
            upper_f = float("inf") if upper is None else float(upper)  # type: ignore[arg-type]
            pairs.append((upper_f, VisualGrade(str(name))))
        if not pairs:
            return cls()
        pairs.sort(key=lambda p: p[0])
        return cls(bounds=tuple(pairs))

    def grade(self, coverage: float) -> VisualGrade:
        cov = max(0.0, min(1.0, float(coverage)))
        for upper, grade in self.bounds:
            if cov < upper:
                return grade
        return self.bounds[-1][1]


def grade_from_coverage(
    coverage: float, thresholds: SeverityThresholds | None = None
) -> VisualGrade:
    """覆盖率 → G 级。边界取「小于上界」，与 rubric 的区间定义一致。"""
    return (thresholds or SeverityThresholds()).grade(coverage)


def grade_description(grade: VisualGrade) -> str:
    return _GRADE_TEXT.get(grade, "")


def coverage_upper_bound(
    grade: VisualGrade, thresholds: SeverityThresholds | None = None
) -> float | None:
    """某等级的覆盖率上界，用于报表与证据图标注。"""
    th = thresholds or SeverityThresholds()
    for upper, g in th.bounds:
        if g is grade:
            return None if upper == float("inf") else upper
    return None
