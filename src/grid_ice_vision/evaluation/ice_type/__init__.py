"""覆冰类型四分类评测、混淆诊断与可视化导出模块。"""

from __future__ import annotations

from grid_ice_vision.evaluation.ice_type.confusion_matrix import (
    CONFUSED_PAIRS,
    analyze_confusion_matrix,
)
from grid_ice_vision.evaluation.ice_type.metrics import (
    compute_ice_type_metrics,
    multiclass_ece,
    multiclass_pr_auc_ovr,
)
from grid_ice_vision.evaluation.ice_type.visualization import (
    render_calibration_plot,
    render_confusion_heatmap,
    render_error_gallery,
)

__all__ = [
    "CONFUSED_PAIRS",
    "analyze_confusion_matrix",
    "compute_ice_type_metrics",
    "multiclass_ece",
    "multiclass_pr_auc_ovr",
    "render_calibration_plot",
    "render_confusion_heatmap",
    "render_error_gallery",
]
