"""纯图像覆冰类型四分类算法子包（只做纯算法，禁止文件 I/O、配置读取与日志）。"""

from __future__ import annotations

from grid_ice_vision.features.ice_type.classical import (
    ClassicalFeatureConfig,
    classical_feature_dim,
    extract_classical_features,
)
from grid_ice_vision.features.ice_type.quality import (
    IceTypeQuality,
    IceTypeQualityThresholds,
    evaluate_ice_type_quality,
)
from grid_ice_vision.features.ice_type.roi import (
    NormalizedRoi,
    center_fallback_crop,
    compute_view_signature,
    crop_normalized_roi,
    detect_view_drift,
)

__all__ = [
    "ClassicalFeatureConfig",
    "IceTypeQuality",
    "IceTypeQualityThresholds",
    "NormalizedRoi",
    "center_fallback_crop",
    "classical_feature_dim",
    "compute_view_signature",
    "crop_normalized_roi",
    "detect_view_drift",
    "evaluate_ice_type_quality",
    "extract_classical_features",
]
