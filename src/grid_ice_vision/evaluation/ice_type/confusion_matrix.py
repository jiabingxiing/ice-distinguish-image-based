"""4×4 混淆矩阵与 4 对易混淆冰型专项诊断（Plan §3.2 & §6.2）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES

CONFUSED_PAIRS: tuple[tuple[str, str], ...] = (
    ("glaze", "wet_snow"),
    ("rime", "mixed"),
    ("glaze", "mixed"),
    ("rime", "wet_snow"),
    ("glaze", "rime"),
)


def analyze_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    class_names: tuple[str, ...] = ICE_TYPE_FOUR_CLASSES,
) -> dict[str, Any]:
    """构建行=真实、列=预测的混淆矩阵，并计算易混淆类别的双向误判率。"""
    y = np.asarray(targets, dtype=np.int64).ravel()
    p = np.asarray(preds, dtype=np.int64).ravel()
    k = len(class_names)
    mat = np.zeros((k, k), dtype=np.int64)
    for t, pr in zip(y, p, strict=True):
        if 0 <= t < k and 0 <= pr < k:
            mat[t, pr] += 1

    row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
    rate_mat = mat.astype(np.float64) / row_sums

    idx_map = {name: i for i, name in enumerate(class_names)}
    pair_stats: dict[str, dict[str, float]] = {}
    for a, b in CONFUSED_PAIRS:
        if a not in idx_map or b not in idx_map:
            continue
        ia, ib = idx_map[a], idx_map[b]
        a_to_b = float(rate_mat[ia, ib]) if mat[ia].sum() > 0 else 0.0
        b_to_a = float(rate_mat[ib, ia]) if mat[ib].sum() > 0 else 0.0
        total_pair = int(mat[ia].sum() + mat[ib].sum())
        bidir_err = float(mat[ia, ib] + mat[ib, ia]) / max(total_pair, 1)
        pair_stats[f"{a}_vs_{b}"] = {
            f"{a}_to_{b}_rate": round(a_to_b, 4),
            f"{b}_to_{a}_rate": round(b_to_a, 4),
            "bidirectional_error_rate": round(bidir_err, 4),
            f"{a}_to_{b}_count": int(mat[ia, ib]),
            f"{b}_to_{a}_count": int(mat[ib, ia]),
        }

    glaze_to_rime = pair_stats.get("glaze_vs_rime", {}).get("glaze_to_rime_rate", 0.0)
    rime_to_glaze = pair_stats.get("glaze_vs_rime", {}).get("rime_to_glaze_rate", 0.0)

    return {
        "labels": list(class_names),
        "matrix": mat.tolist(),
        "row_normalized": [[round(float(v), 4) for v in row] for row in rate_mat],
        "glaze_to_rime": glaze_to_rime,
        "rime_to_glaze": rime_to_glaze,
        "confused_pairs": pair_stats,
    }
