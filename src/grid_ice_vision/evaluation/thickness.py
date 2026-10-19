"""厚度指标：MAPE / RMSE / R² / 像素宽度误差（§15.2）。

注意：MAPE 只在 `gt > 1 mm` 的样本上计算 —— 0 厚度样本会让 MAPE 爆炸，
这是文献里最常见的指标造假点之一。
"""

from __future__ import annotations

import numpy as np


def mape(pred: np.ndarray, gt: np.ndarray, min_gt: float = 1.0) -> float:
    p = np.asarray(pred, dtype=float)
    g = np.asarray(gt, dtype=float)
    m = g >= min_gt
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p[m] - g[m]) / g[m]) * 100.0)


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    g = np.asarray(gt, dtype=float)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(gt, float))))


def r2(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    g = np.asarray(gt, dtype=float)
    ss_res = float(((g - p) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def width_error_px(pred_w: np.ndarray, gt_w: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(pred_w, float) - np.asarray(gt_w, float))
    return {
        "mean_px": float(err.mean()),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(err.max()),
    }


def report(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "mape_pct": mape(pred, gt),
        "rmse_mm": rmse(pred, gt),
        "mae_mm": mae(pred, gt),
        "r2": r2(pred, gt),
        "n": float(len(np.asarray(pred))),
    }
