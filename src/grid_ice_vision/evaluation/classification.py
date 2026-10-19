"""分类指标：Macro-F1 / Balanced Acc / ECE / QWK（§15.2）。"""

from __future__ import annotations

import numpy as np

from grid_ice_vision.features.precipitation.calibrate import expected_calibration_error


def macro_f1(pred: np.ndarray, target: np.ndarray, n_classes: int | None = None) -> float:
    """多类或多标签（0/1）的 Macro-F1。"""
    p = np.asarray(pred).ravel()
    t = np.asarray(target).ravel()
    k = n_classes or int(max(p.max(), t.max())) + 1
    scores: list[float] = []
    for c in range(k):
        tp = float(((p == c) & (t == c)).sum())
        fp = float(((p == c) & (t != c)).sum())
        fn = float(((p != c) & (t == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def balanced_accuracy(pred: np.ndarray, target: np.ndarray, n_classes: int | None = None) -> float:
    p = np.asarray(pred).ravel()
    t = np.asarray(target).ravel()
    k = n_classes or int(max(p.max(), t.max())) + 1
    recalls: list[float] = []
    for c in range(k):
        m = t == c
        if not m.any():
            continue
        recalls.append(float((p[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def quadratic_weighted_kappa(pred: np.ndarray, target: np.ndarray, n_classes: int = 5) -> float:
    """QWK：有序等级评估的首选指标（对跨级误判惩罚更重）。"""
    p = np.asarray(pred).astype(int).ravel()
    t = np.asarray(target).astype(int).ravel()
    if p.size == 0:
        return 0.0
    conf = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(p, t, strict=True):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            conf[a, b] += 1
    w = np.zeros_like(conf)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    hist_p = conf.sum(axis=1)
    hist_t = conf.sum(axis=0)
    expected = np.outer(hist_p, hist_t) / max(conf.sum(), 1)
    num = float((w * conf).sum())
    den = float((w * expected).sum())
    return 1.0 - num / den if den > 0 else 0.0


def calibration_report(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    return {"ece": expected_calibration_error(probs, targets, n_bins)}


def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    pred = np.asarray(logits).argmax(axis=1)
    return float((pred == np.asarray(labels)).mean())
