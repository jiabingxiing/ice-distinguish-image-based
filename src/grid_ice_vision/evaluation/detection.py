"""检测指标：mAP@0.5 / mAP@0.5:0.95 / 分类别 AP（自实现，不依赖 pycocotools）。"""

from __future__ import annotations

import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """按 101 点插值的 AP。"""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre)))


def evaluate_detections(
    preds: list[dict[str, np.ndarray]],
    gts: list[dict[str, np.ndarray]],
    n_classes: int,
    iou_thresholds: tuple[float, ...] = (0.5, 0.75, 0.95),
) -> dict[str, float]:
    """`preds[i]` / `gts[i]` 均含 `boxes (N,4)` 与 `labels (N,)`。"""
    results: dict[str, float] = {}
    aps_per_iou: list[float] = []
    for thr in iou_thresholds:
        aps: list[float] = []
        for c in range(n_classes):
            all_scores: list[np.ndarray] = []
            all_tp: list[np.ndarray] = []
            n_gt = 0
            for p, g in zip(preds, gts, strict=True):
                gb = g["boxes"][g["labels"] == c]
                pb = p["boxes"][p["labels"] == c]
                ps = (
                    p.get("scores", np.ones(len(pb), dtype=np.float32))[p["labels"] == c]
                    if "scores" in p
                    else np.ones(len(pb), dtype=np.float32)
                )
                n_gt += len(gb)
                if len(pb) == 0:
                    continue
                iou = _iou_matrix(pb, gb)
                matched = np.zeros(len(gb), dtype=bool)
                tp = np.zeros(len(pb), dtype=bool)
                order = np.argsort(-ps)
                for i in order:
                    if len(gb) == 0:
                        break
                    j = int(np.argmax(iou[i]))
                    if iou[i, j] >= thr and not matched[j]:
                        matched[j] = True
                        tp[i] = True
                all_scores.append(ps)
                all_tp.append(tp)
            if n_gt == 0:
                continue
            if not all_scores:
                aps.append(0.0)
                continue
            aps.append(average_precision(np.concatenate(all_scores), np.concatenate(all_tp), n_gt))
        if aps:
            aps_per_iou.append(float(np.nanmean(aps)))
        results[f"mAP@{thr:.2f}"] = float(np.nanmean(aps)) if aps else 0.0
    results["mAP@0.5:0.95"] = float(np.mean(aps_per_iou)) if aps_per_iou else 0.0
    return results
