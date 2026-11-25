"""覆冰类型多维评估指标：Macro-F1、Balanced Acc、PR-AUC、ECE、拒识曲线与分组切片。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.evaluation.classification import balanced_accuracy, macro_f1
from grid_ice_vision.evaluation.ice_type.confusion_matrix import analyze_confusion_matrix


def multiclass_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """多分类置信度 ECE = Σ (n_b/N)·|acc_b − conf_b|（以 Top-1 置信度分箱）。"""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).ravel()
    if p.size == 0 or y.size == 0:
        return 0.0
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = float(len(y))
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (float(m.sum()) / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def multiclass_pr_auc_ovr(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: tuple[str, ...] = ICE_TYPE_FOUR_CLASSES,
) -> dict[str, Any]:
    """One-vs-Rest PR-AUC（逐类及宏平均）。"""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).ravel()
    per_cls: dict[str, float | None] = {}
    vals: list[float] = []
    for i, name in enumerate(class_names):
        y_bin = (y == i).astype(np.int32)
        pos = int(y_bin.sum())
        if pos == 0 or pos == len(y_bin):
            per_cls[name] = None
            continue
        ap = float(average_precision_score(y_bin, p[:, i]))
        per_cls[name] = round(ap, 4)
        vals.append(ap)
    return {
        "macro": round(float(np.mean(vals)), 4) if vals else 0.0,
        "per_class": per_cls,
    }


def _per_class_prf(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: tuple[str, ...],
    pr_auc_per_cls: dict[str, float | None],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(class_names):
        p_i = preds == i
        t_i = targets == i
        tp = float((p_i & t_i).sum())
        fp = float((p_i & ~t_i).sum())
        fn = float((~p_i & t_i).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pr_auc": pr_auc_per_cls.get(name),
            "support": int(t_i.sum()),
        }
    return out


def _coverage_accuracy_curve(
    probs: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, float]]:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    n_total = len(targets)
    curve: list[dict[str, float]] = []
    if n_total == 0:
        return curve
    for thr in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90):
        mask = conf >= thr
        cov = float(mask.mean())
        acc = float((preds[mask] == targets[mask]).mean()) if mask.any() else 0.0
        mf1 = macro_f1(preds[mask], targets[mask], probs.shape[1]) if mask.any() else 0.0
        curve.append(
            {
                "tau_p": round(thr, 2),
                "coverage": round(cov, 4),
                "accuracy": round(acc, 4),
                "macro_f1": round(mf1, 4),
            }
        )
    return curve


def _resolution_bucket(res_str: str) -> str:
    try:
        w_s, _ = res_str.lower().split("x", maxsplit=1)
        w = int(w_s)
    except Exception:
        return "unknown"
    return "low_res_le704" if w <= 704 else "high_res_gt704"


def _slice_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    keys: list[str],
    n_classes: int,
    min_samples: int = 1,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, k in enumerate(keys):
        buckets[str(k)].append(idx)
    out: dict[str, dict[str, Any]] = {}
    for k, idxs in sorted(buckets.items()):
        if len(idxs) < min_samples:
            continue
        ii = np.asarray(idxs, dtype=np.int64)
        p_s, t_s = preds[ii], targets[ii]
        out[k] = {
            "n": len(ii),
            "accuracy": round(float((p_s == t_s).mean()), 4),
            "balanced_accuracy": round(balanced_accuracy(p_s, t_s, n_classes), 4),
            "macro_f1": round(macro_f1(p_s, t_s, n_classes), 4),
        }
    return out


def compute_ice_type_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    class_names: tuple[str, ...] = ICE_TYPE_FOUR_CLASSES,
    reject_mask: np.ndarray | None = None,
    reject_reasons: list[str] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """计算四分类全量指标、拒识前后对比与按局/终端/年份/分辨率切片报告。"""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).ravel()
    n_classes = len(class_names)
    preds = p.argmax(axis=1)

    acc = float((preds == y).mean()) if y.size else 0.0
    bacc = balanced_accuracy(preds, y, n_classes) if y.size else 0.0
    mf1 = macro_f1(preds, y, n_classes) if y.size else 0.0
    ece = multiclass_ece(p, y, n_bins=10)
    pr_auc = multiclass_pr_auc_ovr(p, y, class_names=class_names)
    per_cls = _per_class_prf(preds, y, class_names, pr_auc["per_class"])
    conf_diag = analyze_confusion_matrix(y, preds, class_names=class_names)

    report: dict[str, Any] = {
        "n_samples": len(y),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_f1": round(mf1, 4),
        "ece": round(ece, 4),
        "pr_auc_macro": pr_auc["macro"],
        "per_class": per_cls,
        "confusion": conf_diag,
        "coverage_accuracy_curve": _coverage_accuracy_curve(p, y),
    }

    if reject_mask is not None and len(y) > 0:
        rej = np.asarray(reject_mask, dtype=bool).ravel()
        acc_mask = ~rej
        rej_rate = float(rej.mean())
        reason_counts: dict[str, int] = {}
        if reject_reasons:
            for r_flag, r_str in zip(rej.tolist(), reject_reasons, strict=True):
                if r_flag:
                    key = r_str or "unknown"
                    reason_counts[key] = reason_counts.get(key, 0) + 1
        if acc_mask.any():
            p_acc, y_acc = preds[acc_mask], y[acc_mask]
            acc_a = float((p_acc == y_acc).mean())
            bacc_a = balanced_accuracy(p_acc, y_acc, n_classes)
            mf1_a = macro_f1(p_acc, y_acc, n_classes)
            ece_a = multiclass_ece(p[acc_mask], y_acc, n_bins=10)
        else:
            acc_a = bacc_a = mf1_a = ece_a = 0.0
        report["rejection"] = {
            "reject_count": int(rej.sum()),
            "accepted_count": int(acc_mask.sum()),
            "reject_rate": round(rej_rate, 4),
            "coverage": round(1.0 - rej_rate, 4),
            "accepted_accuracy": round(acc_a, 4),
            "accepted_balanced_accuracy": round(bacc_a, 4),
            "accepted_macro_f1": round(mf1_a, 4),
            "accepted_ece": round(ece_a, 4),
            "macro_f1_gain": round(mf1_a - mf1, 4),
            "reasons": reason_counts,
        }

    if groups:
        slices: dict[str, Any] = {}
        if "bureau" in groups:
            slices["bureau"] = _slice_metrics(preds, y, groups["bureau"], n_classes)
        if "year" in groups:
            slices["year"] = _slice_metrics(preds, y, groups["year"], n_classes)
        if "resolution" in groups:
            buckets = [_resolution_bucket(r) for r in groups["resolution"]]
            slices["resolution_bucket"] = _slice_metrics(preds, y, buckets, n_classes)
        if "monitor_id" in groups:
            mon_slices = _slice_metrics(preds, y, groups["monitor_id"], n_classes, min_samples=5)
            ranked = sorted(mon_slices.items(), key=lambda kv: kv[1]["accuracy"])
            slices["monitor_lowest_top5"] = [{"monitor_id": m, **stats} for m, stats in ranked[:5]]
            slices["n_monitors_eval"] = len(set(groups["monitor_id"]))
        report["slices"] = slices

    return report
