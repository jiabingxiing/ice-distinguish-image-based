"""多分类温度标定与三重拒识阈值搜索（Plan §3.5 & §6.5）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from grid_ice_vision.evaluation.classification import balanced_accuracy, macro_f1


def softmax_numpy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def _multiclass_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    z = logits / max(temperature, 1e-6)
    z_shift = z - z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.maximum(np.exp(z_shift).sum(axis=1), 1e-12))
    target_logits = z_shift[np.arange(len(targets)), targets]
    return float(np.mean(log_sum_exp - target_logits))


@dataclass
class MultiClassTemperatureScaler:
    """多分类单标量温度校准器（在验证集 logits 上最小化 NLL）。"""

    temperature: float = 1.0
    fitted: bool = False
    nll_before: float = 0.0
    nll_after: float = 0.0

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> MultiClassTemperatureScaler:
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64).ravel()
        if z.size == 0 or y.size == 0:
            return self
        self.nll_before = _multiclass_nll(z, y, 1.0)
        res = minimize_scalar(
            lambda log_t: _multiclass_nll(z, y, float(np.exp(log_t))),
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
        )
        t_opt = float(np.exp(res.x)) if res.success else 1.0
        self.temperature = max(0.05, min(20.0, t_opt))
        self.nll_after = _multiclass_nll(z, y, self.temperature)
        self.fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax_numpy(logits, temperature=self.temperature)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "temperature": round(self.temperature, 6),
            "fitted": self.fitted,
            "nll_before": round(self.nll_before, 6),
            "nll_after": round(self.nll_after, 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiClassTemperatureScaler:
        return cls(
            temperature=float(d.get("temperature", 1.0)),
            fitted=bool(d.get("fitted", False)),
            nll_before=float(d.get("nll_before", 0.0)),
            nll_after=float(d.get("nll_after", 0.0)),
        )


@dataclass(frozen=True)
class RejectionThresholds:
    """三重拒识阈值：质量 tau_q、最大置信度 tau_p、Top1-Top2 差值 tau_m。"""

    tau_q: float = 0.25
    tau_p: float = 0.45
    tau_m: float = 0.08
    min_coverage: float = 0.80

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> RejectionThresholds:
        if not cfg:
            return cls()
        r = cfg.get("reject", cfg)
        return cls(
            tau_q=float(r.get("tau_q", 0.25)),
            tau_p=float(r.get("tau_p", 0.45)),
            tau_m=float(r.get("tau_m", 0.08)),
            min_coverage=float(r.get("min_coverage", 0.80)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "tau_q": round(self.tau_q, 4),
            "tau_p": round(self.tau_p, 4),
            "tau_m": round(self.tau_m, 4),
            "min_coverage": round(self.min_coverage, 4),
        }


def apply_rejection_gate(
    probs: np.ndarray,
    quality_scores: np.ndarray | None,
    thresholds: RejectionThresholds,
) -> tuple[np.ndarray, list[str]]:
    """逐样本执行三重拒识门控，返回 (reject_mask, review_reasons)。"""
    p = np.asarray(probs, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), []

    sorted_p = np.sort(p, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if p.shape[1] >= 2 else np.zeros(n, dtype=np.float64)
    margin = top1 - top2

    q_arr = (
        np.asarray(quality_scores, dtype=np.float64).ravel()
        if quality_scores is not None
        else np.ones(n, dtype=np.float64)
    )

    reject_mask = np.zeros(n, dtype=bool)
    reasons: list[str] = []
    for i in range(n):
        if q_arr[i] < thresholds.tau_q:
            reject_mask[i] = True
            reasons.append("low_image_quality")
        elif top1[i] < thresholds.tau_p:
            reject_mask[i] = True
            reasons.append("low_confidence")
        elif margin[i] < thresholds.tau_m:
            reject_mask[i] = True
            reasons.append("ambiguous_prediction")
        else:
            reasons.append("")
    return reject_mask, reasons


def search_rejection_thresholds(
    probs: np.ndarray,
    quality_scores: np.ndarray | None,
    targets: np.ndarray,
    *,
    min_coverage: float = 0.80,
    base_tau_q: float = 0.25,
) -> dict[str, Any]:
    """在验证集搜索 (tau_q, tau_p, tau_m)，在覆盖率约束下最大化接受子集 Macro-F1。"""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).ravel()
    n_classes = p.shape[1]
    preds = p.argmax(axis=1)
    base_mf1 = macro_f1(preds, y, n_classes)

    q_arr = (
        np.asarray(quality_scores, dtype=np.float64).ravel()
        if quality_scores is not None
        else np.ones(len(y), dtype=np.float64)
    )

    sorted_p = np.sort(p, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if n_classes >= 2 else np.zeros(len(y), dtype=np.float64)
    margin = top1 - top2

    # tau_q 候选：只覆盖真实低质/退化区间 [base_tau_q .. 0.45]，防止把正常清晰图误判为低质
    q_low = min(0.45, max(base_tau_q, float(np.quantile(q_arr, 0.02))))
    tau_q_cands = sorted({round(base_tau_q, 3), round(q_low, 3)})
    tau_p_cands = [round(float(v), 3) for v in np.linspace(0.25, 0.70, 19)]
    tau_m_cands = [round(float(v), 3) for v in np.linspace(0.0, 0.20, 11)]

    best_score = (-1.0, -1.0, -1.0)
    best_th = RejectionThresholds(
        tau_q=base_tau_q, tau_p=0.45, tau_m=0.08, min_coverage=min_coverage
    )
    best_stats = {"coverage": 1.0, "accepted_macro_f1": base_mf1, "accepted_balanced_acc": 0.0}

    for tq in tau_q_cands:
        q_ok = q_arr >= tq
        if float(q_ok.mean()) < min_coverage * 0.5:
            continue
        for tp in tau_p_cands:
            qp_ok = q_ok & (top1 >= tp)
            if float(qp_ok.mean()) < min_coverage:
                continue
            for tm in tau_m_cands:
                accept = qp_ok & (margin >= tm)
                cov = float(accept.mean())
                if cov < min_coverage or not accept.any():
                    continue
                mf1 = macro_f1(preds[accept], y[accept], n_classes)
                bacc = balanced_accuracy(preds[accept], y[accept], n_classes)
                # 优先更高 Macro-F1，若非常接近则优先更高覆盖率
                key = (round(mf1, 4), round(bacc, 4), round(cov, 4))
                if key > best_score:
                    best_score = key
                    best_th = RejectionThresholds(
                        tau_q=tq, tau_p=tp, tau_m=tm, min_coverage=min_coverage
                    )
                    best_stats = {
                        "coverage": round(cov, 4),
                        "reject_rate": round(1.0 - cov, 4),
                        "accepted_macro_f1": round(mf1, 4),
                        "accepted_balanced_acc": round(bacc, 4),
                    }

    return {
        "thresholds": best_th,
        "val_full_macro_f1": round(base_mf1, 4),
        "val_macro_f1_gain": round(best_stats["accepted_macro_f1"] - base_mf1, 4),
        **best_stats,
    }
