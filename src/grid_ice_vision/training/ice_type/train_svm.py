"""基于离线特征缓存（或传统手工特征）的 SVM / LR / MLP 训练与温度/拒识校准。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.evaluation.classification import balanced_accuracy, macro_f1
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    search_rejection_thresholds,
)
from grid_ice_vision.features.ice_type.classifier import SklearnClassifierAdapter

ClassifierType = Literal["svm_linear", "svm_rbf", "logistic", "mlp"]


@dataclass(frozen=True)
class ClassifierTrainConfig:
    classifier_type: ClassifierType = "svm_rbf"
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    gamma: str | float = "scale"
    class_weight: str | None = "balanced"
    oversample_train: bool = False
    probability_calibration: bool = True
    random_seed: int = 42
    min_coverage: float = 0.80
    tau_q: float = 0.25

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> ClassifierTrainConfig:
        if not cfg:
            return cls()
        c = cfg.get("classifier", {}) or {}
        t = cfg.get("training", {}) or {}
        r = cfg.get("reject", {}) or {}
        q = cfg.get("quality", {}) or {}
        ctype = str(c.get("type", "svm_rbf"))
        if ctype not in ("svm_linear", "svm_rbf", "logistic", "mlp"):
            raise ValueError(f"未知 classifier.type={ctype!r}")
        return cls(
            classifier_type=ctype,  # type: ignore[arg-type]
            c_grid=tuple(float(v) for v in c.get("c_grid", (0.1, 1.0, 10.0))),
            gamma=c.get("gamma", "scale"),
            class_weight=c.get("class_weight", "balanced"),
            oversample_train=bool(c.get("oversample_train", False)),
            probability_calibration=bool(c.get("probability_calibration", True)),
            random_seed=int(t.get("random_seed", 42)),
            min_coverage=float(r.get("min_coverage", 0.80)),
            tau_q=float(q.get("tau_q", r.get("tau_q", 0.25))),
        )


def _oversample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_n = int(np.max(counts))
    idxs: list[np.ndarray] = []
    for c in classes:
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) < target_n:
            extra = rng.choice(c_idx, size=target_n - len(c_idx), replace=True)
            idxs.append(np.concatenate([c_idx, extra]))
        else:
            idxs.append(c_idx)
    merged = np.concatenate(idxs)
    rng.shuffle(merged)
    return X[merged], y[merged]


def _build_estimator(
    ctype: ClassifierType,
    c_val: float,
    cfg: ClassifierTrainConfig,
) -> Any:
    if ctype == "svm_linear":
        return LinearSVC(
            C=c_val,
            class_weight=cfg.class_weight,
            random_state=cfg.random_seed,
            max_iter=5000,
        )
    if ctype == "svm_rbf":
        return SVC(
            C=c_val,
            kernel="rbf",
            gamma=cfg.gamma,
            class_weight=cfg.class_weight,
            decision_function_shape="ovo",
            random_state=cfg.random_seed,
        )
    if ctype == "logistic":
        return LogisticRegression(
            C=c_val,
            class_weight=cfg.class_weight,
            max_iter=2000,
            random_state=cfg.random_seed,
        )
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        alpha=1.0 / max(c_val * 100.0, 1.0),
        max_iter=300,
        early_stopping=True,
        random_state=cfg.random_seed,
    )


def train_cached_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: ClassifierTrainConfig | None = None,
    val_quality_scores: np.ndarray | None = None,
    out_dir: str | Path | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在特征向量上搜索最佳正则化 C 并完成温度标定与三重拒识阈值拟合。"""
    tcfg = cfg or ClassifierTrainConfig()
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int64).ravel()
    X_va = np.asarray(X_val, dtype=np.float32)
    y_va = np.asarray(y_val, dtype=np.int64).ravel()

    if tcfg.oversample_train:
        X_tr, y_tr = _oversample(X_tr, y_tr, seed=tcfg.random_seed)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_tr)

    best_score = -1.0
    best_c = tcfg.c_grid[0]
    best_clf: Any = None
    c_trials: list[dict[str, float]] = []

    for c_val in tcfg.c_grid:
        clf = _build_estimator(tcfg.classifier_type, float(c_val), tcfg)
        clf.fit(Z_tr, y_tr)
        adapter = SklearnClassifierAdapter(scaler, clf)
        val_logits = adapter.predict_logits(X_va)
        val_preds = val_logits.argmax(axis=1)
        mf1 = macro_f1(val_preds, y_va, len(ICE_TYPE_FOUR_CLASSES))
        bacc = balanced_accuracy(val_preds, y_va, len(ICE_TYPE_FOUR_CLASSES))
        c_trials.append(
            {"C": float(c_val), "val_macro_f1": round(mf1, 4), "val_bacc": round(bacc, 4)}
        )
        if mf1 > best_score:
            best_score = mf1
            best_c = float(c_val)
            best_clf = clf

    assert best_clf is not None
    best_adapter = SklearnClassifierAdapter(scaler, best_clf)
    val_logits = best_adapter.predict_logits(X_va)

    temp_scaler = MultiClassTemperatureScaler()
    if tcfg.probability_calibration:
        temp_scaler.fit(val_logits, y_va)
    val_probs = temp_scaler.predict_proba(val_logits)

    rej_search = search_rejection_thresholds(
        val_probs,
        val_quality_scores,
        y_va,
        min_coverage=tcfg.min_coverage,
        base_tau_q=tcfg.tau_q,
    )

    result: dict[str, Any] = {
        "classifier_type": tcfg.classifier_type,
        "best_C": best_c,
        "c_trials": c_trials,
        "val_macro_f1": round(best_score, 4),
        "temperature": temp_scaler.to_dict(),
        "reject_search": {k: v for k, v in rej_search.items() if k != "thresholds"}
        | {"thresholds": rej_search["thresholds"].to_dict()},
        "scaler": scaler,
        "classifier": best_clf,
        "temp_scaler": temp_scaler,
        "reject_thresholds": rej_search["thresholds"],
    }

    if out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, od / "scaler.joblib")
        joblib.dump(best_clf, od / "classifier.joblib")
        # 兼容计划 §3.5 命名约定
        joblib.dump(best_clf, od / "svm_classifier.joblib")
        (od / "temperature.json").write_text(
            json.dumps(temp_scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (od / "reject_thresholds.json").write_text(
            json.dumps(rej_search["thresholds"].to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_cfg = {
            "classifier_type": tcfg.classifier_type,
            "best_C": best_c,
            "classes": list(ICE_TYPE_FOUR_CLASSES),
            "oversample_train": tcfg.oversample_train,
            "feature_dim": int(X_tr.shape[1]),
            **(extra_meta or {}),
        }
        (od / "model_config.yaml").write_text(
            yaml.safe_dump(model_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    return result
