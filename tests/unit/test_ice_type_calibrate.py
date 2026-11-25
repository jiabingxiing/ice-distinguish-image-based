"""多分类温度缩放、三重拒识门控、推理包装与评估指标单测。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.common.schemas import IceType
from grid_ice_vision.evaluation.ice_type import analyze_confusion_matrix, compute_ice_type_metrics
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    RejectionThresholds,
    apply_rejection_gate,
    search_rejection_thresholds,
)
from grid_ice_vision.features.ice_type.classifier import IceTypeLinearHead
from grid_ice_vision.features.ice_type.infer import IceTypeInferConfig, IceTypeInferencer


def test_temperature_scaler_reduces_nll_and_preserves_argmax() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=200)
    logits = rng.normal(0.0, 2.0, size=(200, 4))
    for i in range(200):
        logits[i, y[i]] += 3.0
    logits *= 3.5  # 故意制造过自信

    sc = MultiClassTemperatureScaler().fit(logits, y)
    assert sc.fitted is True
    assert sc.temperature > 1.0
    assert sc.nll_after <= sc.nll_before + 1e-6

    probs = sc.predict_proba(logits)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(probs.argmax(axis=1), logits.argmax(axis=1))

    reloaded = MultiClassTemperatureScaler.from_dict(sc.to_dict())
    assert reloaded.temperature == pytest.approx(sc.temperature, abs=1e-5)


def test_triple_rejection_gate_reasons() -> None:
    probs = np.array(
        [
            [0.85, 0.05, 0.05, 0.05],  # 低质拒识
            [0.35, 0.25, 0.20, 0.20],  # 低置信度拒识 (< 0.45)
            [0.46, 0.43, 0.07, 0.04],  # Top1-Top2 差值过小拒识 (< 0.08)
            [0.75, 0.15, 0.05, 0.05],  # 通过
        ]
    )
    q = np.array([0.10, 0.80, 0.80, 0.80])
    th = RejectionThresholds(tau_q=0.25, tau_p=0.45, tau_m=0.08)
    mask, reasons = apply_rejection_gate(probs, q, th)
    assert mask.tolist() == [True, True, True, False]
    assert reasons == ["low_image_quality", "low_confidence", "ambiguous_prediction", ""]


def test_search_rejection_thresholds_respects_coverage() -> None:
    y = np.array([i % 4 for i in range(120)], dtype=np.int64)
    probs = np.full((120, 4), 0.1)
    for i in range(120):
        if i < 100:
            probs[i] = [0.05, 0.05, 0.05, 0.05]
            probs[i, y[i]] = 0.85
        else:
            # 20 个困难错分样本
            wrong = (y[i] + 1) % 4
            probs[i] = [0.24, 0.24, 0.24, 0.24]
            probs[i, wrong] = 0.34
            probs[i] /= probs[i].sum()
    q = np.full(120, 0.8)
    res = search_rejection_thresholds(probs, q, y, min_coverage=0.80)
    assert res["coverage"] >= 0.80
    assert res["accepted_macro_f1"] >= res["val_full_macro_f1"]


def test_inferencer_end_to_end_and_icing_result_serialization() -> None:
    torch.manual_seed(0)
    bb = IceTypeBackbone("mobilenetv3_small_100", pretrained=False)
    head = IceTypeLinearHead(576, 4, dropout=0.0)
    cfg = IceTypeInferConfig(
        image_size=64,
        rejection_thresholds=RejectionThresholds(tau_q=0.2, tau_p=0.95, tau_m=0.90),
    )
    inf = IceTypeInferencer(backbone=bb, head=head, cfg=cfg)

    rng = np.random.default_rng(1)
    img = rng.integers(40, 210, (96, 128, 3), dtype=np.uint8)
    pred = inf.predict(img)
    # 极端高阈值下必然拒识
    assert pred.ice_type is IceType.UNKNOWN
    assert pred.requires_review is True
    assert set(pred.class_probabilities) == set(ICE_TYPE_FOUR_CLASSES)
    assert sum(pred.class_probabilities.values()) == pytest.approx(1.0, abs=1e-2)

    icing = pred.to_icing_result()
    assert icing.ice_type is IceType.UNKNOWN
    assert icing.requires_review is True


def test_evaluation_metrics_and_confusion_matrix() -> None:
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    p = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.05, 0.05],  # glaze -> rime
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
            [0.05, 0.05, 0.05, 0.85],
        ]
    )
    cm = analyze_confusion_matrix(y, p.argmax(axis=1))
    assert cm["glaze_to_rime"] == pytest.approx(0.5)
    assert cm["rime_to_glaze"] == pytest.approx(0.0)

    rep = compute_ice_type_metrics(
        p,
        y,
        groups={
            "bureau": ["B1"] * 4 + ["B2"] * 4,
            "year": ["2017"] * 8,
            "resolution": ["640x480"] * 6 + ["1280x720"] * 2,
            "monitor_id": ["M1"] * 6 + ["M2"] * 2,
        },
    )
    assert rep["macro_f1"] > 0.7
    assert "bureau" in rep["slices"]
    assert "low_res_le704" in rep["slices"]["resolution_bucket"]
