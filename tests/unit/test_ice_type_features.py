"""覆冰类型特征、质量评分、ROI 偏移检测、576 维骨干与 SVM PyTorch 头单测。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.classical import (
    ClassicalFeatureConfig,
    classical_feature_dim,
    extract_classical_features,
)
from grid_ice_vision.features.ice_type.classifier import (
    LinearSvmHead,
    RbfSvmTorchHead,
    export_torch_head,
)
from grid_ice_vision.features.ice_type.quality import (
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


def _textured(h: int = 160, w: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(35, 220, (h, w, 3), dtype=np.uint8)


def test_classical_features_finite_and_stable_dim() -> None:
    cfg = ClassicalFeatureConfig()
    dim = classical_feature_dim(cfg)
    feat = extract_classical_features(_textured(), cfg)
    assert feat.ndim == 1
    assert feat.shape[0] == dim
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_quality_gate_rejects_dark_blur_and_bad_roi() -> None:
    th = IceTypeQualityThresholds(tau_q=0.25)
    good = evaluate_ice_type_quality(_textured(), th, roi_image_bgr=_textured(64, 64))
    assert good.usable is True
    assert good.q_image >= 0.25

    dark = np.zeros((120, 160, 3), dtype=np.uint8)
    q_dark = evaluate_ice_type_quality(dark, th)
    assert q_dark.usable is False
    assert q_dark.q_image < 0.25

    flat_roi = np.full((64, 64, 3), 120, dtype=np.uint8)
    q_roi = evaluate_ice_type_quality(_textured(), th, roi_image_bgr=flat_roi)
    assert q_roi.roi_readable is False
    assert q_roi.usable is False


def test_roi_validation_crop_and_drift() -> None:
    with pytest.raises(ValueError):
        NormalizedRoi("M1", 0.5, 0.2, 0.4, 0.8)

    img = _textured(100, 200)
    roi = NormalizedRoi("M1", 0.25, 0.20, 0.75, 0.80, "insulator")
    crop = crop_normalized_roi(img, roi)
    assert crop.shape == (60, 100, 3)

    fb = center_fallback_crop(img, crop_ratio=0.5)
    assert fb.shape == (50, 100, 3)

    sig = compute_view_signature(crop)
    drifted_same, sim_same = detect_view_drift(crop, sig, corr_min=0.35)
    assert not drifted_same and sim_same > 0.99

    other = cv2.GaussianBlur(_textured(60, 100, seed=99), (15, 15), 5.0)
    drifted_other, sim_other = detect_view_drift(other, sig, corr_min=0.5)
    assert drifted_other and sim_other < 0.5


@pytest.mark.parametrize("provider", ["timm", "torchvision"])
def test_mobilenetv3_backbone_576_dim_and_unfreeze(provider: str) -> None:
    bb = IceTypeBackbone("mobilenetv3_small_100", provider=provider, pretrained=False)  # type: ignore[arg-type]
    x = torch.randn(2, 3, 224, 224)
    f_single = bb(x)
    f_dual = bb(x, x)
    assert f_single.shape == (2, 576)
    assert f_dual.shape == (2, 1152)

    bb.freeze_all()
    assert sum(p.numel() for p in bb.parameters() if p.requires_grad) == 0

    n1 = bb.unfreeze_last_stages(1)
    n2 = bb.unfreeze_last_stages(2)
    assert 0 < n1 < n2 < sum(p.numel() for p in bb.parameters())

    head = torch.nn.Linear(576, 4)
    groups = bb.stage_parameter_groups(1e-5, 3e-4, head=head)
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(3e-4)


def test_sklearn_to_torch_head_numeric_equivalence() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 16)).astype(np.float32)
    y = np.array([i % 4 for i in range(80)], dtype=np.int64)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    # 1. LinearSVC
    lsvc = LinearSVC(C=1.0, random_state=42).fit(Z, y)
    lhead = export_torch_head(scaler, lsvc)
    assert isinstance(lhead, LinearSvmHead)
    with torch.no_grad():
        got_lin = lhead(torch.from_numpy(X)).numpy()
    assert np.allclose(got_lin, lsvc.decision_function(Z), atol=1e-4)

    # 2. RBF SVC
    rsvc = SVC(
        C=1.0, kernel="rbf", gamma="scale", decision_function_shape="ovo", random_state=42
    ).fit(Z, y)
    rhead = export_torch_head(scaler, rsvc)
    assert isinstance(rhead, RbfSvmTorchHead)
    with torch.no_grad():
        got_ovo = rhead.ovo_decision(torch.from_numpy(X)).numpy()
    assert np.allclose(got_ovo, rsvc.decision_function(Z), atol=1e-4)

    # 3. MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=42).fit(Z, y)
    mhead = export_torch_head(scaler, mlp)
    with torch.no_grad():
        out = mhead(torch.from_numpy(X))
    assert out.shape == (80, 4)
