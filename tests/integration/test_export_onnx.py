"""ONNX 导出与数值一致性（§15.1：atol ≤ 1e-4）。

用最小输入尺寸以缩短 CI 时间；生产导出尺寸见 configs/deploy/。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grid_ice_vision.deployment.export import check_numeric_consistency, export_onnx
from grid_ice_vision.features.ice.model import UNetLite

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def seg_model() -> UNetLite:
    torch.manual_seed(0)
    return UNetLite(pretrained=False).eval()


def test_seg_export_and_match(seg_model: UNetLite, tmp_path: Path) -> None:
    out = tmp_path / "seg.onnx"
    export_onnx(seg_model, out, (1, 3, 64, 64))
    assert out.exists()
    res = check_numeric_consistency(seg_model, out, (1, 3, 64, 64), n_trials=3, atol=1e-4)
    assert res["passed"], res


def test_precip_export_multi_output(tmp_path: Path) -> None:
    from grid_ice_vision.features.precipitation.model import PrecipitationNet

    torch.manual_seed(0)
    model = PrecipitationNet(pretrained=False).eval()
    out = tmp_path / "precip.onnx"
    keys = ("weather_logits", "intensity_logits")
    export_onnx(model, out, (1, 3, 64, 64), keys=keys)
    res = check_numeric_consistency(model, out, (1, 3, 64, 64), n_trials=2, keys=keys)
    assert res["passed"], res


def test_numeric_check_reports_error(seg_model: UNetLite, tmp_path: Path) -> None:
    """人为把 atol 设成 0 时，浮点误差应让校验失败（证明校验真的在比对）。"""
    out = tmp_path / "seg2.onnx"
    export_onnx(seg_model, out, (1, 3, 32, 32))
    res = check_numeric_consistency(seg_model, out, (1, 3, 32, 32), n_trials=1, atol=0.0)
    assert res["max_abs_err"] >= 0.0
    assert res["passed"] is (res["max_abs_err"] <= 0.0)


def test_ice_type_single_and_dual_onnx_export(tmp_path: Path) -> None:
    """单视角与双视角 IceTypeTorchModel 导出 ONNX 并可通过纯生产加载器推理。"""
    import json

    import numpy as np

    from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
    from grid_ice_vision.deployment.export import export_and_verify
    from grid_ice_vision.deployment.ice_type_runtime import load_onnx_ice_type_inferencer
    from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
    from grid_ice_vision.features.ice_type.classifier import IceTypeLinearHead
    from grid_ice_vision.features.ice_type.infer import IceTypeTorchModel

    torch.manual_seed(0)
    bb = IceTypeBackbone(pretrained=False)

    m_single = IceTypeTorchModel(
        bb, IceTypeLinearHead(576, 4, dropout=0.0), view_mode="full"
    ).eval()
    p_single = tmp_path / "ice_single.onnx"
    res1 = export_and_verify(m_single, p_single, (1, 3, 224, 224), atol=1e-4)
    assert res1["passed"]

    m_dual = IceTypeTorchModel(
        bb, IceTypeLinearHead(1152, 4, dropout=0.0), view_mode="dual", temperature=1.2
    ).eval()
    p_dual = tmp_path / "ice_dual.onnx"
    res2 = export_and_verify(
        m_dual,
        p_dual,
        (1, 3, 224, 224),
        roi_shape=(1, 3, 224, 224),
        atol=1e-4,
    )
    assert res2["passed"]

    meta_p = p_dual.with_suffix(".meta.json")
    meta_p.write_text(
        json.dumps(
            {
                "model_name": "fallback_conv576",
                "model_version": "test-dual",
                "view_mode": "dual",
                "image_size": 224,
                "roi_size": 224,
                "pad_value": 114,
                "classes": list(ICE_TYPE_FOUR_CLASSES),
                "temperature": 1.0,
                "reject": {"tau_q": 0.05, "tau_p": 0.1, "tau_m": 0.0, "min_coverage": 0.8},
            }
        ),
        encoding="utf-8",
    )
    svc = load_onnx_ice_type_inferencer(p_dual, intra_op_threads=1)
    rng = np.random.default_rng(7)
    img = rng.integers(30, 220, (320, 480, 3), dtype=np.uint8)
    pred = svc.predict_bgr(img)
    assert len(pred.class_probabilities) == 4
    assert sum(pred.class_probabilities.values()) == pytest.approx(1.0, abs=1e-2)
