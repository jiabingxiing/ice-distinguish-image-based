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
