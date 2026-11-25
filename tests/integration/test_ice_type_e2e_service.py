"""P4 端到端集成测试：ONNX Runtime 加载、InferencePipeline (shadow/model)、FastAPI 与边界守卫。"""

from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from grid_ice_vision.application.pipeline import InferencePipeline
from grid_ice_vision.common.config import load_config
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.common.schemas import (
    Asset,
    AssetClass,
    CalibInput,
    IceType,
    SourceInfo,
)
from grid_ice_vision.deployment.export import export_and_verify
from grid_ice_vision.deployment.ice_type_runtime import (
    OnnxIceTypeService,
    load_onnx_ice_type_inferencer,
)
from grid_ice_vision.features.ice.infer import SegPrediction
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.classifier import IceTypeLinearHead
from grid_ice_vision.features.ice_type.infer import IceTypeTorchModel

pytest.importorskip("onnxruntime")


def _build_temp_onnx_service(
    tmp_path: Path, *, reject_strict: bool = False
) -> tuple[OnnxIceTypeService, Path]:
    torch.manual_seed(42)
    bb = IceTypeBackbone(pretrained=False)
    head = IceTypeLinearHead(576, 4, dropout=0.0)
    # 固定偏置让预测稳定落到 rime（index 1）
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.bias.copy_(torch.tensor([0.1, 4.0, 0.2, 0.1]))
    model = IceTypeTorchModel(bb, head, view_mode="full").eval()
    onnx_p = tmp_path / "ice_type.onnx"
    res = export_and_verify(model, onnx_p, (1, 3, 224, 224), atol=1e-4)
    assert res["passed"]

    rej = (
        {"tau_q": 0.99, "tau_p": 0.99, "tau_m": 0.99, "min_coverage": 0.8}
        if reject_strict
        else {"tau_q": 0.01, "tau_p": 0.25, "tau_m": 0.0, "min_coverage": 0.8}
    )
    meta_p = onnx_p.with_suffix(".meta.json")
    meta_p.write_text(
        json.dumps(
            {
                "model_name": "fallback_conv576",
                "model_version": "ice-test-v1",
                "view_mode": "full",
                "image_size": 224,
                "roi_size": 224,
                "pad_value": 114,
                "classes": list(ICE_TYPE_FOUR_CLASSES),
                "temperature": 1.0,
                "reject": rej,
            }
        ),
        encoding="utf-8",
    )
    return load_onnx_ice_type_inferencer(onnx_p, intra_op_threads=1), onnx_p


def _textured_scene(h: int = 320, w: int = 480) -> np.ndarray:
    rng = np.random.default_rng(11)
    img = rng.integers(50, 210, (h, w, 3), dtype=np.uint8)
    cv2.line(img, (20, 160), (460, 160), (240, 240, 240), 6)
    return img


def _stub_det_and_seg(pipe: InferencePipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = Asset(
        cls=AssetClass.CONDUCTOR, bbox=(40.0, 120.0, 440.0, 200.0), score=0.95, roi_id="r0"
    )
    mask = np.zeros((80, 400), dtype=np.uint8)
    mask[36:44, :] = 1
    mask[32:36, :] = 2
    mask[44:48, :] = 2
    probs = np.zeros((4, 80, 400), dtype=np.float32)
    probs[1] = 0.8
    seg = SegPrediction(
        mask=mask,
        probs=probs,
        coverage=0.5,
        ice_present=True,
        confidence=0.9,
        bridge=None,
        roi_bbox=(40, 120, 440, 200),
    )
    monkeypatch.setattr(pipe.detector, "predict", lambda _im: [asset])
    monkeypatch.setattr(pipe.seg, "predict", lambda *a, **k: seg)


def test_runtime_and_pipeline_have_zero_training_imports() -> None:
    import grid_ice_vision.application.pipeline as pmod
    import grid_ice_vision.deployment.ice_type_runtime as rmod

    assert "grid_ice_vision.training" not in inspect.getsource(pmod)
    assert "grid_ice_vision.training" not in inspect.getsource(rmod)


def test_pipeline_shadow_mode_keeps_rho_900(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, onnx_p = _build_temp_onnx_service(tmp_path)
    cfg = load_config("configs/model/pipeline.yaml")
    cfg["weights"]["ice_type"] = str(onnx_p)
    cfg["severity"]["ice_type"]["mode"] = "shadow"
    cfg["evidence_dir"] = str(tmp_path / "ev")
    pipe = InferencePipeline.from_config(cfg, device="cpu")
    _stub_det_and_seg(pipe, monkeypatch)

    res = pipe.run(
        _textured_scene(),
        "shadow_01",
        SourceInfo(site_id="S1", camera_id="CC0002", timestamp="2026-01-15T08:00:00+08:00"),
        CalibInput(d0_mm=30.0, w0_px=8.0),
        write_evidence=False,
    )
    assert res.icing.ice_type is IceType.RIME
    assert res.icing.ice_type_source == "model"
    assert len(res.icing.class_probabilities) == 4
    assert res.severity.rho_kg_m3 == pytest.approx(900.0)
    assert res.severity.rho_source == "standard"
    assert res.provenance.model_version.get("ice_type") == "ice-test-v1"


def test_pipeline_model_mode_uses_density_only_when_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, onnx_p = _build_temp_onnx_service(tmp_path, reject_strict=False)
    cfg = load_config("configs/model/pipeline.yaml")
    cfg["weights"]["ice_type"] = str(onnx_p)
    cfg["severity"]["ice_type"]["mode"] = "model"
    cfg["evidence_dir"] = str(tmp_path / "ev1")
    pipe = InferencePipeline.from_config(cfg, device="cpu")
    _stub_det_and_seg(pipe, monkeypatch)
    res_ok = pipe.run(
        _textured_scene(),
        "model_ok",
        SourceInfo(site_id="S1", camera_id="C1", timestamp="2026-01-15T08:00:00+08:00"),
        CalibInput(d0_mm=30.0, w0_px=8.0),
        write_evidence=False,
    )
    assert res_ok.icing.ice_type is IceType.RIME
    assert res_ok.icing.requires_review is False
    assert res_ok.severity.rho_kg_m3 == pytest.approx(350.0)
    assert res_ok.severity.rho_source == "ice_type_model"

    # 若触发拒识，即使处于 model 模式也必须退回 UNKNOWN 与 900.0 kg/m³
    _, onnx_rej = _build_temp_onnx_service(tmp_path / "rej", reject_strict=True)
    cfg["weights"]["ice_type"] = str(onnx_rej)
    pipe_rej = InferencePipeline.from_config(cfg, device="cpu")
    _stub_det_and_seg(pipe_rej, monkeypatch)
    res_rej = pipe_rej.run(
        _textured_scene(),
        "model_rej",
        SourceInfo(site_id="S1", camera_id="C1", timestamp="2026-01-15T08:00:00+08:00"),
        CalibInput(d0_mm=30.0, w0_px=8.0),
        write_evidence=False,
    )
    assert res_rej.icing.ice_type is IceType.UNKNOWN
    assert res_rej.icing.requires_review is True
    assert res_rej.severity.rho_kg_m3 == pytest.approx(900.0)
    assert res_rej.severity.rho_source == "standard"


def test_api_endpoints_with_shadow_ice_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from grid_ice_vision.service.api import create_app

    _, onnx_p = _build_temp_onnx_service(tmp_path)
    cfg = load_config("configs/model/pipeline.yaml")
    cfg["weights"]["ice_type"] = str(onnx_p)
    cfg["severity"]["ice_type"]["mode"] = "shadow"
    cfg["evidence_dir"] = str(tmp_path / "ev_api")
    pipe = InferencePipeline.from_config(cfg, device="cpu")
    _stub_det_and_seg(pipe, monkeypatch)

    app = create_app(pipeline=pipe)
    client = TestClient(app)
    info: dict[str, Any] = client.get("/v1/model/info").json()
    assert info["ice_type_mode"] == "shadow"
    assert info["ice_type_model"] == "ice-test-v1"

    ok, buf = cv2.imencode(".jpg", _textured_scene())
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    resp = client.post(
        "/v1/infer/json",
        json={
            "image_id": "api_shadow_01",
            "image_b64": b64,
            "source": {
                "site_id": "S1",
                "camera_id": "C1",
                "timestamp": "2026-01-15T08:00:00+08:00",
            },
            "calib": {"d0_mm": 30.0, "w0_px": 8.0},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["icing"]["ice_type"] == "rime"
    assert data["severity"]["rho_kg_m3"] == pytest.approx(900.0)
    assert data["severity"]["rho_source"] == "standard"
