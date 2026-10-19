"""服务契约测试（§16.3）。

只验证 IO 层：状态码、错误码、契约字段。模型用随机权重。
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from grid_ice_vision.application.pipeline import InferencePipeline
from grid_ice_vision.common.config import load_config
from grid_ice_vision.service.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    cfg = load_config("configs/model/pipeline.yaml")
    cfg["infer"]["detector"]["tile"] = 320
    cfg["evidence_dir"] = None
    pipe = InferencePipeline.from_config(cfg, device="cpu")
    app = create_app(pipeline=pipe)
    return TestClient(app)


def _image_b64(h: int = 240, w: int = 320) -> str:
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _payload() -> dict[str, Any]:
    return {
        "image_id": "S01_C01_20260115T083012",
        "source": {
            "site_id": "S01",
            "camera_id": "C01",
            "asset_id": "LINE_220kV_01",
            "timestamp": "2026-01-15T08:30:12+08:00",
            "ptz": {"zoom": 12.0, "pan": 0.0, "tilt": -3.5},
        },
        "calib": {"d0_mm": 30.0, "design_ice_mm": 20.0},
        "history_d_eq_mm": [3.0, 3.2],
    }


def test_health(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models_loaded"] is True


def test_model_info(client: TestClient) -> None:
    r = client.get("/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "config_hash" in body
    assert body["rule_version"] == "1.0.0"


def test_infer_json_ok(client: TestClient) -> None:
    body = {**_payload(), "image_b64": _image_b64()}
    r = client.post("/v1/infer/json", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["schema_version"] == "1.0.0"
    assert out["image_id"] == body["image_id"]
    assert "quality" in out and "severity" in out


def test_infer_json_missing_image(client: TestClient) -> None:
    r = client.post("/v1/infer/json", json=_payload())
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_infer_json_bad_contract(client: TestClient) -> None:
    body = {"image_id": "x", "image_b64": _image_b64()}
    r = client.post("/v1/infer/json", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_contract"


def test_infer_json_bad_base64(client: TestClient) -> None:
    body = {**_payload(), "image_b64": "@@@not-base64@@@"}
    r = client.post("/v1/infer/json", json=body)
    assert r.status_code == 400


def test_metrics(client: TestClient) -> None:
    client.get("/v1/health")
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["requests"] >= 1
    assert 0.0 <= body["error_rate"] <= 1.0


def test_schema_endpoint(client: TestClient) -> None:
    r = client.get("/v1/schema")
    assert r.status_code == 200
    assert "request" in r.json()


def test_multipart_infer(client: TestClient) -> None:
    img = np.full((160, 200, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    files = {"image": ("test.png", buf.tobytes(), "image/png")}
    data = {"payload": __import__("json").dumps(_payload())}
    r = client.post("/v1/infer", files=files, data=data)
    assert r.status_code == 200, r.text
