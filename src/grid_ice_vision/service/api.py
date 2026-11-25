"""FastAPI 服务（§16.3）。

只做 IO：校验 → 调 pipeline → 返回契约 JSON。不做任何算法。
错误码语义固定，不泄露堆栈。
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from grid_ice_vision.application.pipeline import InferencePipeline
from grid_ice_vision.common.config import load_config
from grid_ice_vision.common.logging import get_logger
from grid_ice_vision.common.schemas import InferenceRequest, SourceInfo
from grid_ice_vision.service.errors import (
    ErrorCode,
    ImageDecodeError,
    ModelNotReadyError,
    ServiceError,
)

log = get_logger("service")


class _Metrics:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.latencies: list[float] = []

    def record(self, ms: float, ok: bool) -> None:
        self.requests += 1
        self.errors += int(not ok)
        self.latencies.append(ms)
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    def snapshot(self) -> dict[str, Any]:
        lat = np.asarray(self.latencies) if self.latencies else np.zeros(1)
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.errors / max(self.requests, 1),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
        }


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError()
    return img


def create_app(
    pipeline: InferencePipeline | None = None, config: dict[str, Any] | None = None
) -> FastAPI:
    state: dict[str, Any] = {"pipeline": pipeline, "metrics": _Metrics(), "started_at": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state["pipeline"] is None:
            cfg = config or load_config("model/pipeline.yaml")
            log.info("加载模型（device=%s）", cfg.get("device", "cpu"))
            state["pipeline"] = InferencePipeline.from_config(
                cfg, device=str(cfg.get("device", "cpu"))
            )
        yield

    app = FastAPI(title="grid-ice-vision", version="0.1.0", lifespan=lifespan)
    app.state.store = state

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status, content={"error": exc.code.value, "message": exc.message}
        )

    @app.exception_handler(Exception)
    async def _internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.error("未捕获异常：%s", exc)
        return JSONResponse(
            status_code=500, content={"error": ErrorCode.INTERNAL.value, "message": "内部错误"}
        )

    def _pipe() -> InferencePipeline:
        p = state["pipeline"]
        if p is None:
            raise ModelNotReadyError()
        return p

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        p = state["pipeline"]
        return {
            "status": "ok" if p is not None else "loading",
            "models_loaded": p is not None,
            "uptime_s": round(time.time() - state["started_at"], 1),
        }

    @app.get("/v1/model/info")
    def model_info() -> dict[str, Any]:
        p = _pipe()
        return {
            "precip_backbone": p.precip.model.backbone.spec.name,
            "detector_backbone": p.detector.model.backbone.spec.name,
            "seg_backbone": p.seg.model.backbone.spec.name,
            "ice_type_mode": p.rho.mode,
            "ice_type_model": p.ice_type_service.model_version if p.ice_type_service else None,
            "config_hash": p._cfg_hash[:16],
            "rule_version": p.alarm_rules.version,
            "schema_version": "1.0.0",
        }

    @app.get("/v1/metrics")
    def metrics() -> dict[str, Any]:
        return state["metrics"].snapshot()

    def _run(image: np.ndarray, req: InferenceRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = _pipe().run(
                image=image,
                image_id=req.image_id,
                source=req.source,
                calib=req.calib,
                history_d_eq_mm=req.history_d_eq_mm,
            )
            state["metrics"].record((time.perf_counter() - t0) * 1000, True)
            return result.to_json_dict()
        except Exception:
            state["metrics"].record((time.perf_counter() - t0) * 1000, False)
            raise

    @app.post("/v1/infer")
    async def infer(
        image: UploadFile = File(...),  # noqa: B008 - FastAPI 依赖注入惯用法
        payload: str = Form(..., description="InferenceRequest 的 JSON 字符串"),
    ) -> dict[str, Any]:
        try:
            req = InferenceRequest(**json.loads(payload))
        except Exception as exc:
            raise ServiceError(ErrorCode.INVALID_CONTRACT, f"请求契约校验失败: {exc}", 422) from exc
        data = await image.read()
        return _run(_decode_image(data), req)

    @app.post("/v1/infer/json")
    def infer_json(body: dict[str, Any]) -> dict[str, Any]:
        b64 = body.pop("image_b64", None)
        if not b64:
            raise ServiceError(ErrorCode.BAD_REQUEST, "缺少 image_b64", 400)
        try:
            req = InferenceRequest(**body)
        except Exception as exc:
            raise ServiceError(ErrorCode.INVALID_CONTRACT, f"请求契约校验失败: {exc}", 422) from exc
        try:
            raw = base64.b64decode(b64)
        except Exception as exc:
            raise ServiceError(ErrorCode.BAD_REQUEST, "image_b64 解码失败", 400) from exc
        return _run(_decode_image(raw), req)

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        return {
            "request": InferenceRequest.model_json_schema(),
            "source": SourceInfo.model_json_schema(),
        }

    return app


app = create_app()
