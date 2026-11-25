"""纯 CPU 覆冰类型 ONNX Runtime 生产加载与推理封装（无 training/PyTorch 训练依赖）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from grid_ice_vision.common import paths
from grid_ice_vision.data.datasets.ice_type import load_monitor_roi_table
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    RejectionThresholds,
)
from grid_ice_vision.features.ice_type.infer import (
    IceTypeInferConfig,
    IceTypeInferencer,
    IceTypePrediction,
)
from grid_ice_vision.features.ice_type.quality import IceTypeQualityThresholds
from grid_ice_vision.features.ice_type.roi import NormalizedRoi


@dataclass
class OnnxIceTypeService:
    """封装 ONNX Session +  companion `.meta.json` + 终端 ROI 表。"""

    inferencer: IceTypeInferencer
    meta: dict[str, Any]
    roi_map: dict[str, NormalizedRoi]
    onnx_path: Path

    @property
    def model_version(self) -> str:
        return str(self.meta.get("model_version", self.onnx_path.stem))

    def resolve_roi(self, monitor_id: str | None) -> NormalizedRoi | None:
        if not monitor_id:
            return None
        return self.roi_map.get(str(monitor_id).strip())

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        monitor_id: str | None = None,
        roi: NormalizedRoi | None = None,
    ) -> IceTypePrediction:
        use_roi = roi if roi is not None else self.resolve_roi(monitor_id)
        return self.inferencer.predict(image_bgr, roi=use_roi)


def load_onnx_ice_type_inferencer(
    onnx_path: str | Path,
    *,
    meta_path: str | Path | None = None,
    roi_csv: str | Path | None = None,
    intra_op_threads: int = 4,
    providers: list[str] | None = None,
) -> OnnxIceTypeService:
    """加载 ONNX 模型与伴随 `.meta.json`（不依赖 timm / torch 训练模块）。"""
    import onnxruntime as ort

    model_p = Path(onnx_path)
    if not model_p.is_absolute():
        model_p = paths.PROJ / model_p
    if not model_p.exists():
        raise FileNotFoundError(f"覆冰类型 ONNX 模型不存在: {model_p}")

    mp = Path(meta_path) if meta_path is not None else model_p.with_suffix(".meta.json")
    if not mp.is_absolute():
        mp = paths.PROJ / mp
    meta: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(intra_op_threads))
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model_p),
        sess_options=so,
        providers=providers or ["CPUExecutionProvider"],
    )
    in_names = [i.name for i in sess.get_inputs()]

    raw_vm = str(meta.get("view_mode", "dual" if len(in_names) > 1 else "full"))
    vm: Literal["full", "roi", "dual"] = (
        raw_vm if raw_vm in ("full", "roi", "dual") else "full"  # type: ignore[assignment]
    )

    def _logits_fn(x_full: np.ndarray, x_roi: np.ndarray | None) -> np.ndarray:
        feed: dict[str, np.ndarray] = {in_names[0]: x_full.astype(np.float32)}
        if len(in_names) > 1:
            feed[in_names[1]] = (x_full if x_roi is None else x_roi).astype(np.float32)
        out = sess.run(None, feed)[0]
        return np.asarray(out, dtype=np.float64)

    rej_th = RejectionThresholds.from_config(meta.get("reject"))
    q_cfg = meta.get("quality")
    q_th = (
        IceTypeQualityThresholds.from_config(q_cfg)
        if isinstance(q_cfg, dict)
        else IceTypeQualityThresholds(tau_q=rej_th.tau_q)
    )
    infer_cfg = IceTypeInferConfig(
        image_size=int(meta.get("image_size", 224)),
        roi_size=int(meta.get("roi_size", 224)),
        pad_value=int(meta.get("pad_value", 114)),
        view_mode=vm,
        model_name=str(meta.get("model_name", "mobilenetv3_small_100")),
        model_version=str(meta.get("model_version", model_p.stem)),
        quality_thresholds=q_th,
        rejection_thresholds=rej_th,
    )
    scaler = MultiClassTemperatureScaler(
        temperature=float(meta.get("temperature", 1.0)),
        fitted=True,
    )
    inferencer = IceTypeInferencer(logits_fn=_logits_fn, scaler=scaler, cfg=infer_cfg)

    roi_map: dict[str, NormalizedRoi] = {}
    csv_candidate = roi_csv if roi_csv is not None else meta.get("roi_csv")
    if csv_candidate:
        cp = Path(str(csv_candidate))
        if not cp.is_absolute():
            cp = paths.PROJ / cp
        if cp.exists():
            roi_map = load_monitor_roi_table(cp)

    return OnnxIceTypeService(
        inferencer=inferencer,
        meta=meta,
        roi_map=roi_map,
        onnx_path=model_p,
    )
