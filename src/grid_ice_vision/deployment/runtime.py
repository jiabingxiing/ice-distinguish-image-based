"""ONNXRuntime 封装：与 PyTorch 侧同一套预处理，避免训练/部署不一致。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OnnxRuntimeModel:
    def __init__(self, path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {p}")
        self.session = ort.InferenceSession(str(p), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})

    def info(self) -> dict[str, Any]:
        return {
            "inputs": [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            "outputs": [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
            "providers": self.session.get_providers(),
        }
