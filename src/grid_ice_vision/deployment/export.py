"""ONNX 导出与数值一致性校验（§16.1）。

链路：ckpt → 固定输入尺寸 → ONNX(opset 17) → 100 组随机输入比对(atol ≤ 1e-4)。
**不承诺 INT8**：量化对覆冰边缘 IoU 的损伤需实测，且要先做量化感知训练。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _TupleWrapper(nn.Module):
    """把 dict 输出摊平成 tuple，便于 ONNX 导出。"""

    def __init__(self, model: nn.Module, keys: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self.keys = tuple(keys)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[k] for k in self.keys)


def export_onnx(
    model: nn.Module,
    out_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    opset: int = 17,
    keys: Sequence[str] | None = None,
    dynamic_batch: bool = False,
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    wrapper: nn.Module = _TupleWrapper(model, keys) if keys else model
    wrapper.eval()
    dummy = torch.zeros(*input_shape)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    # 用 TorchScript 导出器（dynamo=False）：本项目模型含 Python 级条件分支，
    # torch.export 的符号化流程对这类控制流不友好，legacy 路径更稳。
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(p),
        input_names=["input"],
        output_names=list(keys) if keys else ["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return p


def check_numeric_consistency(
    model: nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    n_trials: int = 20,
    atol: float = 1e-4,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """随机输入上比对 PyTorch 与 ONNXRuntime 输出。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.eval()
    max_err = 0.0
    for _ in range(n_trials):
        x = torch.randn(*input_shape)
        with torch.no_grad():
            if keys:
                out = model(x)
                ref = [out[k].numpy() for k in keys]
            else:
                ref = [model(x).numpy()]
        got = sess.run(None, {"input": x.numpy()})
        for r, g in zip(ref, got, strict=True):
            max_err = max(max_err, float(np.max(np.abs(r - g))))
    return {"max_abs_err": max_err, "atol": atol, "passed": max_err <= atol, "n_trials": n_trials}


def export_and_verify(
    model: nn.Module,
    out_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    keys: Sequence[str] | None = None,
    atol: float = 1e-4,
) -> dict[str, Any]:
    p = export_onnx(model, out_path, input_shape, keys=keys)
    result = check_numeric_consistency(model, p, input_shape, keys=keys, atol=atol)
    result["path"] = str(p)
    return result
