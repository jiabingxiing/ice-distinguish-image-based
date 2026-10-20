"""训练引擎与 AMP 策略单测（A3、A6）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from grid_ice_vision.training import engine
from grid_ice_vision.training.engine import (
    TrainConfig,
    build_optimizer,
    evaluate,
    fit,
    resolve_amp,
)


class _DS(Dataset[dict[str, Tensor]]):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return {"image": torch.zeros(2), "y": torch.tensor(float(i))}


def _loader(n: int, bs: int = 2) -> DataLoader[Any]:
    return DataLoader(_DS(n), batch_size=bs)


def _step(
    model: nn.Module, batch: dict[str, Tensor], phase: str
) -> tuple[Tensor, dict[str, float]]:
    pred = model(batch["image"]).squeeze(-1)
    loss = torch.nn.functional.mse_loss(pred, batch["y"])
    return loss, {"mae": float(loss.detach())}


def test_empty_val_raises() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="验证集为空"):
        fit(model, _loader(4), _loader(0), _step, TrainConfig(epochs=1))


def test_empty_train_raises() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="训练集为空"):
        fit(model, _loader(0), _loader(4), _step, TrainConfig(epochs=1))


def test_evaluate_empty_raises() -> None:
    with pytest.raises(ValueError, match="验证集为空"):
        evaluate(nn.Linear(2, 1), _loader(0), _step)


def test_missing_monitor_raises() -> None:
    """监控指标缺失时必须报错：原实现会退化成"最大化 loss"。"""
    model = nn.Linear(2, 1)
    with pytest.raises(KeyError, match="监控指标"):
        fit(
            model,
            _loader(4),
            _loader(2),
            _step,
            TrainConfig(epochs=1),
            monitor="dice_ice",
            mode="max",
        )


def test_fit_writes_best_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    best = fit(
        model,
        _loader(4),
        _loader(2),
        _step,
        TrainConfig(epochs=2, out_dir=str(tmp_path), task="seg", version="smoke"),
        monitor="mae",
        mode="min",
    )
    out = tmp_path / "seg" / "smoke"
    assert (out / "best.pt").exists()
    assert (out / "epoch000.pt").exists()
    assert best["mae"] > 0
    assert "epoch" in best


# ------------------------------------------------------------------ AMP
def test_cpu_auto_falls_back_to_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "cpu_bf16_available", lambda: False)
    p = resolve_amp("auto", "cpu")
    assert (p.enabled, p.use_scaler, p.dtype) == (False, False, None)
    assert "fp32" in p.reason


def test_cpu_bf16_uses_no_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "cpu_bf16_available", lambda: True)
    p = resolve_amp("bf16", "cpu")
    assert p.enabled is True
    assert p.dtype is torch.bfloat16
    assert p.use_scaler is False


def test_cpu_fp16_is_rejected() -> None:
    p = resolve_amp("fp16", "cpu")
    assert p.enabled is False
    assert "fp16" in p.reason


def test_cuda_fp16_uses_scaler() -> None:
    p = resolve_amp("auto", "cuda:0")
    assert p.enabled is True
    assert p.dtype is torch.float16
    assert p.use_scaler is True


def test_cuda_off() -> None:
    p = resolve_amp(False, "cuda:0")
    assert p.enabled is False


def test_unknown_amp_value_raises() -> None:
    with pytest.raises(ValueError, match="未知 amp"):
        resolve_amp("half", "cpu")


# ------------------------------------------------------------------ 优化器
class _WithBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 4)
        self.head = nn.Linear(4, 1)


def test_backbone_lr_multiplier_splits_groups() -> None:
    opt = build_optimizer(_WithBackbone(), TrainConfig(lr=1e-3, backbone_lr_mult=0.1))
    lrs = sorted(round(g["lr"], 6) for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3]


def test_default_single_group() -> None:
    opt = build_optimizer(_WithBackbone(), TrainConfig(lr=1e-3))
    assert len(opt.param_groups) == 1
