"""通用训练引擎：单卡、AMP、梯度裁剪、早停与 checkpoint。

刻意保持极简：不引入 PyTorch Lightning / mmengine，避免依赖链与许可风险。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from grid_ice_vision.common.logging import get_logger

log = get_logger("train")


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 5.0
    patience: int = 8
    out_dir: str = "artifacts"
    task: str = "seg"
    version: str = "0.1.0"
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.3)


class EarlyStopper:
    def __init__(self, patience: int = 8, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.bad = 0

    def step(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad = 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[nn.Module, Any, str], tuple[Tensor, dict[str, float]]],
    device: str = "cpu",
    amp: bool = True,
    grad_clip: float = 5.0,
    epoch: int = 0,
) -> EpochMetrics:
    """`step_fn(model, batch, phase) -> (loss, metrics)` 由各任务自行实现。"""
    model.train()
    total, n = 0.0, 0
    agg: dict[str, float] = {}
    scaler = torch.amp.GradScaler(enabled=amp and device.startswith("cuda"))

    for i, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            enabled=amp and device.startswith("cuda"),
        ):
            loss, metrics = step_fn(model, batch, "train")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = len(batch["image"]) if isinstance(batch, dict) and "image" in batch else 1
        total += float(loss.detach()) * bs
        n += bs
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
        if i % 50 == 0:
            log.info("epoch %d iter %d loss=%.4f", epoch, i, float(loss.detach()))

    return EpochMetrics(loss=total / max(n, 1), metrics={k: v / max(n, 1) for k, v in agg.items()})


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    step_fn: Callable[[nn.Module, Any, str], tuple[Tensor, dict[str, float]]],
    device: str = "cpu",
) -> EpochMetrics:
    model.eval()
    total, n = 0.0, 0
    agg: dict[str, float] = {}
    for batch in loader:
        loss, metrics = step_fn(model, batch, "val")
        bs = len(batch["image"]) if isinstance(batch, dict) and "image" in batch else 1
        total += float(loss) * bs
        n += bs
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
    return EpochMetrics(loss=total / max(n, 1), metrics={k: v / max(n, 1) for k, v in agg.items()})


def save_checkpoint(
    model: nn.Module, cfg: TrainConfig, metrics: dict[str, float], epoch: int
) -> Path:
    out = Path(cfg.out_dir) / cfg.task / cfg.version
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"epoch{epoch:03d}.pt"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "epoch": epoch}, path)
    best = out / "best.pt"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "epoch": epoch}, best)
    return path


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    step_fn: Callable[[nn.Module, Any, str], tuple[Tensor, dict[str, float]]],
    cfg: TrainConfig,
    monitor: str = "loss",
    mode: str = "min",
    device: str = "cpu",
) -> dict[str, float]:
    model.to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    stopper = EarlyStopper(cfg.patience, mode)
    best: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(
            model, train_loader, opt, step_fn, device, cfg.amp, cfg.grad_clip, epoch
        )
        va = evaluate(model, val_loader, step_fn, device)
        sched.step()
        log.info(
            "epoch %d | train_loss=%.4f | val_loss=%.4f | %s",
            epoch,
            tr.loss,
            va.loss,
            " ".join(f"{k}={v:.3f}" for k, v in va.metrics.items()),
        )
        value = va.metrics.get(monitor, va.loss)
        if stopper.step(value):
            best = {**va.metrics, "loss": va.loss, "epoch": float(epoch)}
            save_checkpoint(model, cfg, best, epoch)
        if stopper.should_stop:
            log.info("早停于 epoch %d（best=%s=%.4f）", epoch, monitor, stopper.best)
            break
    return best
