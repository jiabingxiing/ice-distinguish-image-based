"""通用训练引擎：单卡、AMP 策略、梯度裁剪、早停与 checkpoint。

刻意保持极简：不引入 PyTorch Lightning / mmengine，避免依赖链与许可风险。

设计原则：**宁可训练前报错，也不要训练两小时后拿到一个看起来正常的废模型**。
因此空数据集、指标缺失、监控值非有限、标注未生效等情形一律显式失败。
"""

from __future__ import annotations

import math
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
    # bool 保持向后兼容（旧 yaml 的 true/false）；str 支持 auto|off|bf16|fp16
    amp: bool | str = "auto"
    grad_clip: float = 5.0
    patience: int = 8
    out_dir: str = "artifacts"
    task: str = "seg"
    version: str = "0.1.0"
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    # 预训练骨干用小学习率，头用大学习率；1.0 = 不分组（旧行为）
    backbone_lr_mult: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


# ------------------------------------------------------------------ AMP 策略
@dataclass(frozen=True)
class AmpPolicy:
    enabled: bool
    dtype: torch.dtype | None
    use_scaler: bool
    reason: str


def cpu_bf16_available() -> bool:
    """CPU 是否有 bf16 加速单元（AVX512-BF16 / AMX）。

    本机 Ryzen 7 4800U 是 Zen 2（AVX2），返回 False —— 此时强行开 bf16
    只会更慢或走模拟路径，因此显式回退 fp32 并记录原因。
    """
    try:
        if torch.cpu._is_avx512_bf16_supported():  # type: ignore[attr-defined]
            return True
        return bool(torch.cpu._is_amx_tile_supported())  # type: ignore[attr-defined]
    except Exception:
        return False


def resolve_amp(amp: bool | str, device: str) -> AmpPolicy:
    """把配置里的 amp 解析成明确的执行策略（含理由，便于审计）。"""
    is_cuda = device.startswith("cuda")
    if is_cuda:
        if amp in (False, "off"):
            return AmpPolicy(False, None, False, "显式关闭")
        if amp == "bf16":
            return AmpPolicy(True, torch.bfloat16, True, "cuda bf16 + scaler")
        return AmpPolicy(True, torch.float16, True, "cuda fp16 + scaler")

    if amp in (False, "off"):
        return AmpPolicy(False, None, False, "显式关闭")
    if amp == "fp16":
        return AmpPolicy(False, None, False, "CPU 不支持 fp16 autocast，回退 fp32")
    if amp in (True, "auto", "bf16"):
        if cpu_bf16_available():
            return AmpPolicy(True, torch.bfloat16, False, "cpu bf16（无需 scaler）")
        return AmpPolicy(False, None, False, "CPU 无 AVX512-BF16/AMX，bf16 无加速，回退 fp32")
    raise ValueError(f"未知 amp 取值: {amp!r}")


# ------------------------------------------------------------------ 优化器
def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """骨干与头分组学习率。

    预训练骨干配统一 1e-3 会把 ImageNet 特征打坏；`backbone_lr_mult=0.1`
    是微调小数据的常规做法。`1.0` 时等价于原来的单组优化器。
    """
    if cfg.backbone_lr_mult == 1.0:
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad
    ]
    groups: list[dict[str, Any]] = []
    if bb:
        groups.append({"params": bb, "lr": cfg.lr * cfg.backbone_lr_mult})
    if head:
        groups.append({"params": head, "lr": cfg.lr})
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


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


def _dataset_len(loader: DataLoader[Any]) -> int:
    ds = getattr(loader, "dataset", None)
    try:
        return len(ds)  # type: ignore[arg-type]
    except TypeError:
        return -1


def _check_loader(loader: DataLoader[Any], what: str) -> None:
    n = _dataset_len(loader)
    if n == 0:
        raise ValueError(
            f"{what}为空：拒绝训练/评估。空验证集会让 EarlyStopper 把 loss=0 当成最优，"
            "从而把随机初始化的权重保存为 best.pt。"
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[nn.Module, Any, str], tuple[Tensor, dict[str, float]]],
    device: str = "cpu",
    amp: bool | str = "auto",
    grad_clip: float = 5.0,
    epoch: int = 0,
) -> EpochMetrics:
    """`step_fn(model, batch, phase) -> (loss, metrics)` 由各任务自行实现。"""
    model.train()
    policy = resolve_amp(amp, device)
    if epoch == 0:
        log.info("AMP 策略：%s", policy.reason)

    total, n = 0.0, 0
    agg: dict[str, float] = {}
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(dev_type, enabled=policy.use_scaler)

    for i, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, dtype=policy.dtype, enabled=policy.enabled):
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

    return EpochMetrics(
        loss=total / max(n, 1),
        metrics={k: v / max(n, 1) for k, v in agg.items()},
        n_samples=n,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    step_fn: Callable[[nn.Module, Any, str], tuple[Tensor, dict[str, float]]],
    device: str = "cpu",
) -> EpochMetrics:
    _check_loader(loader, "验证集")
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
    return EpochMetrics(
        loss=total / max(n, 1),
        metrics={k: v / max(n, 1) for k, v in agg.items()},
        n_samples=n,
    )


def save_checkpoint(
    model: nn.Module, cfg: TrainConfig, metrics: dict[str, float], epoch: int
) -> Path:
    out = Path(cfg.out_dir) / cfg.task / cfg.version
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"epoch{epoch:03d}.pt"
    payload = {"state_dict": model.state_dict(), "metrics": metrics, "epoch": epoch}
    torch.save(payload, path)
    torch.save(payload, out / "best.pt")
    return path


def _monitor_value(va: EpochMetrics, monitor: str) -> float:
    """取监控值。缺失时**报错**——原来的 `metrics.get(monitor, va.loss)`
    在 mode="max" 下会去最大化 loss，方向完全反了。"""
    if monitor == "loss":
        return va.loss
    if monitor not in va.metrics:
        raise KeyError(
            f"监控指标 {monitor!r} 不在验证指标 {sorted(va.metrics)} 中；"
            "请检查 step_fn 的 metrics 键名，或改用 monitor='loss'"
        )
    return va.metrics[monitor]


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
    _check_loader(train_loader, "训练集")
    _check_loader(val_loader, "验证集")

    model.to(device)
    spec = getattr(getattr(model, "backbone", None), "spec", None)
    if spec is not None:
        log.info(
            "骨干：%s | pretrained=%s | fallback=%s %s",
            spec.name,
            spec.pretrained,
            spec.is_fallback,
            f"({spec.fallback_reason})" if spec.fallback_reason else "",
        )

    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    stopper = EarlyStopper(cfg.patience, mode)
    best: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(
            model, train_loader, opt, step_fn, device, cfg.amp, cfg.grad_clip, epoch
        )
        # 检测任务专用兜底：整轮没有正样本说明标注没生效
        if "n_pos" in tr.metrics and tr.metrics["n_pos"] <= 0.0:
            raise RuntimeError(
                "训练集整轮 n_pos=0：检测标注未生效（COCO 路径/文件名不匹配），"
                "继续训练只会得到全背景模型"
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

        value = _monitor_value(va, monitor)
        if not math.isfinite(value):
            log.warning("epoch %d 的 %s 非有限值（%s），跳过保存", epoch, monitor, value)
            continue
        if stopper.step(value):
            best = {**va.metrics, "loss": va.loss, "epoch": float(epoch)}
            save_checkpoint(model, cfg, best, epoch)
        if stopper.should_stop:
            log.info("早停于 epoch %d（best=%s=%.4f）", epoch, monitor, stopper.best)
            break

    if not best:
        raise RuntimeError(f"训练结束但从未产出可用 checkpoint（monitor={monitor} 始终非有限？）")
    return best
