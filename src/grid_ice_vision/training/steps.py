"""各任务的单步计算（供 engine.fit 调用）。

把「模型 + 损失 + 指标」封装成 `step_fn(model, batch, phase)`，训练引擎与
CLI 都不需要知道任务细节。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from grid_ice_vision.training.losses import (
    CoralLoss,
    SegmentationLoss,
    coral_predict,
)


def precip_step(
    model: nn.Module, batch: dict[str, Any], phase: str
) -> tuple[Tensor, dict[str, float]]:
    """天气多标签 BCE + 强度 CORAL。

    `intensity < 0`（unknown）的样本由 `CoralLoss(ignore_index=-1)` 自动跳过，
    不计入强度损失与准确率 —— 否则会被当成 "none" 监督。
    """
    image = batch["image"]
    weather: Tensor = batch["weather"]
    intensity: Tensor = batch["intensity"]
    met = batch.get("met")
    # 可选类别掩码：某类没有正样本（如 WEAPD 缺 clear）时屏蔽该维
    mask: Tensor | None = batch.get("weather_mask")

    out = model(image, met)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        out["weather_logits"], weather, reduction="none"
    )
    bce = raw.mean() if mask is None else (raw * mask).sum() / mask.sum().clamp_min(1.0)

    valid = intensity >= 0
    coral = CoralLoss(int(out["intensity_logits"].shape[1]) + 1)(out["intensity_logits"], intensity)
    loss = bce + 0.5 * coral

    with torch.no_grad():
        pred = (out["weather_logits"].sigmoid() > 0.5).float()
        if mask is not None:
            pred, target = pred * mask, weather * mask
        else:
            target = weather
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
        metrics = {
            "bce": float(bce.detach()),
            "macro_f1_micro": f1,
            "n_intensity_valid": float(valid.sum()),
        }
        if bool(valid.any()):
            level = coral_predict(out["intensity_logits"])
            metrics["intensity_acc"] = (level[valid] == intensity[valid]).float().mean().item()
    return loss, metrics


def seg_step(
    model: nn.Module, batch: dict[str, Any], phase: str
) -> tuple[Tensor, dict[str, float]]:
    """4 类分割：FN-aware Dice + Focal + Boundary。"""
    logits = model(batch["image"])
    target: Tensor = batch["mask"]
    criterion = SegmentationLoss(num_classes=logits.shape[1])
    loss = criterion(logits, target)
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        ice_p, ice_t = pred == 2, target == 2
        inter = (ice_p & ice_t).sum().float()
        dice = (2 * inter / (ice_p.sum() + ice_t.sum() + 1e-6)).item()
    return loss, {"dice_ice": dice}


def det_step(
    model: nn.Module, batch: dict[str, Any], phase: str
) -> tuple[Tensor, dict[str, float]]:
    from grid_ice_vision.features.detection.loss import DetectionLoss, DetTargets

    outs = model(batch["image"])
    targets = [DetTargets(boxes=t["boxes"], labels=t["labels"]) for t in batch["targets"]]
    # reg_max / strides 必须与模型一致，否则分布回归的通道数会对不上
    criterion = DetectionLoss(
        num_classes=outs[0]["cls"].shape[-1],
        reg_max=int(getattr(model, "reg_max", 8)),
        strides=tuple(getattr(model, "strides", (8, 16, 32))),
    )
    loss, metrics = criterion(outs, targets)
    return loss, metrics


def qwk_step_placeholder() -> None:  # pragma: no cover - 预留接口
    """占位：等级 QWK 在 evaluation/classification.py 中计算，不进训练循环。"""
    return None


__all__ = ["det_step", "precip_step", "seg_step"]
