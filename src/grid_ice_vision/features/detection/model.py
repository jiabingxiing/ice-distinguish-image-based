"""T2 部件 ROI 检测（§10.3）。

选型：`timm backbone + 自建 GFL 头`（Apache-2.0 / MIT 依赖链），**不用 Ultralytics**
（AGPL-3.0，私有部署需商业授权）。代价是多花 3–5 天自建检测头，收益是
彻底规避许可传染 —— 对电网企业交付，这个交换值得。

实现是 GFL 的简化版（FCOS 式点分配 + 质量焦点损失 + 分布回归），
足够跑通导线/绝缘子/杆塔/参考金具四类，且无第三方检测框架依赖。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grid_ice_vision.common.constants import ASSET_CLASSES
from grid_ice_vision.features.backbone import Backbone, ConvBNAct, build_backbone


def _make_points(h: int, w: int, stride: int, device: torch.device) -> Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


class FPN(nn.Module):
    """轻量 FPN：1×1 侧向 + 自顶向下相加 + 3×3 平滑。"""

    def __init__(self, in_channels: list[int], out_channels: int = 64) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(out_channels, out_channels, 3) for _ in in_channels])

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral, feats, strict=True)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        return [smooth(x) for smooth, x in zip(self.smooth, laterals, strict=True)]


class GFLHead(nn.Module):
    """Distribution-Guided Quality Focal Loss 头（简化版）。"""

    def __init__(
        self, in_channels: int, num_classes: int, feat_channels: int = 64, reg_max: int = 8
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_tower = nn.Sequential(
            ConvBNAct(in_channels, feat_channels, 3), ConvBNAct(feat_channels, feat_channels, 3)
        )
        self.reg_tower = nn.Sequential(
            ConvBNAct(in_channels, feat_channels, 3), ConvBNAct(feat_channels, feat_channels, 3)
        )
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 1)

        # focal 初始化：初始正样本概率 ≈ 0.01，避免训练初期梯度爆炸
        assert self.cls_pred.bias is not None
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.zeros_(self.reg_pred.weight)
        assert self.reg_pred.bias is not None
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cls = self.cls_pred(self.cls_tower(x))
        reg = self.reg_pred(self.reg_tower(x))
        b, _, h, w = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(b, h * w, self.num_classes)
        reg = reg.permute(0, 2, 3, 1).reshape(b, h * w, 4 * (self.reg_max + 1))
        return cls, reg


class DetectionNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = False,
        num_classes: int = len(ASSET_CLASSES),
        fpn_channels: int = 64,
        reg_max: int = 8,
    ) -> None:
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained)
        chans = list(self.backbone.spec.channels)
        strides = list(self.backbone.spec.strides)
        # 取最后 3 层（stride 8/16/32），导线宽度只有 8–20 px，太深会丢细节
        self.fpn = FPN(chans[-3:], fpn_channels)
        self.heads = nn.ModuleList(
            [GFLHead(fpn_channels, num_classes, fpn_channels, reg_max) for _ in range(3)]
        )
        self.strides = tuple(strides[-3:])
        self.num_classes = num_classes
        self.reg_max = reg_max

    def forward(self, x: Tensor) -> list[dict[str, Any]]:
        feats = self.backbone(x)
        pyramids = self.fpn(feats[-3:])
        outs: list[dict[str, Any]] = []
        for head, fmap, stride in zip(self.heads, pyramids, self.strides, strict=True):
            cls, reg = head(fmap)
            b, _, h, w = fmap.shape
            outs.append(
                {
                    "cls": cls,
                    "reg": reg,
                    "points": _make_points(h, w, stride, fmap.device),
                    "stride": stride,
                    "shape": (h, w),
                    "batch": b,
                }
            )
        return outs


def distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """(x, y, l, t, r, b) → xyxy。支持 (A,2)/(A,4) 与 (B,A,2)/(B,A,4)。"""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def distribution_expectation(reg: Tensor, reg_max: int) -> Tensor:
    """离散分布 → 期望距离（特征图单位）。

    支持 `(B, A, 4·(reg_max+1))` 与 `(N, 4·(reg_max+1))` 两种形状。
    """
    lead = reg.shape[:-1]
    probs = reg.reshape(-1, 4, reg_max + 1).softmax(dim=-1)
    bins = torch.arange(reg_max + 1, device=reg.device, dtype=reg.dtype)
    return (probs * bins).sum(dim=-1).reshape(*lead, 4)


def decode_level(
    cls: Tensor, reg: Tensor, points: Tensor, stride: int, reg_max: int, score_thr: float = 0.05
) -> tuple[Tensor, Tensor, Tensor]:
    """解码单层输出 → (boxes, scores, labels)，仅保留过阈值的点。"""
    scores_all = cls.sigmoid()
    scores, labels = scores_all.max(dim=-1)
    keep = scores > score_thr
    if not bool(keep.any()):
        return (
            torch.zeros((0, 4), device=cls.device),
            torch.zeros((0,), device=cls.device),
            torch.zeros((0,), dtype=torch.long, device=cls.device),
        )
    dist = distribution_expectation(reg, reg_max) * float(stride)
    boxes = distance2bbox(points.expand(dist.shape[0], -1), dist)
    return boxes[keep], scores[keep], labels[keep]
