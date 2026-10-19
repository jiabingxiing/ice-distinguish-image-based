"""检测损失：FCOS 式点分配 + Quality Focal + Distribution Focal + GIoU。

为什么不用 IoU 做分配：导线在 1080P 图里常只有 8–20 px 宽，IoU 对
细长目标的微小偏移极其敏感，正样本会大量流失。点分配 + 距离回归更稳。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from grid_ice_vision.training.losses import (
    distribution_focal_loss,
    giou_loss,
    quality_focal_loss,
)


@dataclass
class DetTargets:
    boxes: Tensor  # (N, 4) xyxy
    labels: Tensor  # (N,)


def box_iou_paired(a: Tensor, b: Tensor) -> Tensor:
    """逐元素 IoU：a/b 形状均为 (N,4)，返回 (N,)。用于正样本与匹配 GT 的一一比对。"""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((0,), device=a.device)
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-7)


def box_iou(a: Tensor, b: Tensor) -> Tensor:
    """成对 IoU 矩阵，a/b 形状均为 (N,4)，返回 (N,N)。"""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0],), device=a.device)
    lt = torch.max(a[:, None, :2], b[:, None, :2])
    rb = torch.min(a[:, None, 2:], b[:, None, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-7
    return inter / union


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 8,
        strides: tuple[int, ...] = (8, 16, 32),
        center_radius: float = 1.5,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        lambda_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.center_radius = center_radius
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_giou = lambda_giou

    def _level_ranges(self, device: torch.device) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for i, s in enumerate(self.strides):
            lo = 0.0 if i == 0 else float(self.strides[i - 1] * 4)
            hi = float(s * self.reg_max) if i < len(self.strides) - 1 else float("inf")
            ranges.append((lo, hi))
        return ranges

    @torch.no_grad()
    def assign(
        self,
        points: Tensor,
        stride: int,
        boxes: Tensor,
        labels: Tensor,
        level_range: tuple[float, float],
    ) -> tuple[Tensor, Tensor]:
        """返回 (matched_gt_idx (A,), matched_label (A,))，未匹配为 -1。"""
        a = points.shape[0]
        matched = torch.full((a,), -1, dtype=torch.long, device=points.device)
        matched_label = torch.full((a,), -1, dtype=torch.long, device=points.device)
        if boxes.numel() == 0:
            return matched, matched_label

        px = points[:, 0:1]
        py = points[:, 1:2]
        left = px - boxes[:, 0]
        t = py - boxes[:, 1]
        r = boxes[:, 2] - px
        b = boxes[:, 3] - py
        inside = torch.minimum(torch.minimum(left, r), torch.minimum(t, b)) > 0

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        radius = self.center_radius * stride
        center_ok = (px - cx).abs() < radius
        center_ok = center_ok & ((py - cy).abs() < radius)

        max_dist = torch.maximum(torch.maximum(left, r), torch.maximum(t, b))
        lo, hi = level_range
        in_range = (max_dist >= lo) & (max_dist < hi)

        cand = inside & center_ok & in_range
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_big = torch.where(
            cand, area.unsqueeze(0), torch.full_like(area.unsqueeze(0), float("inf"))
        )
        best = area_big.argmin(dim=1)
        has = cand.any(dim=1)
        matched[has] = best[has]
        matched_label[has] = labels[best[has]]
        return matched, matched_label

    def forward(
        self, outputs: list[dict[str, Any]], targets: list[DetTargets]
    ) -> tuple[Tensor, dict[str, float]]:
        device = outputs[0]["cls"].device
        batch = outputs[0]["batch"]
        ranges = self._level_ranges(device)

        cls_cat: list[Tensor] = []
        reg_cat: list[Tensor] = []
        points_cat: list[Tensor] = []
        stride_cat: list[Tensor] = []
        for out, _rng in zip(outputs, ranges, strict=True):
            cls_cat.append(out["cls"])
            reg_cat.append(out["reg"])
            points_cat.append(out["points"])
            stride_cat.append(
                torch.full((out["points"].shape[0],), float(out["stride"]), device=device)
            )

        cls_all = torch.cat(cls_cat, dim=1)  # (B, A, C)
        reg_all = torch.cat(reg_cat, dim=1)  # (B, A, 4*(reg_max+1))
        points = torch.cat(points_cat, dim=0)  # (A, 2)
        strides = torch.cat(stride_cat, dim=0)  # (A,)

        from grid_ice_vision.features.detection.model import (
            distance2bbox,
            distribution_expectation,
        )

        # 逐点按其所属层的 stride 反算距离（不同层的 stride 不同，不能共用）
        dist_px = distribution_expectation(reg_all.reshape(-1, reg_all.shape[-1]), self.reg_max)
        dist_px = dist_px.view(batch, -1, 4) * strides.view(1, -1, 1)
        pred_boxes = distance2bbox(points.unsqueeze(0).expand(batch, -1, -1), dist_px)  # (B, A, 4)

        quality = torch.zeros((batch, points.shape[0], self.num_classes), device=device)
        pos_mask = torch.zeros((batch, points.shape[0]), dtype=torch.bool, device=device)
        pos_reg_target = torch.zeros((batch, points.shape[0], 4), device=device)
        pos_boxes_pred: list[Tensor] = []
        pos_boxes_gt: list[Tensor] = []

        # 逐图分配（点数量级 1e4，循环 batch 足够快）
        level_slices: list[tuple[int, int]] = []
        start = 0
        for out in outputs:
            n = out["points"].shape[0]
            level_slices.append((start, start + n))
            start += n

        for b in range(batch):
            gt = (
                targets[b]
                if b < len(targets)
                else DetTargets(
                    boxes=torch.zeros((0, 4), device=device),
                    labels=torch.zeros((0,), dtype=torch.long, device=device),
                )
            )
            matched = torch.full((points.shape[0],), -1, dtype=torch.long, device=device)
            labels_m = torch.full((points.shape[0],), -1, dtype=torch.long, device=device)
            for (s0, s1), out, rng in zip(level_slices, outputs, ranges, strict=True):
                m, lab = self.assign(
                    out["points"],
                    int(out["stride"]),
                    gt.boxes.to(device),
                    gt.labels.to(device),
                    rng,
                )
                matched[s0:s1] = m
                labels_m[s0:s1] = lab

            has = matched >= 0
            if not bool(has.any()):
                continue
            idx = matched[has]
            gt_boxes = gt.boxes.to(device)[idx]
            iou = box_iou_paired(pred_boxes[b][has], gt_boxes).clamp(min=0.0)
            quality[b, has, labels_m[has]] = iou
            pos_mask[b, has] = True

            # 真实距离：点到 GT 四边（像素）→ 除以 stride 变特征图单位
            px = points[has, 0]
            py = points[has, 1]
            left = px - gt_boxes[:, 0]
            t = py - gt_boxes[:, 1]
            r = gt_boxes[:, 2] - px
            btm = gt_boxes[:, 3] - py
            dist = torch.stack([left, t, r, btm], dim=1).clamp(min=0.0)
            pos_reg_target[b, has] = dist / strides[has].unsqueeze(1)

            pos_boxes_pred.append(pred_boxes[b][has])
            pos_boxes_gt.append(gt_boxes)

        cls_loss = quality_focal_loss(
            cls_all.reshape(-1, self.num_classes), quality.reshape(-1, self.num_classes)
        )
        if not pos_mask.any():
            return cls_loss, {
                "cls": float(cls_loss.detach()),
                "reg": 0.0,
                "giou": 0.0,
                "n_pos": 0.0,
            }

        pred_reg = reg_all[pos_mask]
        tgt_reg = pos_reg_target[pos_mask]
        reg_loss = distribution_focal_loss(pred_reg, tgt_reg)
        giou = giou_loss(torch.cat(pos_boxes_pred, dim=0), torch.cat(pos_boxes_gt, dim=0))

        total = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss + self.lambda_giou * giou
        return total, {
            "cls": float(cls_loss.detach()),
            "reg": float(reg_loss.detach()),
            "giou": float(giou.detach()),
            "n_pos": float(pos_mask.sum()),
        }
