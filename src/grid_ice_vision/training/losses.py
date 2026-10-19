"""损失函数集合（§10.2/§10.4/§10.5）。

命名与论文一致，便于对照：
    DiceLoss / FocalLoss / BoundaryWeightedBCE / CoralLoss / ClassBalancedCE
    QualityFocalLoss / DistributionFocalLoss / GIoULoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ------------------------------------------------------------------ 分割损失
class DiceLoss(nn.Module):
    """FN-aware Dice：漏检权重高于误检。

    标准 Dice 对 FN 与 FP 等权，但漏掉覆冰的代价远高于多标一块冰。
    权重只加在分母的 FN 项上：`w_FN=2.0` 时 FN 的惩罚是 FP 的两倍。
    """

    def __init__(self, fn_weight: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.fn_weight = fn_weight
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        tp = (prob * target).sum()
        fn = ((1.0 - prob) * target).sum() * self.fn_weight
        fp = (prob * (1.0 - target)).sum()
        return 1.0 - (2.0 * tp + self.eps) / (2.0 * tp + fn + fp + self.eps)


class FocalLoss(nn.Module):
    """二值 Focal：抑制易分负样本（大片背景）。"""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def boundary_band(target: Tensor, kernel: int = 3) -> Tensor:
    """由二值目标图提取边界带（0/1）。"""
    t = target.float().unsqueeze(1)
    pad = kernel // 2
    dil = F.max_pool2d(t, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-t, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0).squeeze(1)


class BoundaryLoss(nn.Module):
    """边界加权 BCE。

    覆冰厚度精度完全取决于边界位置，因此边界像素的误差要单独放大。
    """

    def __init__(self, weight: float = 5.0, kernel: int = 5) -> None:
        super().__init__()
        self.weight = weight
        self.kernel = kernel

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        band = boundary_band(target, self.kernel)
        w = 1.0 + self.weight * band
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        return (loss * w).mean()


class SegmentationLoss(nn.Module):
    """L_seg = λ_dice·Dice + λ_focal·Focal + λ_bnd·Boundary（默认 1.0/1.0/0.5）。

    只在**前景类**（ice_accretion / icicle_bridge）上算 Dice 与 Boundary：
    背景与裸露设备由 Focal 负责压制，否则背景占比过高会淹没梯度。
    """

    def __init__(
        self,
        num_classes: int = 4,
        foreground: tuple[int, ...] = (2, 3),
        ice_class: int = 2,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_bnd: float = 0.5,
        fn_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.foreground = foreground
        self.ice_class = ice_class
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_bnd = lambda_bnd
        self.dice = DiceLoss(fn_weight=fn_weight)
        self.focal = FocalLoss()
        self.bnd = BoundaryLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """logits: (B, C, H, W)；target: (B, H, W) 的 long 类别索引。"""
        dice_term = logits.new_zeros(())
        for c in self.foreground:
            dice_term = dice_term + self.dice(logits[:, c], (target == c).float())
        dice_term = dice_term / max(len(self.foreground), 1)

        focal_term = logits.new_zeros(())
        for c in self.foreground:
            focal_term = focal_term + self.focal(logits[:, c], (target == c).float())
        focal_term = focal_term / max(len(self.foreground), 1)

        bnd_term = self.bnd(logits[:, self.ice_class], (target == self.ice_class).float())

        return (
            self.lambda_dice * dice_term
            + self.lambda_focal * focal_term
            + self.lambda_bnd * bnd_term
        )

    @torch.no_grad()
    def components(self, logits: Tensor, target: Tensor) -> dict[str, float]:
        """消融实验用：返回各分量数值。"""
        return {
            "dice": float(
                self.dice(logits[:, self.foreground[0]], (target == self.foreground[0]).float())
            ),
            "focal": float(
                self.focal(logits[:, self.foreground[0]], (target == self.foreground[0]).float())
            ),
            "boundary": float(
                self.bnd(logits[:, self.ice_class], (target == self.ice_class).float())
            ),
        }


# --------------------------------------------------------------- 有序回归损失
class CoralLoss(nn.Module):
    """CORAL：K−1 个阈值共享权重、单一递增 bias。

    target: (B,) 取值 0..K−1；logits: (B, K−1)。
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 至少为 2")
        self.num_classes = num_classes

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        binary = (target.unsqueeze(1) > levels.unsqueeze(0)).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, binary)


def coral_predict(logits: Tensor) -> Tensor:
    """推理：sum(sigmoid(logits) > 0.5)，沿 K−1 维。

    对全负 logits 返回 0、全正返回 K−1，天然保证单调。
    """
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class CoralHead(nn.Module):
    """CORAL 输出层：单一权重 + 递增 bias，保证 P(y>k) 单调。"""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(in_features, num_classes - 1))
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight + self.bias
        if self.num_classes > 2:
            # 累加约束：bias_k 单调递增 → logits 单调递减
            out = out[:, :1] + torch.cumsum(
                torch.cat([torch.zeros_like(out[:, :1]), out[:, 1:] - out[:, :-1]], dim=1), dim=1
            )
        return out


class ClassBalancedCE(nn.Module):
    """按有效样本数重加权，缓解长尾（`w_c ∝ 1/(1−β^{n_c})`）。"""

    def __init__(self, counts: list[int], beta: float = 0.9999) -> None:
        super().__init__()
        if not counts or min(counts) <= 0:
            raise ValueError("counts 必须为正整数列表")
        n = torch.tensor(counts, dtype=torch.float64)
        eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), n)
        w = 1.0 / eff
        w = w / w.sum() * len(counts)
        self.register_buffer("weight", w.to(torch.float32))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target, weight=self.get_buffer("weight"))


# ------------------------------------------------------------------ 检测损失
def quality_focal_loss(pred: Tensor, target: Tensor, beta: float = 2.0) -> Tensor:
    """GFL 的 Quality Focal Loss。target 为 0–1 的连续质量（此处取 IoU）。"""
    pred_sigmoid = pred.sigmoid()
    scale = pred_sigmoid - target
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (loss * scale.abs().pow(beta)).mean()


def distribution_focal_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Distribution Focal Loss：把连续距离回归为离散分布。

    pred: (N, 4·(reg_max+1))；target: (N, 4) 的距离（px）。
    """
    reg_max = pred.shape[1] // 4 - 1
    target = target.clamp(0, reg_max - 1e-4)
    tl = target.floor().long()
    tr = tl + 1
    wl = tr.to(pred.dtype) - target
    wr = 1.0 - wl
    pred = pred.view(-1, 4, reg_max + 1)
    tl = tl.view(-1, 4, 1)
    tr = tr.view(-1, 4, 1)
    wl = wl.view(-1, 4)
    wr = wr.view(-1, 4)
    loss = (
        F.cross_entropy(pred.reshape(-1, reg_max + 1), tl.reshape(-1), reduction="none").view(-1, 4)
        * wl
    )
    loss = (
        loss
        + F.cross_entropy(pred.reshape(-1, reg_max + 1), tr.reshape(-1), reduction="none").view(
            -1, 4
        )
        * wr
    )
    return loss.sum(dim=1).mean()


def giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """GIoU 损失，输入为 (N, 4) 的 xyxy。"""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    enclose = (cw * ch).clamp(min=0) + 1e-7
    giou = iou - (enclose - union) / enclose
    return (1.0 - giou).mean()
