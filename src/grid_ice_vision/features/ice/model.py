"""T3 覆冰分割 + T4 类型/程度头（§10.4、§10.5）。

分割默认 UNet-lite 而非 LR-ASPP：覆冰边界就是厚度精度的上限，
ASPP 的全局池化会把边界糊掉。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grid_ice_vision.common.constants import ICE_TYPES, VISUAL_GRADES
from grid_ice_vision.features.backbone import Backbone, ConvBNAct, build_backbone
from grid_ice_vision.training.losses import CoralHead


class _UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(cin + cskip, cout, 3)
        self.conv2 = ConvBNAct(cout, cout, 3)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class UNetLite(nn.Module):
    """轻量 UNet：timm 编码器 + 三级跳连解码器，输出 stride=4 后上采样。"""

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = False,
        num_classes: int = 4,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained)
        chans = list(self.backbone.spec.channels)
        c1, c2, c3, c4 = chans[-4:]
        d = decoder_channels
        self.up3 = _UpBlock(c4, c3, d)
        self.up2 = _UpBlock(d, c2, d)
        self.up1 = _UpBlock(d, c1, d)
        self.head = nn.Sequential(ConvBNAct(d, d, 3), nn.Conv2d(d, num_classes, 1))
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        f1, f2, f3, f4 = self.backbone(x)[-4:]
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        logits = self.head(d1)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


class IceTypeGradeNet(nn.Module):
    """T4：ROI 图像 + 4 类掩码 → 覆冰类型（5 类）+ 视觉等级（G0–G4, CORAL）。"""

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = False,
        num_types: int = len(ICE_TYPES),
        num_grades: int = len(VISUAL_GRADES),
        mask_classes: int = 4,
    ) -> None:
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained)
        feat_dim = self.backbone.spec.channels[-1]
        self.mask_encoder = nn.Sequential(
            ConvBNAct(mask_classes, 32, 3, 2),
            ConvBNAct(32, 64, 3, 2),
            ConvBNAct(64, 64, 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + 64, 256), nn.SiLU(inplace=True), nn.Dropout(0.2)
        )
        self.type_head = nn.Linear(256, num_types)
        self.grade_head = CoralHead(256, num_grades)
        self.num_types = num_types
        self.num_grades = num_grades

    def forward(self, roi: Tensor, mask_onehot: Tensor) -> dict[str, Tensor]:
        f_img = F.adaptive_avg_pool2d(self.backbone(roi)[-1], 1).flatten(1)
        f_mask = self.mask_encoder(mask_onehot).flatten(1)
        f = self.fuse(torch.cat([f_img, f_mask], dim=1))
        return {"type_logits": self.type_head(f), "grade_logits": self.grade_head(f)}
