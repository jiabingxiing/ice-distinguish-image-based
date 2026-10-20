"""T1 降水与天气分类网络（§10.2）。

结构：backbone → [FiLM 气象调制] → 8 类多标签 sigmoid + 强度 CORAL。
多标签而非 softmax：现场常见「冻雨 + 浓雾」「雨夹雪」。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from grid_ice_vision.common.constants import INTENSITY_LEVELS, WEATHER_LABELS
from grid_ice_vision.features.backbone import Backbone, build_backbone
from grid_ice_vision.training.losses import CoralHead


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：用气象向量调制图像特征。"""

    def __init__(self, met_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(met_dim, feat_dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, f_img: Tensor, f_met: Tensor) -> Tensor:
        gb = self.to_gamma_beta(f_met)
        gamma, beta = gb.chunk(2, dim=1)
        return f_img * (1.0 + gamma) + beta


class MetEncoder(nn.Module):
    """气象向量编码：4 个物理量 + 1 个可用性标志。

    缺失时不做任何插补（不猜），只用 flag=0 告知网络「没有气象输入」。
    """

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.register_buffer("_mean", torch.tensor([0.0, 60.0, 5.0, 1013.0]))
        self.register_buffer("_std", torch.tensor([15.0, 25.0, 5.0, 10.0]))

    def forward(self, met: Tensor | None) -> Tensor:
        """`met=None` 表示「没有气象输入」，此时**不插补任何物理量**。

        早前版本把缺失输入伪造成 (0℃, 60%RH, 5m/s) 且可用性标志置 1，
        等于告诉网络"有传感器且读数是这些值"。上线接真传感器
        （如 -5℃ / 95%RH）时输入分布整体跳变，FiLM 的调制系数会落在
        训练时没见过的区间。
        """
        if met is None:
            # 4 个物理量 + avail 全 0；Python 级分支，导出时是常量，不破坏 ONNX
            return self.net(torch.zeros(1, 5, device=next(self.parameters()).device))
        if met.ndim == 1:
            met = met.unsqueeze(0)
        avail = torch.ones(met.shape[0], 1, device=met.device, dtype=met.dtype)
        mean = self.get_buffer("_mean")
        std = self.get_buffer("_std")
        norm = (met - mean) / std
        return self.net(torch.cat([norm, avail], dim=1))


class PrecipitationNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = False,
        num_weather: int = len(WEATHER_LABELS),
        num_intensity: int = len(INTENSITY_LEVELS),
        met_dim: int = 32,
        use_film: bool = True,
        met_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained)
        feat_dim = self.backbone.spec.channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.use_film = use_film
        self.met_dropout = met_dropout
        self.met_encoder = MetEncoder(met_dim)
        self.film = FiLM(met_dim, feat_dim) if use_film else None
        self.drop = nn.Dropout(0.2)
        self.weather_head = nn.Linear(feat_dim, num_weather)
        self.intensity_head = CoralHead(feat_dim, num_intensity)

    def forward(self, x: Tensor, met: Tensor | None = None) -> dict[str, Tensor]:
        feats = self.backbone(x)
        f = self.pool(feats[-1]).flatten(1)

        if self.use_film and self.film is not None:
            if self.training and met is not None and torch.rand(1).item() < self.met_dropout:
                met = None
            f_met = self.met_encoder(met)
            # 无气象输入时编码器只输出 1 行，广播到整批（expand 对同尺寸是无操作）
            f_met = f_met.expand(f.shape[0], -1)
            f = self.film(f, f_met)

        f = self.drop(f)
        return {
            "weather_logits": self.weather_head(f),
            "intensity_logits": self.intensity_head(f),
            "feat": f,
        }
