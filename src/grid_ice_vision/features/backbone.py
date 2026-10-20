"""共享骨干网络。

默认走 `timm`（Apache-2.0 依赖链），失败时退化到内置 `MiniResNet`，
保证**离线、无预训练权重**时整条链路仍可构建与单测。
生产训练务必使用 `pretrained=True` + ImageNet 权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, g: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class _Residual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.SiLU(inplace=True)
        self.short = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.short(x))


class MiniResNet(nn.Module):
    """极简 ResNet 变体，仅用于离线兜底与单测（实测 2.77M 参数）。"""

    out_channels: tuple[int, ...] = (32, 64, 128, 256)
    out_strides: tuple[int, ...] = (4, 8, 16, 32)

    def __init__(self, in_chans: int = 3, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (width, width * 2, width * 4, width * 8)
        self.stem = nn.Sequential(ConvBNAct(in_chans, c1, 3, 2), ConvBNAct(c1, c1, 3, 1))
        self.s2 = nn.Sequential(_Residual(c1, c2, 2), _Residual(c2, c2, 1))
        self.s3 = nn.Sequential(_Residual(c2, c3, 2), _Residual(c3, c3, 1))
        self.s4 = nn.Sequential(_Residual(c3, c4, 2), _Residual(c4, c4, 1))
        self.out_channels = (c1, c2, c3, c4)

    def forward(self, x: Tensor) -> list[Tensor]:
        c1 = self.stem(x)  # stride 4
        c2 = self.s2(c1)  # stride 8
        c3 = self.s3(c2)  # stride 16
        c4 = self.s4(c3)  # stride 32
        return [c1, c2, c3, c4]


@dataclass
class BackboneSpec:
    name: str
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    is_fallback: bool
    pretrained: bool = False
    fallback_reason: str = ""


class Backbone(nn.Module):
    """统一包装：`forward(x) -> list[Tensor]`（低层 → 高层）。

    `pretrained=True` 时**不允许静默降级**：拿不到权重就抛错。
    理由：离线环境下 timm 会失败，若不报错，训练两小时后拿到的骨干
    与随机初始化的 MiniResNet 等价，而配置、日志、checkpoint 三处
    都写着 `pretrained: true`，事后无法分辨。
    """

    def __init__(
        self,
        name: str = "mobilenetv4_conv_small",
        pretrained: bool = False,
        allow_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.spec: BackboneSpec
        self.body: nn.Module
        reason = ""
        try:
            import timm

            body = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
            info: Any = body.feature_info
            chans = tuple(int(c) for c in info.channels())
            strides = tuple(int(s) for s in info.reduction())
            if len(chans) < 3:
                raise ValueError("timm 骨干输出层数不足")
            self.body = body
            self.spec = BackboneSpec(
                name=name, channels=chans, strides=strides, is_fallback=False, pretrained=pretrained
            )
            return
        except ImportError as exc:
            reason = f"timm 不可用: {exc}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"

        if pretrained and not allow_fallback:
            raise RuntimeError(
                f"骨干 {name} 的预训练权重加载失败（{reason}）。"
                "离线环境请先设置 HF_ENDPOINT=https://hf-mirror.com 或调用 "
                "common.config.configure_hf_mirror()；"
                "确需随机初始化请显式设置 model.pretrained=false。"
            )

        mini = MiniResNet()
        self.body = mini
        self.spec = BackboneSpec(
            name="mini_resnet",
            channels=mini.out_channels,
            strides=mini.out_strides,
            is_fallback=True,
            pretrained=False,
            fallback_reason=reason,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = self.body(x)
        return list(feats)


def build_backbone(
    name: str = "mobilenetv4_conv_small", pretrained: bool = False, allow_fallback: bool = False
) -> Backbone:
    return Backbone(name=name, pretrained=pretrained, allow_fallback=allow_fallback)
