"""MobileNetV3-Small 576 维特征提取与末端选择性解冻接口（Plan §3.5）。"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grid_ice_vision.features.backbone import BackboneSpec, ConvBNAct


class _FallbackConv576(nn.Module):
    """离线无权重且不允许网络下载时的最小 576 通道单测兜底骨干。"""

    def __init__(self) -> None:
        super().__init__()
        self.s1 = ConvBNAct(3, 32, 3, 2)
        self.s2 = ConvBNAct(32, 96, 3, 2)
        self.s3 = ConvBNAct(96, 256, 3, 2)
        self.s4 = ConvBNAct(256, 576, 3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.s4(self.s3(self.s2(self.s1(x))))


def _normalize_timm_name(name: str) -> str:
    aliases = {
        "mobilenet_v3_small": "mobilenetv3_small_100",
        "mobilenetv3_small": "mobilenetv3_small_100",
    }
    return aliases.get(name, name)


class IceTypeBackbone(nn.Module):
    """覆冰类型识别轻量视觉骨干。

    - 支持 `timm`（默认 `mobilenetv3_small_100`，`features_only=True` 末层精确 576 维）
      与 `torchvision`（`mobilenet_v3_small.features` 576 维）；
    - 默认 `forward(x)` 输出 `(B, 576)`；传入 `roi` 时共享权重输出双视角拼接 `(B, 1152)`；
    - 严禁在本模块内进行磁盘文件读写或环境变量修改。
    """

    def __init__(
        self,
        name: str = "mobilenetv3_small_100",
        *,
        provider: Literal["timm", "torchvision"] = "timm",
        pretrained: bool = False,
        allow_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.spec: BackboneSpec
        self.body: nn.Module
        self.feature_dim: int = 576

        reason = ""
        if provider == "timm":
            timm_name = _normalize_timm_name(name)
            try:
                import timm

                body = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    features_only=True,
                )
                feat_info: Any = body.feature_info
                chans = tuple(int(c) for c in feat_info.channels())
                strides = tuple(int(s) for s in feat_info.reduction())
                self.body = body
                self.feature_dim = int(chans[-1])
                self.spec = BackboneSpec(
                    name=timm_name,
                    channels=chans,
                    strides=strides,
                    is_fallback=False,
                    pretrained=pretrained,
                )
                return
            except Exception as exc:
                reason = f"timm {type(exc).__name__}: {str(exc)[:200]}"
        elif provider == "torchvision":
            try:
                import torchvision.models as tvm

                weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
                tv_model = tvm.mobilenet_v3_small(weights=weights)
                self.body = tv_model.features
                self.feature_dim = 576
                self.spec = BackboneSpec(
                    name="torchvision_mobilenet_v3_small",
                    channels=(16, 16, 24, 48, 576),
                    strides=(2, 4, 8, 16, 32),
                    is_fallback=False,
                    pretrained=pretrained,
                )
                return
            except Exception as exc:
                reason = f"torchvision {type(exc).__name__}: {str(exc)[:200]}"
        else:
            raise ValueError(f"不支持的 provider: {provider!r}")

        if pretrained and not allow_fallback:
            raise RuntimeError(
                f"覆冰类型骨干 {name} ({provider}) 预训练权重加载失败（{reason}）。"
                "请先调用 common.config.configure_hf_mirror() 或检查本地权重缓存。"
            )

        self.body = _FallbackConv576()
        self.feature_dim = 576
        self.spec = BackboneSpec(
            name="fallback_conv576",
            channels=(32, 96, 256, 576),
            strides=(2, 4, 8, 16),
            is_fallback=True,
            pretrained=False,
            fallback_reason=reason,
        )

    def extract_map(self, x: Tensor) -> Tensor:
        if self.spec.is_fallback or self.provider == "torchvision":
            return self.body(x)
        feats = self.body(x)
        return feats[-1]

    def forward_single(self, x: Tensor) -> Tensor:
        fmap = self.extract_map(x)
        return F.adaptive_avg_pool2d(fmap, 1).flatten(1)

    def forward_dual(self, x_full: Tensor, x_roi: Tensor) -> Tensor:
        return torch.cat([self.forward_single(x_full), self.forward_single(x_roi)], dim=1)

    def forward(self, x: Tensor, roi: Tensor | None = None) -> Tensor:
        if roi is None:
            return self.forward_single(x)
        return self.forward_dual(x, roi)

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def _stage_modules(self) -> list[nn.Module]:
        if self.spec.is_fallback:
            fb = self.body
            assert isinstance(fb, _FallbackConv576)
            return [fb.s1, fb.s2, fb.s3, fb.s4]
        if self.provider == "timm" and hasattr(self.body, "blocks"):
            blocks = self.body.blocks
            if isinstance(blocks, nn.Sequential):
                return list(blocks.children())
        if self.provider == "torchvision" and isinstance(self.body, nn.Sequential):
            children = list(self.body.children())
            # torchvision features 0..12，按末端通道语义聚合最后若干阶段
            return [
                nn.Sequential(*children[:4]),
                nn.Sequential(*children[4:9]),
                nn.Sequential(*children[9:12]),
                children[12],
            ]
        return [self.body]

    def unfreeze_last_stages(self, n_stages: int = 1) -> int:
        """冻结前置全部层，仅解冻最后 n_stages 个特征阶段，返回可训练参数量。"""
        self.freeze_all()
        if n_stages <= 0:
            return 0
        stages = self._stage_modules()
        k = min(n_stages, len(stages))
        for stage in stages[-k:]:
            for p in stage.parameters():
                p.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def stage_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        head: nn.Module | None = None,
    ) -> list[dict[str, Any]]:
        """构造分层学习率优化器参数组（主干小学习率 + 分类头大学习率）。"""
        groups: list[dict[str, Any]] = []
        bb_params = [p for p in self.parameters() if p.requires_grad]
        if bb_params:
            groups.append({"params": bb_params, "lr": float(backbone_lr)})
        if head is not None:
            hd_params = [p for p in head.parameters() if p.requires_grad]
            if hd_params:
                groups.append({"params": hd_params, "lr": float(head_lr)})
        return groups
