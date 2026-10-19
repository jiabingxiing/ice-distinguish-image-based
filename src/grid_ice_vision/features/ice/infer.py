"""T3 推理：ROI 裁切 → 分割 → 覆盖率 / 桥接率 / 掩码 RLE。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from grid_ice_vision.common.constants import SEG_CLASSES
from grid_ice_vision.common.schemas import MaskRLE
from grid_ice_vision.features.ice.model import UNetLite
from grid_ice_vision.features.severity.geometry import bridge_ratio, coverage_ratio

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SegInferConfig:
    input_size: int = 512
    ice_conf_thr: float = 0.5
    min_coverage: float = 0.03

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> SegInferConfig:
        if not cfg:
            return cls()
        s = cfg.get("seg", cfg)
        return cls(
            input_size=int(s.get("input_size", 512)),
            ice_conf_thr=float(s.get("ice_conf_thr", 0.5)),
            min_coverage=float(s.get("min_coverage", 0.03)),
        )


def encode_rle(mask: np.ndarray) -> MaskRLE:
    """COCO 风格列优先 RLE（0/1 交替计数的字符串表示）。"""
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    counts: list[int] = []
    if flat.size == 0:
        return MaskRLE(size=(int(mask.shape[0]), int(mask.shape[1])), counts="")
    cur = 0
    run = 0
    for v in flat:
        if v != cur:
            counts.append(run)
            run = 1
            cur = int(v)
        else:
            run += 1
    counts.append(run)
    return MaskRLE(
        size=(int(mask.shape[0]), int(mask.shape[1])), counts=",".join(str(c) for c in counts)
    )


def decode_rle(rle: MaskRLE) -> np.ndarray:
    h, w = rle.size
    counts = [int(c) for c in rle.counts.split(",") if c != ""] if rle.counts else []
    out = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        if val:
            out[pos : pos + c] = 1
        pos += c
        val = 1 - val
    return out.reshape((h, w), order="F")


@dataclass
class SegPrediction:
    mask: np.ndarray  # (H, W) uint8，0..3
    probs: np.ndarray  # (C, H, W) float32，ROI 尺寸
    coverage: float | None
    ice_present: bool
    confidence: float
    bridge: float | None
    roi_bbox: tuple[int, int, int, int]

    def mask_rle(self) -> MaskRLE:
        return encode_rle(self.mask)


class IceSegInferencer:
    def __init__(
        self, model: UNetLite, cfg: SegInferConfig | None = None, device: str = "cpu"
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg or SegInferConfig()
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image_bgr: np.ndarray,
        roi_bbox: tuple[float, float, float, float] | None = None,
        is_insulator: bool = False,
    ) -> SegPrediction:
        h, w = image_bgr.shape[:2]
        if roi_bbox is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = (round(v) for v in roi_bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
        crop = image_bgr[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = F.interpolate(
            t, size=(self.cfg.input_size, self.cfg.input_size), mode="bilinear", align_corners=False
        )

        logits = self.model(t)
        probs = logits.softmax(dim=1)
        probs_full = F.interpolate(probs, size=crop.shape[:2], mode="bilinear", align_corners=False)
        mask = probs_full.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        prob_np = probs_full[0].cpu().numpy().astype(np.float32)

        ice = ((mask == 2) | (mask == 3)).sum()
        exposed = (mask == 1).sum()
        cov: float | None
        try:
            cov = coverage_ratio(int(ice), int(exposed))
        except ValueError:
            cov = None

        # 置信度：覆冰像素上的平均冰类概率（无冰时取背景置信度）
        if ice > 0:
            conf = float(prob_np[2:4][:, mask >= 2].mean()) if ice > 0 else 0.0
        else:
            conf = float(prob_np[0].mean())

        bridge: float | None = None
        if is_insulator:
            gap = int((mask == 1).sum() + ice)  # 以设备像素近似伞裙间隙
            bridge = bridge_ratio(int((mask == 3).sum()), gap)

        return SegPrediction(
            mask=mask,
            probs=prob_np,
            coverage=cov,
            ice_present=bool(cov is not None and cov >= self.cfg.min_coverage),
            confidence=round(min(max(conf, 0.0), 1.0), 4),
            bridge=bridge,
            roi_bbox=(x1, y1, x2, y2),
        )


def mask_class_names() -> tuple[str, ...]:
    return SEG_CLASSES
