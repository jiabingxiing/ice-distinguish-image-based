"""T2 推理：切片推理（SAHI 风格）+ 全局 NMS。

导线在 1080P 图中常仅 8–20 px 宽，直接整图下采样到 640 会把它压成 5 px。
切片推理保证小目标在原始分辨率上被看到。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from grid_ice_vision.common.constants import ASSET_CLASSES
from grid_ice_vision.common.schemas import Asset, AssetClass
from grid_ice_vision.features.detection.model import DetectionNet, decode_level

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class DetectorInferConfig:
    score_thr: float = 0.30
    nms_thr: float = 0.50
    tile: int = 640
    overlap: float = 0.20
    max_dets: int = 300
    batch: int = 4

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> DetectorInferConfig:
        if not cfg:
            return cls()
        d = cfg.get("detector", cfg)
        return cls(
            score_thr=float(d.get("score_thr", 0.30)),
            nms_thr=float(d.get("nms_thr", 0.50)),
            tile=int(d.get("tile", 640)),
            overlap=float(d.get("overlap", 0.20)),
            max_dets=int(d.get("max_dets", 300)),
            batch=int(d.get("batch", 4)),
        )


def preprocess(image_bgr: np.ndarray) -> Tensor:
    """BGR uint8 → (1,3,H,W) float32 张量。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _windows(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(int(tile * (1.0 - overlap)), 1)
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if not xs or xs[-1] + tile < w:
        xs.append(max(w - tile, 0))
    if not ys or ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    out: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            out.append((x, y, min(x + tile, w), min(y + tile, h)))
    return out


def _pad_to_multiple(t: Tensor, m: int = 32) -> tuple[Tensor, int, int]:
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate")
    return t, h, w


class DetectorInferencer:
    def __init__(
        self,
        model: DetectionNet,
        cfg: DetectorInferConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg or DetectorInferConfig()
        self.device = device
        self.reg_max = model.reg_max

    @torch.no_grad()
    def _run_tile(self, tile_bgr: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        t = preprocess(tile_bgr).to(self.device)
        t, _h, _w = _pad_to_multiple(t)
        outs = self.model(t)
        boxes: list[Tensor] = []
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        for out in outs:
            boxes_t, scores_t, labels_t = decode_level(
                out["cls"][0],
                out["reg"][0],
                out["points"],
                int(out["stride"]),
                self.reg_max,
                self.cfg.score_thr,
            )
            boxes.append(boxes_t)
            scores.append(scores_t)
            labels.append(labels_t)
        return torch.cat(boxes, 0), torch.cat(scores, 0), torch.cat(labels, 0)

    @torch.no_grad()
    def predict_boxes(self, image_bgr: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        """返回整图坐标下的 (boxes xyxy, scores, labels)。"""
        h, w = image_bgr.shape[:2]
        wins = _windows(h, w, self.cfg.tile, self.cfg.overlap)
        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_labels: list[Tensor] = []

        for i in range(0, len(wins), self.cfg.batch):
            chunk = wins[i : i + self.cfg.batch]
            for x1, y1, x2, y2 in chunk:
                tile = image_bgr[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                boxes, scores, labels = self._run_tile(tile)
                if boxes.numel() == 0:
                    continue
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        if not all_boxes:
            return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

        boxes = torch.cat(all_boxes, 0).cpu()
        scores = torch.cat(all_scores, 0).cpu()
        labels = torch.cat(all_labels, 0).cpu()

        # 裁到图像范围，去掉退化框
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 逐类 NMS
        from torchvision.ops import nms

        final_idx: list[Tensor] = []
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            keep_idx = nms(boxes[idx], scores[idx], self.cfg.nms_thr)
            final_idx.append(idx[keep_idx])
        if not final_idx:
            return boxes, scores, labels
        sel = torch.cat(final_idx)
        order = scores[sel].argsort(descending=True)[: self.cfg.max_dets]
        sel = sel[order]
        return boxes[sel], scores[sel], labels[sel]

    def predict(self, image_bgr: np.ndarray) -> list[Asset]:
        boxes, scores, labels = self.predict_boxes(image_bgr)
        assets: list[Asset] = []
        for i, (box, score, label) in enumerate(
            zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True)
        ):
            cls_name = ASSET_CLASSES[label] if 0 <= label < len(ASSET_CLASSES) else ASSET_CLASSES[0]
            assets.append(
                Asset(
                    cls=AssetClass(cls_name),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(min(max(score, 0.0), 1.0)),
                    roi_id=f"r{i}",
                )
            )
        return assets
