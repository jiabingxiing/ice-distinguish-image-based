"""SCUT 覆冰类型四分类数据集（支持全图 / ROI / 双视角、安全增强与硬防泄漏隔离）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from grid_ice_vision.common import paths
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.data.manifest import ManifestRecord
from grid_ice_vision.data.transforms.common import letterbox, load_image, to_chw
from grid_ice_vision.data.transforms.weather_aug import (
    adjust_brightness_contrast,
    color_shift,
    jpeg_artifacts,
)
from grid_ice_vision.features.ice_type.roi import (
    NormalizedRoi,
    center_fallback_crop,
    crop_normalized_roi,
)

ViewMode = Literal["full", "roi", "dual"]


def load_monitor_roi_table(csv_path: str | Path) -> dict[str, NormalizedRoi]:
    """读取终端归一化 ROI 配置表 `monitor_roi_v1.csv`（不存在时返回空映射）。"""
    p = Path(csv_path)
    if not p.is_absolute():
        p = paths.PROJ / p
    if not p.exists():
        return {}
    out: dict[str, NormalizedRoi] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = str(row.get("monitor_id", "")).strip()
            if not mid or mid.startswith("#") or mid.endswith("_EXAMPLE"):
                continue
            out[mid] = NormalizedRoi(
                monitor_id=mid,
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                asset_type=str(row.get("asset_type", "unknown")).strip() or "unknown",
                view_version=str(row.get("view_version", "v1")).strip() or "v1",
            )
    return out


class IceTypeAugment:
    """覆冰类型安全图像增强（仅 train 启用）。

    严格遵守物理约束：
    - 允许：亮度/对比度、轻微色彩偏移、Gamma、JPEG 压缩(QF 45–85)、轻微平移缩放(±5%)、
      小角度旋转(±5°)、可选水平翻转（默认关闭）；
    - 严禁：垂直翻转、强色彩畸变、MixUp/CutMix。
    """

    def __init__(
        self,
        *,
        brightness: float = 0.15,
        contrast: float = 0.15,
        gamma_range: tuple[float, float] = (0.85, 1.15),
        jpeg_p: float = 0.30,
        shift_scale_p: float = 0.40,
        rotate_deg: float = 5.0,
        allow_hflip: bool = False,
        seed: int = 42,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.gamma_range = gamma_range
        self.jpeg_p = jpeg_p
        self.shift_scale_p = shift_scale_p
        self.rotate_deg = rotate_deg
        self.allow_hflip = allow_hflip
        self.rng = np.random.default_rng(seed)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        img = image_bgr
        if self.rng.random() < 0.7:
            alpha = 1.0 + float(self.rng.uniform(-self.contrast, self.contrast))
            beta = float(self.rng.uniform(-self.brightness, self.brightness)) * 255.0
            img = adjust_brightness_contrast(img, alpha, beta)
        if self.rng.random() < 0.3:
            gamma = float(self.rng.uniform(*self.gamma_range))
            lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lut.astype(np.uint8))
        if self.rng.random() < 0.25:
            img = color_shift(
                img,
                float(self.rng.uniform(-6.0, 6.0)),
                float(self.rng.uniform(-6.0, 6.0)),
                float(self.rng.uniform(-6.0, 6.0)),
            )
        if self.rng.random() < self.shift_scale_p:
            h, w = img.shape[:2]
            angle = float(self.rng.uniform(-self.rotate_deg, self.rotate_deg))
            scale = float(self.rng.uniform(0.95, 1.05))
            tx = float(self.rng.uniform(-0.05, 0.05)) * w
            ty = float(self.rng.uniform(-0.05, 0.05)) * h
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if self.allow_hflip and self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        if self.rng.random() < self.jpeg_p:
            img = jpeg_artifacts(img, int(self.rng.integers(45, 86)))
        return img


class IceTypeDataset(Dataset[dict[str, Any]]):
    """SCUT 四分类数据集。

    硬防泄漏：`monitor_id`/`bureau`/`year` 仅随元数据返回用于评估切片，禁止送入模型计算。
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        root: str | Path | None = None,
        *,
        image_size: int = 224,
        roi_size: int = 224,
        pad_value: int = 114,
        view_mode: ViewMode = "full",
        roi_map: dict[str, NormalizedRoi] | None = None,
        train: bool = False,
        allow_hflip: bool = False,
        exclude_noise_candidates: bool = True,
        return_bgr: bool = False,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) if root is not None else paths.scut_root()
        filtered: list[ManifestRecord] = []
        for r in records:
            if exclude_noise_candidates and bool(r.labels.get("noise_candidate", False)):
                continue
            ice_t = str(r.labels.get("ice_type", "unknown"))
            if ice_t not in ICE_TYPE_FOUR_CLASSES:
                continue
            filtered.append(r)
        self.records = filtered
        self.image_size = image_size
        self.roi_size = roi_size
        self.pad_value = pad_value
        self.view_mode = view_mode
        self.roi_map = roi_map or {}
        self.return_bgr = return_bgr
        self.aug = IceTypeAugment(allow_hflip=allow_hflip, seed=seed) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.root / p

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        img = load_image(self._resolve_path(r.path))
        h, w = img.shape[:2]

        roi_obj = self.roi_map.get(r.camera_id)
        roi_available = roi_obj is not None

        if self.view_mode in ("roi", "dual"):
            roi_raw = (
                crop_normalized_roi(img, roi_obj)
                if roi_obj is not None
                else center_fallback_crop(img)
            )
            if self.aug is not None:
                roi_raw = self.aug(roi_raw)
            roi_lb = letterbox(roi_raw, self.roi_size, pad_value=self.pad_value)
        else:
            roi_lb = None

        full_raw = self.aug(img) if self.aug is not None else img
        full_lb = letterbox(full_raw, self.image_size, pad_value=self.pad_value)

        label_str = str(r.labels["ice_type"])
        label = ICE_TYPE_FOUR_CLASSES.index(label_str)

        item: dict[str, Any] = {
            "image": torch.from_numpy(to_chw(full_lb)),
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": r.image_id,
            "monitor_id": r.camera_id,
            "bureau": str(r.labels.get("bureau", r.site_id)),
            "year": str(r.labels.get("ice_season", r.weather_event_id)),
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "roi_available": roi_available,
        }
        if roi_lb is not None:
            item["roi_image"] = torch.from_numpy(to_chw(roi_lb))
        if self.return_bgr:
            item["image_bgr"] = full_lb
            if roi_lb is not None:
                item["roi_bgr"] = roi_lb
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "monitor_id": [b["monitor_id"] for b in batch],
        "bureau": [b["bureau"] for b in batch],
        "year": [b["year"] for b in batch],
        "resolution": [b["resolution"] for b in batch],
        "width": [b["width"] for b in batch],
        "height": [b["height"] for b in batch],
        "roi_available": [b["roi_available"] for b in batch],
    }
    if "roi_image" in batch[0]:
        out["roi_image"] = torch.stack([b["roi_image"] for b in batch])
    if "image_bgr" in batch[0]:
        out["image_bgr"] = [b["image_bgr"] for b in batch]
    if "roi_bgr" in batch[0]:
        out["roi_bgr"] = [b["roi_bgr"] for b in batch]
    return out
