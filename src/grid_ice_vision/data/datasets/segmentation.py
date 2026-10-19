"""分割数据集：图像 + 4 类掩码。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from grid_ice_vision.data.manifest import ManifestRecord
from grid_ice_vision.data.transforms.common import load_image, load_mask, to_chw
from grid_ice_vision.data.transforms.ice_aug import IceAugment


class SegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[ManifestRecord],
        root: str | Path,
        size: int = 512,
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.size = size
        self.aug = IceAugment(out_size=size, seed=seed) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        mask_rel = r.labels.get("mask_path")
        if not mask_rel:
            raise KeyError(f"{r.image_id} 缺少 labels.mask_path")
        img = load_image(self.root / r.path)
        mask = load_mask(self.root / str(mask_rel))

        if self.aug is not None:
            img, mask = self.aug(img, mask)
        else:
            img = cv2.resize(img, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return {
            "image": torch.from_numpy(to_chw(img)),
            "mask": torch.from_numpy(mask.astype("int64")),
            "image_id": r.image_id,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
    }
