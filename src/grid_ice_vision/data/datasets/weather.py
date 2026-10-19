"""天气分类数据集：manifest → (图像, 8 维多标签, 强度等级)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from grid_ice_vision.common.constants import INTENSITY_LEVELS, WEATHER_LABELS
from grid_ice_vision.data.manifest import ManifestRecord
from grid_ice_vision.data.transforms.common import letterbox, load_image, to_chw
from grid_ice_vision.data.transforms.weather_aug import WeatherAugment


class WeatherDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[ManifestRecord],
        root: str | Path,
        size: int = 384,
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.size = size
        self.aug = WeatherAugment(seed=seed) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        img = load_image(self.root / r.path)
        if self.aug is not None:
            img = self.aug(img)
        img = letterbox(img, self.size)

        labels = set(r.labels.get("weather_visual", []) or [])
        target = np.zeros(len(WEATHER_LABELS), dtype=np.float32)
        for i, name in enumerate(WEATHER_LABELS):
            if name in labels:
                target[i] = 1.0

        intensity = str(r.labels.get("intensity", "unknown"))
        level = INTENSITY_LEVELS.index(intensity) if intensity in INTENSITY_LEVELS else -1

        return {
            "image": torch.from_numpy(to_chw(img)),
            "weather": torch.from_numpy(target),
            "intensity": torch.tensor(level, dtype=torch.long),
            "image_id": r.image_id,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "weather": torch.stack([b["weather"] for b in batch]),
        "intensity": torch.stack([b["intensity"] for b in batch]),
    }
    out["image_id"] = [b["image_id"] for b in batch]
    return out
