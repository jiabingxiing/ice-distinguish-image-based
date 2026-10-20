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

        # 类别掩码：数据源缺某一类（如 WEAPD 没有 clear）时屏蔽该维，避免
        # 模型被"永远为 0"的标签误导成"clear 一定不成立"
        raw_mask = r.labels.get("weather_mask")
        mask = (
            np.asarray(raw_mask, dtype=np.float32)
            if raw_mask is not None
            else np.ones(len(WEATHER_LABELS), dtype=np.float32)
        )

        intensity = str(r.labels.get("intensity", "unknown"))
        level = INTENSITY_LEVELS.index(intensity) if intensity in INTENSITY_LEVELS else -1

        item: dict[str, Any] = {
            "image": torch.from_numpy(to_chw(img)),
            "weather": torch.from_numpy(target),
            "weather_mask": torch.from_numpy(mask),
            "intensity": torch.tensor(level, dtype=torch.long),
            "image_id": r.image_id,
        }
        # 现场气象读数（可选）：没有就完全不传，模型侧按 avail=0 处理
        ws = r.labels.get("weather_sensor")
        if isinstance(ws, dict):
            item["met"] = torch.tensor(
                [
                    [
                        float(ws.get("t_amb_c", 0.0)),
                        float(ws.get("rh", 0.0)),
                        float(ws.get("v_wind_ms", 0.0)),
                        float(ws.get("p_atm_hpa", 1013.0)),
                    ]
                ],
                dtype=torch.float32,
            )
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "weather": torch.stack([b["weather"] for b in batch]),
        "weather_mask": torch.stack([b["weather_mask"] for b in batch]),
        "intensity": torch.stack([b["intensity"] for b in batch]),
    }
    if all("met" in b for b in batch):
        out["met"] = torch.cat([b["met"] for b in batch], dim=0)
    out["image_id"] = [b["image_id"] for b in batch]
    return out
