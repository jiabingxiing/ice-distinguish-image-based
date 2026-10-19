"""场景分层采样器（§10.4.3、§11.5）。

Ice-Seg 的三类失败模式对应三组样本：晴天（FP 多）、恶劣天气（FN 多）、
设备背景（杆塔几何混淆）。按组加权采样，比全局过采样更精准。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from torch.utils.data import Sampler

DEFAULT_SCENE_WEIGHTS: dict[str, float] = {
    "clear": 1.0,
    "adverse": 1.5,
    "equipment": 1.3,
    "normal": 1.0,
}


@dataclass
class SceneStratifiedSampler(Sampler[int]):
    """按 `scene_group` 加权的有放回采样。

    `num_samples` 缺省等于数据集大小，保证每个 epoch 的步数与普通训练一致。
    """

    scene_groups: list[str]
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCENE_WEIGHTS))
    num_samples: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self._weights = np.array(
            [self.weights.get(g, 1.0) for g in self.scene_groups], dtype=np.float64
        )
        self._weights = self._weights / self._weights.sum()
        self._n = self.num_samples or len(self.scene_groups)
        self._rng = np.random.default_rng(self.seed)

    def __iter__(self) -> Iterator[int]:
        idx = self._rng.choice(len(self.scene_groups), size=self._n, replace=True, p=self._weights)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self._n


def scene_histogram(scene_groups: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in scene_groups:
        out[g] = out.get(g, 0) + 1
    return out
