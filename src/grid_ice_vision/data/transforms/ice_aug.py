"""覆冰外观增强（C 组，仅 train）。

红线：任何改变「是否存在覆冰」语义的增强（例如给晴天图强加冰层却保留
`ice_type=none`）必须同步更新标签，否则禁止使用。
"""

from __future__ import annotations

import cv2
import numpy as np

from grid_ice_vision.data.transforms.common import random_hflip, random_scale_crop


class IceAugment:
    """图像 + 掩码同步增强。"""

    def __init__(
        self,
        out_size: int = 512,
        scale: tuple[float, float] = (0.8, 1.3),
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.out_size = out_size
        self.scale = scale
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img, m = random_scale_crop(image, mask, self.scale, self.out_size, self.rng)
        assert m is not None
        img, m = random_hflip(img, m, p=self.hflip_p, rng=self.rng)
        if self.rng.random() < self.vflip_p:
            img = np.ascontiguousarray(img[::-1])
            m = np.ascontiguousarray(m[::-1])
        if self.rng.random() < 0.5:
            k = int(self.rng.integers(0, 4))
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                m = np.ascontiguousarray(np.rot90(m, k))
        return img, m


def dilate_ice_mask(mask: np.ndarray, ice_class: int = 2, ksize: int = 3) -> np.ndarray:
    """把冰类边界扩一圈，用于 boundary loss 的容差带可视化（不用于训练标签）。"""
    m = (mask == ice_class).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(m, k)
