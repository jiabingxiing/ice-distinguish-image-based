"""通用图像变换：只用 numpy + OpenCV，不引入重型增强框架。

理由：增强必须能被单测、能离线跑、且不能改变标签语义。
需要更强增强时再装 `.[aug]`（albumentations），但核心链路不依赖它。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str | Path) -> np.ndarray:
    """读图为 BGR uint8。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def load_mask(path: str | Path) -> np.ndarray:
    """读掩码为 uint8 类别索引（单通道，最近邻）。"""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"无法读取掩码: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def resize_keep_ratio(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """等比缩放到 size×size 并居中填充。"""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def to_chw(image_bgr: np.ndarray, normalize: bool = True) -> np.ndarray:
    """BGR uint8 → (3, H, W) float32。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if normalize:
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def random_hflip(
    image: np.ndarray, *arrays: np.ndarray, p: float = 0.5, rng: np.random.Generator
) -> tuple[np.ndarray, ...]:
    if rng.random() < p:
        return (
            np.ascontiguousarray(image[:, ::-1]),
            *(np.ascontiguousarray(a[:, ::-1]) for a in arrays),
        )
    return (image, *arrays)


def random_scale_crop(
    image: np.ndarray,
    mask: np.ndarray | None,
    scale: tuple[float, float],
    out_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """随机缩放后中心裁切到固定尺寸（同步作用于掩码）。"""
    s = float(rng.uniform(*scale))
    h, w = image.shape[:2]
    nh, nw = max(int(h * s), 1), max(int(w * s), 1)
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    m = None if mask is None else cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    if nh < out_size or nw < out_size:
        pad_h = max(out_size - nh, 0)
        pad_w = max(out_size - nw, 0)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if m is not None:
            m = cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        nh, nw = img.shape[:2]

    top = int(rng.integers(0, nh - out_size + 1))
    left = int(rng.integers(0, nw - out_size + 1))
    img = img[top : top + out_size, left : left + out_size]
    if m is not None:
        m = m[top : top + out_size, left : left + out_size]
    return img, m
