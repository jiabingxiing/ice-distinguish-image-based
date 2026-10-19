"""天气增强（算子 A 的轻量版）：亮度/对比度/色偏/JPEG 噪声。

对应 §11.5 的 A 组（常规）与 B 组（天气鲁棒）中不改变标签语义的部分。
重度雾/雨/雪合成见 `scripts/synthesize_weather.py`。
"""

from __future__ import annotations

import cv2
import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def color_shift(image: np.ndarray, db: float, dg: float, dr: float) -> np.ndarray:
    out: np.ndarray = image.astype(np.int16)
    out[..., 0] = np.clip(out[..., 0] + db, 0, 255)
    out[..., 1] = np.clip(out[..., 1] + dg, 0, 255)
    out[..., 2] = np.clip(out[..., 2] + dr, 0, 255)
    return out.astype(np.uint8)


def jpeg_artifacts(image: np.ndarray, quality: int = 45) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return np.asarray(cv2.imdecode(buf, cv2.IMREAD_COLOR), dtype=np.uint8)


def motion_blur(image: np.ndarray, ksize: int = 7, angle_deg: float = 0.0) -> np.ndarray:
    kernel: np.ndarray = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    kernel = np.asarray(cv2.warpAffine(kernel, rot, (ksize, ksize)), dtype=np.float32)
    kernel /= max(float(kernel.sum()), 1e-6)
    return np.asarray(cv2.filter2D(image, -1, kernel), dtype=np.uint8)


class WeatherAugment:
    """随机组合的常规增强。**不改变**任何标签语义。"""

    def __init__(
        self,
        brightness: float = 0.25,
        contrast: float = 0.25,
        jpeg_p: float = 0.3,
        blur_p: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.jpeg_p = jpeg_p
        self.blur_p = blur_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image
        if self.rng.random() < 0.8:
            alpha = 1.0 + float(self.rng.uniform(-self.contrast, self.contrast))
            beta = float(self.rng.uniform(-self.brightness, self.brightness)) * 255.0
            img = adjust_brightness_contrast(img, alpha, beta)
        if self.rng.random() < 0.3:
            img = color_shift(
                img,
                float(self.rng.uniform(-10, 10)),
                float(self.rng.uniform(-10, 10)),
                float(self.rng.uniform(-10, 10)),
            )
        if self.rng.random() < self.blur_p:
            img = motion_blur(
                img,
                ksize=int(self.rng.choice([3, 5, 7])),
                angle_deg=float(self.rng.uniform(0, 180)),
            )
        if self.rng.random() < self.jpeg_p:
            img = jpeg_artifacts(img, int(self.rng.integers(35, 90)))
        return img
