"""P1 传统图像特征基线：HSV/RGB 直方图 + LBP + HOG + Gabor（纯函数）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


@dataclass(frozen=True)
class ClassicalFeatureConfig:
    """传统特征超参数配置（全部由 configs/ 注入）。"""

    resize: int = 128
    hsv_bins: tuple[int, int, int] = (16, 8, 8)
    rgb_bins: tuple[int, int, int] = (8, 8, 8)
    lbp_radius: int = 2
    lbp_points: int = 16
    hog_resize: int = 64
    hog_orientations: int = 8
    hog_pixels_per_cell: tuple[int, int] = (16, 16)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    gabor_resize: int = 64
    gabor_frequencies: tuple[float, ...] = (0.1, 0.25)
    gabor_thetas: int = 4

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> ClassicalFeatureConfig:
        if not cfg:
            return cls()
        c = cfg.get("classical", cfg)
        return cls(
            resize=int(c.get("resize", 128)),
            hsv_bins=tuple(int(x) for x in c.get("hsv_bins", (16, 8, 8))),  # type: ignore[arg-type]
            rgb_bins=tuple(int(x) for x in c.get("rgb_bins", (8, 8, 8))),  # type: ignore[arg-type]
            lbp_radius=int(c.get("lbp_radius", 2)),
            lbp_points=int(c.get("lbp_points", 16)),
            hog_resize=int(c.get("hog_resize", 64)),
            hog_orientations=int(c.get("hog_orientations", 8)),
            hog_pixels_per_cell=tuple(  # type: ignore[arg-type]
                int(x) for x in c.get("hog_pixels_per_cell", (16, 16))
            ),
            hog_cells_per_block=tuple(  # type: ignore[arg-type]
                int(x) for x in c.get("hog_cells_per_block", (2, 2))
            ),
            gabor_resize=int(c.get("gabor_resize", 64)),
            gabor_frequencies=tuple(float(x) for x in c.get("gabor_frequencies", (0.1, 0.25))),
            gabor_thetas=int(c.get("gabor_thetas", 4)),
        )


def _norm_hist_1d(arr: np.ndarray, bins: int, upper: float) -> np.ndarray:
    hist = cv2.calcHist([arr], [0], None, [bins], [0.0, upper]).reshape(-1)
    s = float(hist.sum())
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


def extract_classical_features(
    image_bgr: np.ndarray,
    cfg: ClassicalFeatureConfig | None = None,
) -> np.ndarray:
    """提取拼接特征 F = [F_HSV_RGB, F_LBP, F_HOG, F_Gabor]，输出 1D float32 向量。"""
    c = cfg or ClassicalFeatureConfig()
    resized = cv2.resize(image_bgr, (c.resize, c.resize), interpolation=cv2.INTER_AREA)

    # 1. 颜色统计：HSV + RGB 通道直方图与一二阶矩
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    color_parts: list[np.ndarray] = []
    hsv_ranges = (180.0, 256.0, 256.0)
    for ch, (bins, upper) in enumerate(zip(c.hsv_bins, hsv_ranges, strict=True)):
        color_parts.append(_norm_hist_1d(hsv[:, :, ch], bins, upper))
    for ch, bins in enumerate(c.rgb_bins):
        color_parts.append(_norm_hist_1d(rgb[:, :, ch], bins, 256.0))
    hsv_f = hsv.astype(np.float32) / np.array([180.0, 255.0, 255.0], dtype=np.float32)
    rgb_f = rgb.astype(np.float32) / 255.0
    color_parts.append(hsv_f.mean(axis=(0, 1)).astype(np.float32))
    color_parts.append(hsv_f.std(axis=(0, 1)).astype(np.float32))
    color_parts.append(rgb_f.mean(axis=(0, 1)).astype(np.float32))
    color_parts.append(rgb_f.std(axis=(0, 1)).astype(np.float32))

    # 2. 纹理特征：Uniform LBP 直方图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=c.lbp_points, R=c.lbp_radius, method="uniform")
    n_lbp_bins = c.lbp_points + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_lbp_bins, range=(0, n_lbp_bins), density=False)
    lbp_vec = (lbp_hist / max(float(lbp_hist.sum()), 1.0)).astype(np.float32)

    # 3. 方向与轮廓：HOG
    gray_hog = cv2.resize(gray, (c.hog_resize, c.hog_resize), interpolation=cv2.INTER_AREA)
    hog_vec = hog(
        gray_hog,
        orientations=c.hog_orientations,
        pixels_per_cell=c.hog_pixels_per_cell,
        cells_per_block=c.hog_cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # 4. 频域多尺度方向纹理：Gabor 滤波器组响应均值/方差（实/虚部正交核 filter2D，10x+ 提速）
    gray_gab = (
        cv2.resize(gray, (c.gabor_resize, c.gabor_resize), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        / 255.0
    )
    gab_vals: list[float] = []
    for freq in c.gabor_frequencies:
        lambd = 1.0 / max(float(freq), 1e-3)
        sigma = 0.56 * lambd
        ksize = int(max(7, min(21, round(sigma * 3) * 2 + 1)))
        for t_idx in range(c.gabor_thetas):
            theta = t_idx / max(c.gabor_thetas, 1) * np.pi
            k_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0.0, cv2.CV_32F)
            k_imag = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, 0.5, np.pi / 2.0, cv2.CV_32F
            )
            r_resp = cv2.filter2D(gray_gab, cv2.CV_32F, k_real)
            i_resp = cv2.filter2D(gray_gab, cv2.CV_32F, k_imag)
            mag = cv2.magnitude(r_resp, i_resp)
            gab_vals.append(float(mag.mean()))
            gab_vals.append(float(mag.std()))
    gab_vec = np.asarray(gab_vals, dtype=np.float32)

    feat = np.concatenate([*color_parts, lbp_vec, hog_vec, gab_vec]).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def classical_feature_dim(cfg: ClassicalFeatureConfig | None = None) -> int:
    """返回指定配置下的特征向量长度。"""
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    return int(extract_classical_features(dummy, cfg).shape[0])
