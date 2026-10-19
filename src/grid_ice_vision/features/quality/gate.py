"""T0 图像质量门控（§10.1）。

规则优先、零成本：先判「这张图能不能用」，再谈算法。
这一步看似琐碎，但它决定了后面所有结论的可信度 —— 镜头结冰时
模型仍然会输出一个漂亮的概率，那才是最危险的。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from grid_ice_vision.common.schemas import Exposure, QualityInfo, QualityStatus

_EPS = 1e-8


@dataclass(frozen=True)
class QualityThresholds:
    """判据阈值。默认值与 `configs/model/quality.yaml` 保持一致。"""

    blur_degraded: float = 0.60
    blur_unusable: float = 0.80
    exposure_degraded: float = 0.25
    exposure_unusable: float = 0.45
    block_energy_degraded: float = 0.15
    block_energy_unusable: float = 0.05
    night_mean_gray: float = 20.0
    grid: tuple[int, int] = (6, 8)  # (rows, cols)
    lens_window: int = 5
    lens_frames: int = 3

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> QualityThresholds:
        if not cfg:
            return cls()
        q = cfg.get("quality", cfg)
        return cls(
            blur_degraded=float(q.get("blur_degraded", 0.60)),
            blur_unusable=float(q.get("blur_unusable", 0.80)),
            exposure_degraded=float(q.get("exposure_degraded", 0.25)),
            exposure_unusable=float(q.get("exposure_unusable", 0.45)),
            block_energy_degraded=float(q.get("block_energy_degraded", 0.15)),
            block_energy_unusable=float(q.get("block_energy_unusable", 0.05)),
            night_mean_gray=float(q.get("night_mean_gray", 20.0)),
            grid=tuple(q.get("grid", (6, 8))),  # type: ignore[arg-type]
            lens_window=int(q.get("lens_window", 5)),
            lens_frames=int(q.get("lens_frames", 3)),
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的通道数: {image.shape}")


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalized_blur_score(gray: np.ndarray, ref_var: float = 1000.0) -> float:
    """拉普拉斯方差 → 0–1 的模糊分（1 = 最糊）。

    对数压缩后归一：不同分辨率下量纲差异被压平，阈值才可复用。
    """
    var = laplacian_variance(gray)
    sharp = np.log1p(max(var, 0.0)) / np.log1p(ref_var)
    return float(np.clip(1.0 - sharp, 0.0, 1.0))


def block_highfreq_energy(gray: np.ndarray, grid: tuple[int, int] = (6, 8)) -> np.ndarray:
    """网格高频能量图（0–1），用于发现局部结冰/污损/遮挡。"""
    rows, cols = grid
    h, w = gray.shape
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * h // rows : (r + 1) * h // rows, c * w // cols : (c + 1) * w // cols]
            if patch.size == 0:
                continue
            out[r, c] = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    peak = float(np.max(out))
    return out / peak if peak > _EPS else out


@dataclass
class LensIcingTracker:
    """镜头结冰/污损自检：固定区域高频能量持续走低 → 怀疑结冰。

    需要跨帧状态，因此单独成类；`features/` 内不落盘，状态由调用方持有。
    """

    frames: int = 3
    window: int = 5
    energy_floor: float = 0.05
    _history: dict[str, deque[np.ndarray]] = field(default_factory=dict)

    def update(self, camera_id: str, energy: np.ndarray) -> bool:
        hist = self._history.setdefault(camera_id, deque(maxlen=self.window))
        hist.append(np.asarray(energy, dtype=np.float32))
        if len(hist) < self.frames:
            return False
        recent = np.stack(list(hist)[-self.frames :], axis=0)
        # 每一帧都低于阈值、且位置相同（同一 block 连续为最低）→ 结冰
        persistent = (recent < self.energy_floor).all(axis=0)
        return bool(persistent.any())

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._history.clear()
        else:
            self._history.pop(camera_id, None)


def assess_quality(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
    camera_id: str | None = None,
    tracker: LensIcingTracker | None = None,
) -> QualityInfo:
    """单帧质量评估。返回的 `status` 决定下游是否允许给结论。"""
    th = thresholds or QualityThresholds()
    gray = _to_gray(image)

    blur = normalized_blur_score(gray)
    over = float((gray > 250).mean())
    under = float((gray < 15).mean())
    exposure_ratio = max(over, under)
    energy = block_highfreq_energy(gray, th.grid)
    min_block = float(energy.min())
    mean_gray = float(gray.mean())

    reasons: list[str] = []
    status = QualityStatus.USABLE

    if mean_gray < th.night_mean_gray:
        reasons.append(f"night_or_underexposed(mean={mean_gray:.1f})")
        status = QualityStatus.UNUSABLE

    if blur >= th.blur_unusable:
        reasons.append(f"severe_blur(blur={blur:.2f})")
        status = QualityStatus.UNUSABLE
    elif blur >= th.blur_degraded:
        reasons.append(f"blur(blur={blur:.2f})")
        if status is QualityStatus.USABLE:
            status = QualityStatus.DEGRADED

    if exposure_ratio >= th.exposure_unusable:
        reasons.append(f"severe_exposure(ratio={exposure_ratio:.2f})")
        status = QualityStatus.UNUSABLE
    elif exposure_ratio >= th.exposure_degraded:
        reasons.append(f"exposure(ratio={exposure_ratio:.2f})")
        if status is QualityStatus.USABLE:
            status = QualityStatus.DEGRADED

    if min_block <= th.block_energy_unusable:
        reasons.append(f"local_flat(min_block={min_block:.2f})")
        if status is QualityStatus.USABLE:
            status = QualityStatus.DEGRADED
    elif min_block < th.block_energy_degraded:
        reasons.append(f"local_low_texture(min_block={min_block:.2f})")
        if status is QualityStatus.USABLE:
            status = QualityStatus.DEGRADED

    lens = False
    if tracker is not None and camera_id:
        lens = tracker.update(camera_id, energy)
        if lens:
            reasons.append("lens_icing_suspected")
            if status is QualityStatus.USABLE:
                status = QualityStatus.DEGRADED

    exposure = Exposure.NORMAL
    if over > max(under, th.exposure_degraded):
        exposure = Exposure.OVER
    elif under > th.exposure_degraded:
        exposure = Exposure.UNDER

    # 遮挡率：极暗且平坦的网格占比（镜头被遮挡/积雪覆盖的常见表现）
    flat_dark = float(((energy < th.block_energy_degraded) & (mean_gray < 60)).mean())

    return QualityInfo(
        status=status,
        blur_score=round(blur, 4),
        exposure=exposure,
        occlusion_ratio=round(min(flat_dark, 1.0), 4),
        lens_icing_suspected=lens,
        reasons=reasons,
    )
