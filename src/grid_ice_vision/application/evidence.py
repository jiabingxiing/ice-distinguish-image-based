"""证据图（§16.6）：卡尺叠加图 / 掩码叠加 / 概率热图。

卡尺图是最有说服力的一层 —— 它把「算法凭什么说 7.1 mm」摊在老师傅面前：
覆冰上下包络、裸导线基准线、每个采样截面的位置与读数，全在图上。
不做 LRP：实现复杂、对自定义网络支持差、运维人员看不懂。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from grid_ice_vision.features.severity.geometry import Section

_ICE_COLOR = (60, 60, 220)  # BGR 红
_EXPOSED_COLOR = (200, 140, 60)  # BGR 蓝
_ICICLE_COLOR = (40, 160, 240)  # BGR 橙
_BASE_COLOR = (80, 200, 80)  # BGR 绿
_SECTION_COLOR = (240, 240, 240)


def draw_caliper(
    image_bgr: np.ndarray,
    sections: Sequence[Section],
    w0_px: float,
    out_path: str | Path,
    d_radial_mm: float | None = None,
    alpha_mm_per_px: float | None = None,
    grade: str | None = None,
    max_sections: int = 16,
) -> str:
    """绘制测量卡尺图并落盘，返回路径。"""
    canvas = image_bgr.copy()
    valid = [s for s in sections if s.valid]
    if not valid:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)

    step = max(len(valid) // max_sections, 1)
    picked = valid[::step]

    upper: list[tuple[int, int]] = []
    lower: list[tuple[int, int]] = []
    base_up: list[tuple[int, int]] = []
    base_lo: list[tuple[int, int]] = []
    for s in picked:
        cx, cy = s.center
        nx, ny = s.normal
        p_up = (round(cx + s.t_up * nx), round(cy + s.t_up * ny))
        p_lo = (round(cx + s.t_down * nx), round(cy + s.t_down * ny))
        upper.append(p_up)
        lower.append(p_lo)
        base_up.append((round(cx - w0_px / 2 * nx), round(cy - w0_px / 2 * ny)))
        base_lo.append((round(cx + w0_px / 2 * nx), round(cy + w0_px / 2 * ny)))

    cv2.polylines(canvas, [np.array(upper, np.int32)], False, _ICE_COLOR, 2)
    cv2.polylines(canvas, [np.array(lower, np.int32)], False, _ICE_COLOR, 2)
    cv2.polylines(canvas, [np.array(base_up, np.int32)], False, _BASE_COLOR, 1)
    cv2.polylines(canvas, [np.array(base_lo, np.int32)], False, _BASE_COLOR, 1)

    for p_up, p_lo in zip(upper, lower, strict=True):
        cv2.line(canvas, p_up, p_lo, _SECTION_COLOR, 1)
        cv2.circle(canvas, p_up, 2, _ICE_COLOR, -1)
        cv2.circle(canvas, p_lo, 2, _ICE_COLOR, -1)

    lines = []
    if d_radial_mm is not None:
        lines.append(f"d_radial = {d_radial_mm:.1f} mm")
    if alpha_mm_per_px is not None:
        lines.append(f"alpha = {alpha_mm_per_px:.4f} mm/px")
    lines.append(f"W0 = {w0_px:.1f} px   sections = {len(valid)}")
    if grade:
        lines.append(f"grade = {grade}")

    # 文字底衬：深色半透明面板，保证在雪地/天空等亮背景下也能读清
    line_h = 24
    panel_w = max(len(t) for t in lines) * 13 + 24
    panel_h = line_h * len(lines) + 12
    panel = canvas.copy()
    cv2.rectangle(panel, (4, 4), (min(panel_w, canvas.shape[1] - 4), panel_h), (20, 20, 20), -1)
    canvas = cv2.addWeighted(panel, 0.55, canvas, 0.45, 0)

    y = 26
    for text in lines:
        cv2.putText(
            canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += line_h

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def draw_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    out_path: str | Path,
    alpha: float = 0.45,
) -> str:
    """4 类掩码半透明叠加（背景不着色）。"""
    canvas = image_bgr.copy()
    x1, y1, x2, y2 = roi_bbox
    h, w = y2 - y1, x2 - x1
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = canvas[y1:y2, x1:x2].copy()
    for cls, color in ((1, _EXPOSED_COLOR), (2, _ICE_COLOR), (3, _ICICLE_COLOR)):
        overlay[mask == cls] = color
    canvas[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, canvas[y1:y2, x1:x2], 1 - alpha, 0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def draw_heatmap(prob: np.ndarray, out_path: str | Path, colormap: int = cv2.COLORMAP_JET) -> str:
    """冰类概率伪彩图。"""
    p = np.asarray(prob, dtype=np.float32)
    p = (p - p.min()) / max(float(p.max() - p.min()), 1e-6)
    img = cv2.applyColorMap((p * 255).astype(np.uint8), colormap)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)
