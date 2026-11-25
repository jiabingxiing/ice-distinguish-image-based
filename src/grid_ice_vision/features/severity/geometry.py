"""覆冰形态几何：骨架 → 法向 → 亚像素截面宽度。

为什么不用 Hough / LSD：覆冰导线存在悬链线弧垂，冰形不对称（新月形 / D 形），
全局直线拟合会在局部产生数像素误差。骨架 + 法向逐截面测量对曲率不敏感（§10.6.3）。

坐标约定：一律 `(x, y)` = `(列, 行)`，与 OpenCV 一致；仅在访问数组时转成 `[y, x]`。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    sobel,
    spline_filter,
)
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Section:
    """一个采样截面的测量结果。"""

    s: float  # 弧长位置（px）
    center: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    width_px: float
    t_up: float  # 法向负侧边界参数（px）
    t_down: float  # 法向正侧边界参数（px）
    valid: bool


@dataclass(frozen=True)
class WidthMeasurement:
    """整条设备的宽度测量汇总。"""

    width_px: float
    widths: np.ndarray
    sections: tuple[Section, ...]
    valid_ratio: float

    @property
    def n_sections(self) -> int:
        return len(self.sections)


# ------------------------------------------------------------------ 骨架提取
_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _bfs(
    start: tuple[int, int], index: dict[tuple[int, int], int], coords: np.ndarray
) -> tuple[int, np.ndarray]:
    """在骨架像素图上做 BFS，返回 (最远点索引, 前驱数组)。"""
    n = len(coords)
    dist = np.full(n, -1, dtype=np.int32)
    prev = np.full(n, -1, dtype=np.int32)
    start_idx = index[start]
    dist[start_idx] = 0
    q: deque[int] = deque([start_idx])
    far = start_idx
    while q:
        cur = q.popleft()
        cy, cx = coords[cur]
        for dy, dx in _NEIGHBORS:
            nb = (int(cy) + dy, int(cx) + dx)
            j = index.get(nb)
            if j is None or dist[j] >= 0:
                continue
            dist[j] = dist[cur] + 1
            prev[j] = cur
            if dist[j] > dist[far]:
                far = j
            q.append(j)
    return far, prev


def _trace_path(end: int, prev: np.ndarray) -> np.ndarray:
    path: list[int] = []
    cur = end
    while cur >= 0:
        path.append(cur)
        cur = int(prev[cur])
    return np.asarray(path[::-1], dtype=np.int64)


#: 骨架像素上限。超过即判为碎片化掩码，抛错降级为 visual_only（见 `skeleton_polyline`）。
#: 依据：真实导线/绝缘子骨架 10²–10³ px（实测 378 px → 10.9 ms）；
#: 随机掩码可到 8.8×10⁴ px（→ 992 ms）。20,000 px 留足长导线余量，
#: 同时把病态输入的耗时压到可接受范围。
MAX_SKEL_PX: Final[int] = 20_000


def skeleton_polyline(
    mask: np.ndarray, min_pixels: int = 8, max_skel_px: int = MAX_SKEL_PX
) -> np.ndarray:
    """提取掩码的最长中轴折线，返回 `(N, 2)` 的 `(x, y)` 浮点坐标。

    用「两次 BFS 求图直径」而非遍历所有端点：分支（毛刺、冰凌）会被自然剪掉，
    只保留最长的连通路径，正好对应一根导线的走向。

    退化保护
    --------
    BFS 是纯 Python 逐像素迭代，复杂度 O(骨架像素数)。正常的导线/绝缘子串骨架
    只有数百到数千像素；但若上游给出**碎片化**掩码（例如随机权重的分割头，
    或极端噪声），骨架会膨胀到数万像素，单次测量可达 **~1 s**，直接击穿
    180 ms 延迟预算。

    实测对照：真实竖向设备（400×400、骨架 378 px）10.9 ms；
    随机 4 类掩码（骨架 88,075 px）992 ms。因此设上限：
    骨架像素超过 `max_skel_px` 时抛 `ValueError`，由 `pipeline.py` 的既有
    `except ValueError` 分支降级为 `visual_only + requires_review=True`，
    绝不静默输出可疑厚度。
    """
    binary = np.asarray(mask).astype(bool)
    if binary.sum() < min_pixels:
        raise ValueError(f"掩码有效像素过少（{int(binary.sum())}），无法提取骨架")

    skel = skeletonize(binary)
    coords = np.argwhere(skel)  # (N, 2) 的 (row, col)
    if len(coords) < min_pixels:
        raise ValueError(f"骨架像素过少（{len(coords)}）")
    if len(coords) > max_skel_px:
        raise ValueError(
            f"骨架过于碎片化（{len(coords)} px > 上限 {max_skel_px}），"
            "宽度测量不可靠，应降级为 visual_only"
        )

    index = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    a, _ = _bfs((int(coords[0, 0]), int(coords[0, 1])), index, coords)
    b, prev = _bfs((int(coords[a, 0]), int(coords[a, 1])), index, coords)
    path = _trace_path(b, prev)
    pts = coords[path][:, ::-1].astype(float)  # (x, y)
    return pts


def resample_polyline(points: np.ndarray, spacing: float = 4.0) -> np.ndarray:
    """按弧长等距重采样。"""
    if spacing <= 0:
        raise ValueError("spacing 必须为正")
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < spacing:
        return np.vstack([pts[0], pts[-1]])

    n = int(np.floor(total / spacing)) + 1
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def tangents_and_normals(points: np.ndarray, smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """中心差分求切向 τ 与法向 n = (−τ_y, τ_x)。

    先做窗口为 `smooth` 的滑动平均，抑制骨架的像素级锯齿。
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("至少需要 2 个点")
    if smooth > 1 and len(pts) >= smooth:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
        pts = np.stack(
            [
                np.convolve(padded[:, 0], k, mode="valid"),
                np.convolve(padded[:, 1], k, mode="valid"),
            ],
            axis=1,
        )
        pts = pts[: len(points)]

    gx = np.gradient(pts[:, 0])
    gy = np.gradient(pts[:, 1])
    norm = np.hypot(gx, gy)
    norm[norm < 1e-9] = 1.0
    tau = np.stack([gx / norm, gy / norm], axis=1)
    normal = np.stack([-tau[:, 1], tau[:, 0]], axis=1)
    return tau, normal


def curvature(points: np.ndarray) -> np.ndarray:
    """离散曲率 |dθ/ds|（1/px）。"""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.zeros(len(pts))
    d = np.gradient(pts, axis=0)
    dd = np.gradient(d, axis=0)
    cross = np.abs(d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0])
    denom = np.power(np.hypot(d[:, 0], d[:, 1]), 3)
    denom[denom < 1e-9] = 1e-9
    return cross / denom


# ---------------------------------------------------------------- 边缘概率图
def edge_probability(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """由掩码生成归一化边缘概率图。

    对软化的掩码求梯度：对称边缘的梯度极大值恰在边界上，便于后续
    沿法向做亚像素峰位拟合。若上游分割网络能直接给出边缘概率，
    优先传入真实概率图（精度更高）。
    """
    m = gaussian_filter(np.asarray(mask).astype(np.float32), sigma)
    gx = sobel(m, axis=1)
    gy = sobel(m, axis=0)
    e = np.hypot(gx, gy)
    peak = float(e.max())
    return e / peak if peak > 1e-8 else e


def _parabolic_peak(values: np.ndarray, ts: np.ndarray, idx: int) -> float:
    """三点抛物线拟合求亚像素峰位。"""
    if idx <= 0 or idx >= len(values) - 1:
        return float(ts[idx])
    y0, y1, y2 = float(values[idx - 1]), float(values[idx]), float(values[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(ts[idx])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = float(ts[1] - ts[0])
    return float(ts[idx] + delta * step)


def _side_peak(
    vals: np.ndarray,
    ts: np.ndarray,
    side: str,
    min_peak: float,
) -> tuple[float, bool]:
    """在法向的一侧寻找边界峰。返回 (亚像素位置, 是否有效)。"""
    mask = ts < 0 if side == "up" else ts > 0
    if not np.any(mask):
        return 0.0, False
    idxs = np.flatnonzero(mask)
    sub = vals[idxs]
    if sub.size < 3:
        return 0.0, False
    local = int(np.argmax(sub))
    idx = int(idxs[local])
    # 峰落在搜索窗边缘 → 说明边界在窗外，本次测量不可信
    if local == 0 or local == len(idxs) - 1:
        return float(ts[idx]), False
    if float(sub[local]) < min_peak:
        return float(ts[idx]), False
    return _parabolic_peak(vals, ts, idx), True


def measure_widths(
    mask: np.ndarray,
    edge_prob: np.ndarray | None = None,
    spacing: float = 4.0,
    step: float = 0.25,
    search_scale: float = 2.5,
    min_search_px: float = 3.0,
    min_peak: float = 0.05,
    kappa_max: float = 0.5,
    min_valid_ratio: float = 0.3,
) -> WidthMeasurement:
    """沿骨架逐截面测量设备（导线/冰壳）的像素宽度。

    步骤对应 §10.6.1 的 ②–⑦：
        骨架 → 等距重采样 → 切向/法向 → 法向亚像素采样 → 双侧峰位 → 宽度序列
        → 曲率/边界有效性过滤 → IQR 截尾均值。
    """
    binary = np.asarray(mask).astype(bool)
    poly = skeleton_polyline(binary)
    poly = resample_polyline(poly, spacing=spacing)
    _, normals = tangents_and_normals(poly)
    kappa = curvature(poly)

    edge = (
        edge_probability(binary) if edge_prob is None else np.asarray(edge_prob, dtype=np.float32)
    )
    dt = distance_transform_edt(binary)

    # 三次样条系数必须**预计算一次**：`map_coordinates(order=3)` 默认每次调用都会
    # 对整幅图重做 spline_filter。实测 400×400 竖向设备、95 个截面时，
    # 逐截面调用 → 427 ms；预滤波后同一批采样仅 7.8 ms（55× 加速）。
    # 这是 §7.2「CPU P95 ≤ 180 ms」门槛能否达成的关键，且数值结果完全一致
    # （同一插值核，只是把可复用的滤波提出来做一次）。
    edge_spline = spline_filter(np.asarray(edge, dtype=np.float64), order=3)

    sections: list[Section] = []
    for i, (p, n) in enumerate(zip(poly, normals, strict=True)):
        half = max(min_search_px, search_scale * float(dt[round(p[1]), round(p[0])]))
        ts = np.arange(-half, half + step * 0.5, step)
        pts = p[None, :] + ts[:, None] * n[None, :]
        vals = map_coordinates(
            edge_spline, [pts[:, 1], pts[:, 0]], order=3, mode="nearest", prefilter=False
        )

        t_up, ok_up = _side_peak(vals, ts, "up", min_peak)
        t_down, ok_down = _side_peak(vals, ts, "down", min_peak)
        width = t_down - t_up
        valid = bool(ok_up and ok_down and width > 0 and abs(kappa[i]) <= kappa_max)
        sections.append(
            Section(
                s=float(i * spacing),
                center=(float(p[0]), float(p[1])),
                tangent=(float(normals[i, 1]), float(-normals[i, 0])),
                normal=(float(n[0]), float(n[1])),
                width_px=float(width),
                t_up=t_up,
                t_down=t_down,
                valid=valid,
            )
        )

    widths = np.array([s.width_px for s in sections if s.valid], dtype=float)
    ratio = float(widths.size / max(len(sections), 1))
    if widths.size == 0:
        raise ValueError("无有效截面：掩码可能过细、边界模糊或被 ROI 截断")
    if ratio < min_valid_ratio:
        raise ValueError(f"有效截面比例过低（{ratio:.2f} < {min_valid_ratio}）")

    from grid_ice_vision.features.severity.thickness import trimmed_mean_width

    return WidthMeasurement(
        width_px=trimmed_mean_width(widths),
        widths=widths,
        sections=tuple(sections),
        valid_ratio=ratio,
    )


# ------------------------------------------------------------ 覆盖率 / 桥接率
def coverage_ratio(ice_pixels: int, exposed_pixels: int) -> float:
    """覆盖率 = N_ice / (N_ice + N_exposed)。

    分母必须是**有效设备像素**。若只标了冰没标裸露设备，这个数没有意义
    —— 这是 §8.4 坚持保留 `exposed_asset` 类的原因。
    """
    denom = ice_pixels + exposed_pixels
    if denom <= 0:
        raise ValueError("有效设备像素为 0，无法计算覆盖率")
    return float(ice_pixels) / float(denom)


def bridge_ratio(icicle_pixels: int, gap_pixels: int) -> float | None:
    """桥接率 = 伞裙间隙被冰凌占据的比例。

    仅在 ROI 为 `insulator_string` 且能定位伞裙间隙时计算，否则返回 None。
    """
    if gap_pixels <= 0:
        return None
    return float(min(icicle_pixels / gap_pixels, 1.0))
