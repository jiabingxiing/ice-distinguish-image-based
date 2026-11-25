"""同终端无冰基准自动检索与配准（计划 §3.3 BASE、§4.2）。

用途：为绝缘子面积变化率 `Pa` 提供「无冰期参考帧」。没有基准帧时 `Pa = null`，
Track A 退化为单帧覆盖率 + 偏冰分布（仍然可用，但少了最有力的时序证据）。

两级策略
--------
1. **候选检索**：同 `monitorId` 的无冰帧，按 IoU ≥ 阈值与时间邻近度排序；
2. **几何配准**：对候选用 ECC（`cv2.findTransformECC`）做仿射微配准，
   把基准 ROI 变换到当前帧坐标系，残余相关低于阈值则弃用该候选。

设计取舍
--------
- 用 **ECC 相关** 而非 ORB 特征点：监控图多为低清（93.2% ≤704 px）、纹理少，
  ORB 在平坦天空/雾天几乎无特征点；ECC 直接优化像素相关，对低纹理更鲁棒。
- **不做** 全自动在线学习：基准库由离线脚本 `scripts/build_baseline_index.py`
  按 `monitorId` 建索引，运行时只查不改，避免线上状态漂移。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineConfig:
    """匹配阈值。默认值与 `configs/model/pipeline.yaml` 的 `baseline` 段一致。"""

    #: 候选框与当前框的 IoU 下限（计划 §3.3 规定 0.6）
    iou_min: float = 0.60
    #: ECC 配准后的最小相关系数；低于该值说明机位已偏移，弃用基准
    ecc_corr_min: float = 0.60
    #: 同一 monitorId 最多参与配准的候选数（控制时延）
    max_candidates: int = 3
    #: 基准图与当前帧的最大时间差（小时）；0 表示不限制
    max_age_hours: float = 0.0
    #: 配准失败时是否允许退化为「不配准直接用」，默认禁止（宁可不给 Pa）
    allow_unregistered: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> BaselineConfig:
        sec = (cfg or {}).get("baseline", {}) or {}
        return cls(
            iou_min=float(sec.get("iou_min", 0.60)),
            ecc_corr_min=float(sec.get("ecc_corr_min", 0.60)),
            max_candidates=int(sec.get("max_candidates", 3)),
            max_age_hours=float(sec.get("max_age_hours", 0.0)),
            allow_unregistered=bool(sec.get("allow_unregistered", False)),
        )


@dataclass(frozen=True)
class BaselineEntry:
    """基准库中的一条记录。"""

    monitor_id: str
    image_ref: str
    """基准图路径（项目相对 POSIX 或外部库相对路径）。"""
    bbox: tuple[float, float, float, float]
    timestamp: str = ""
    is_clean: bool = True


@dataclass(frozen=True)
class MatchResult:
    """一次基准匹配的结果。`entry is None` 表示无可用基准。"""

    entry: BaselineEntry | None
    iou: float
    ecc_corr: float | None
    registered_bbox: tuple[float, float, float, float] | None
    reason: str


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """两个 (x1, y1, x2, y2) 框的 IoU；任一退化框返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _crop(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 2)), min(h, max(y2, y1 + 2))
    return image[y1:y2, x1:x2]


def register_ecc(
    current: np.ndarray,
    baseline: np.ndarray,
    max_iters: int = 60,
    eps: float = 1e-5,
) -> tuple[float, np.ndarray] | None:
    """用 ECC 估计基准 → 当前的单应（这里用仿射）变换。

    返回 `(相关系数, 2×3 仿射矩阵)`；失败返回 None（尺寸退化、相关过低等）。
    """
    cur_gray = _to_gray(current)
    base_gray = _to_gray(baseline)
    if cur_gray is None or base_gray is None:
        return None
    if cur_gray.shape[0] < 8 or cur_gray.shape[1] < 8:
        return None
    # 统一到基准尺寸，避免尺度差异直接导致 ECC 不收敛
    cur_gray = np.ascontiguousarray(
        cv2.resize(
            cur_gray,
            (base_gray.shape[1], base_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.float32,
    )
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
    try:
        # 运行时返回 (corr, warpMatrix)。OpenCV stubs 的部分重载标注与此不一致，
        # 故显式忽略该重载告警，返回值统一按二元组解包。
        corr_raw, warp_raw = cv2.findTransformECC(  # type: ignore[call-overload]
            base_gray, cur_gray, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        return None
    return float(corr_raw), np.asarray(warp_raw, dtype=np.float32)


def _to_gray(image: np.ndarray) -> np.ndarray[Any, np.dtype[np.float32]] | None:
    a = np.asarray(image)
    if a.size == 0:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    out = np.ascontiguousarray(a, dtype=np.float32)
    if float(np.max(out)) > 1.5:
        out /= 255.0
    return out


def match_baseline(
    current_image: np.ndarray,
    current_bbox: tuple[float, float, float, float],
    monitor_id: str,
    candidates: list[BaselineEntry],
    load_image: Any,
    cfg: BaselineConfig | None = None,
) -> MatchResult:
    """在候选中挑选并配准最合适的无冰基准。

    `load_image(entry) -> np.ndarray | None` 由调用方注入，用于把 `image_ref`
    解析成真实图像（模块本身不读文件，保持可单测）。
    """
    c = cfg or BaselineConfig()
    pool = [e for e in candidates if e.monitor_id == monitor_id and e.is_clean]
    if not pool:
        return MatchResult(None, 0.0, None, None, "no_candidate")

    scored: list[tuple[float, BaselineEntry]] = []
    for e in pool:
        iou = bbox_iou(current_bbox, e.bbox)
        if iou >= c.iou_min:
            scored.append((iou, e))
    if not scored:
        best_iou = max((bbox_iou(current_bbox, e.bbox) for e in pool), default=0.0)
        return MatchResult(None, best_iou, None, None, "iou_below_threshold")

    scored.sort(key=lambda t: t[0], reverse=True)
    cur_crop = _crop(current_image, current_bbox)
    best_corr = -1.0
    best: MatchResult | None = None

    for iou, e in scored[: max(1, c.max_candidates)]:
        base_img = load_image(e)
        if base_img is None:
            continue
        base_crop = _crop(base_img, e.bbox)
        got = register_ecc(cur_crop, base_crop)
        if got is None:
            continue
        corr, warp = got
        if corr < c.ecc_corr_min:
            if best is None:
                best = MatchResult(e, iou, corr, None, "ecc_corr_below_threshold")
            continue
        # 把基准框经仿射变换映射到当前帧坐标
        cx = np.array([[e.bbox[0], e.bbox[1]], [e.bbox[2], e.bbox[3]]], dtype=np.float32)
        mapped = (warp @ np.vstack([cx.T, np.ones(2, dtype=np.float32)])).T
        xs, ys = mapped[:, 0], mapped[:, 1]
        reg = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        # 配准后框应落在当前图内且与当前框重合度不能太差
        if bbox_iou(reg, current_bbox) < c.iou_min:
            continue
        if corr > best_corr:
            best_corr = corr
            best = MatchResult(e, iou, corr, reg, "matched")

    if best is not None:
        return best
    if c.allow_unregistered:
        iou, e = scored[0]
        return MatchResult(e, iou, None, e.bbox, "unregistered_allowed")
    return MatchResult(None, scored[0][0], None, None, "registration_failed")


# ------------------------------------------------------------------ 离线索引
def load_index(path: str | Path) -> dict[str, list[BaselineEntry]]:
    """读取按 monitorId 分组的基准索引（JSONL，每行一条）。

    离线由 `scripts/build_baseline_index.py` 生成；运行时只读。
    索引内路径只存相对 POSIX，换盘后配合 `common/paths.py` 解析。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"基准索引不存在: {p}")
    out: dict[str, list[BaselineEntry]] = {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            d = json.loads(s)
            bbox = tuple(float(v) for v in d["bbox"])
            if len(bbox) != 4:
                raise ValueError("bbox 必须 4 个元素")
            entry = BaselineEntry(
                monitor_id=str(d["monitor_id"]),
                image_ref=str(d["image_ref"]),
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                timestamp=str(d.get("timestamp", "")),
                is_clean=bool(d.get("is_clean", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{p}:{i} 解析失败: {exc}") from exc
        out.setdefault(entry.monitor_id, []).append(entry)
    return out
