"""覆冰类型评估可视化：混淆矩阵热力图、校准可靠性曲线与错判样本图集（纯 OpenCV/NumPy）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.data.transforms.common import _imread_unicode


def _save_png(out_path: Path, img_bgr: np.ndarray) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {out_path}")
    out_path.write_bytes(buf.tobytes())
    return out_path


def render_confusion_heatmap(
    matrix: list[list[int]] | np.ndarray,
    labels: tuple[str, ...] | list[str] = ICE_TYPE_FOUR_CLASSES,
    out_path: str | Path = "artifacts/ice_type/v1/confusion_matrix.png",
    *,
    title: str = "Ice Type Confusion Matrix (Row=True, Col=Pred)",
) -> Path:
    """绘制 4x4 混淆矩阵热力图并保存为 PNG（兼容中文路径）。"""
    mat = np.asarray(matrix, dtype=np.int64)
    k = len(labels)
    row_sum = np.maximum(mat.sum(axis=1, keepdims=True), 1)
    norm = mat.astype(np.float64) / row_sum

    cell = 110
    margin_l, margin_t, margin_r, margin_b = 140, 95, 40, 80
    w = margin_l + k * cell + margin_r
    h = margin_t + k * cell + margin_b
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(
        canvas,
        "Predicted",
        (margin_l + (k * cell) // 2 - 45, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (60, 60, 60),
        1,
    )

    for j, name in enumerate(labels):
        cx = margin_l + j * cell + 14
        cv2.putText(
            canvas,
            str(name)[:10],
            (cx, margin_t - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
        )

    for i, name in enumerate(labels):
        y0 = margin_t + i * cell
        cv2.putText(
            canvas,
            str(name)[:10],
            (16, y0 + cell // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
        )
        for j in range(k):
            x0 = margin_l + j * cell
            ratio = float(norm[i, j])
            # 蓝色渐变 (245->30 B, 245->110 G, 250->220 R inverted)
            b = int(245 - ratio * 55)
            g = int(245 - ratio * 155)
            r = int(245 - ratio * 210)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (b, g, r), -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (180, 180, 180), 1)
            txt_col = (255, 255, 255) if ratio >= 0.52 else (20, 20, 20)
            cv2.putText(
                canvas,
                f"{int(mat[i, j])}",
                (x0 + 22, y0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                txt_col,
                2,
            )
            cv2.putText(
                canvas,
                f"{ratio:.1%}",
                (x0 + 20, y0 + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                txt_col,
                1,
            )

    return _save_png(Path(out_path), canvas)


def render_calibration_plot(
    probs: np.ndarray,
    targets: np.ndarray,
    out_path: str | Path = "artifacts/ice_type/v1/calibration_curve.png",
    *,
    n_bins: int = 10,
    ece: float | None = None,
    title: str = "Reliability Diagram (Confidence vs Accuracy)",
) -> Path:
    """绘制置信度-准确率可靠性校准图。"""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).ravel()
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)

    w, h = 560, 480
    pad_l, pad_t, pad_r, pad_b = 75, 65, 35, 65
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, title, (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
    if ece is not None:
        cv2.putText(
            canvas,
            f"ECE = {ece:.4f}",
            (pad_l + 15, pad_t + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 90, 190),
            2,
        )

    cv2.rectangle(canvas, (pad_l, pad_t), (pad_l + pw, pad_t + ph), (120, 120, 120), 1)
    # 理想对角线 y = x
    cv2.line(canvas, (pad_l, pad_t + ph), (pad_l + pw, pad_t), (150, 150, 150), 1, cv2.LINE_AA)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    bw = pw / n_bins
    for b in range(n_bins):
        m = idx == b
        x1 = round(pad_l + b * bw + 2)
        x2 = round(pad_l + (b + 1) * bw - 2)
        if m.any():
            acc = float(correct[m].mean())
            y_bar = round(pad_t + ph * (1.0 - acc))
            cv2.rectangle(canvas, (x1, y_bar), (x2, pad_t + ph), (220, 130, 50), -1)
            cv2.rectangle(canvas, (x1, y_bar), (x2, pad_t + ph), (160, 80, 20), 1)
        tick_x = round(pad_l + b * bw)
        cv2.putText(
            canvas,
            f"{b / n_bins:.1f}",
            (tick_x - 8, pad_t + ph + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (50, 50, 50),
            1,
        )

    cv2.putText(
        canvas,
        "1.0",
        (pad_l + pw - 14, pad_t + ph + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (50, 50, 50),
        1,
    )
    cv2.putText(
        canvas,
        "Confidence",
        (pad_l + pw // 2 - 40, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 40, 40),
        1,
    )
    cv2.putText(
        canvas, "Acc", (15, pad_t + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1
    )
    return _save_png(Path(out_path), canvas)


def render_error_gallery(
    image_paths: list[Path],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    out_dir: str | Path,
    *,
    class_names: tuple[str, ...] = ICE_TYPE_FOUR_CLASSES,
    max_per_pair: int = 4,
    tile_size: int = 224,
) -> list[Path]:
    """为主要易混淆错判对生成对比拼接图，落盘供人工复盘。"""
    gallery_dir = Path(out_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    pr = np.asarray(probs, dtype=np.float64)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (t, pd_val) in enumerate(zip(y, p, strict=True)):
        if t != pd_val:
            buckets.setdefault((int(t), int(pd_val)), []).append(idx)

    saved: list[Path] = []
    # 按错误样本数降序取最严重的混淆对
    ranked = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)[:6]
    for (t_idx, p_idx), idxs in ranked:
        # 优先挑最高自信错判（最危险样本）
        idxs_sorted = sorted(idxs, key=lambda i: float(pr[i, p_idx]), reverse=True)[:max_per_pair]
        tiles: list[np.ndarray] = []
        for i in idxs_sorted:
            im = _imread_unicode(image_paths[i], cv2.IMREAD_COLOR)
            if im is None:
                continue
            tile = cv2.resize(im, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            header = np.full((34, tile_size, 3), 25, dtype=np.uint8)
            t_name = class_names[t_idx]
            p_name = class_names[p_idx]
            conf = float(pr[i, p_idx])
            cv2.putText(
                header,
                f"T:{t_name} P:{p_name} {conf:.2f}",
                (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 255),
                1,
            )
            tiles.append(np.vstack([header, tile]))
        if not tiles:
            continue
        grid: Any = np.hstack(tiles)
        fname = f"err_{class_names[t_idx]}_as_{class_names[p_idx]}.png"
        saved.append(_save_png(gallery_dir / fname, grid))
    return saved
