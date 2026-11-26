"""基于 SCUT 多帧结构显著性生成终端归一化 ROI 表与预览图。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402
from grid_ice_vision.data.manifest import ManifestRecord, load_manifest  # noqa: E402
from grid_ice_vision.data.transforms.common import _imread_unicode  # noqa: E402
from grid_ice_vision.features.ice_type.roi import (  # noqa: E402
    NormalizedRoi,
    compute_view_signature,
    crop_normalized_roi,
)


def _edge_saliency_map(img_bgr: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (7, 7), 0)
    # 屏蔽顶部与底部 6% OSD 时间戳条带
    h, w = mag.shape
    os_h = max(1, round(h * 0.06))
    mag[:os_h, :] = 0.0
    mag[h - os_h :, :] = 0.0
    # 抑制大面积低对比背景纹理，仅保留高于中位数的显著结构轮廓
    med = float(np.median(mag[os_h : h - os_h, :]))
    mag = np.maximum(mag - med * 0.85, 0.0)
    # 轻微中心先验权重
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.5, w * 0.5
    gauss = np.exp(-(((xx - cx) / (0.52 * w)) ** 2 + ((yy - cy) / (0.55 * h)) ** 2))
    weighted = mag * (0.55 + 0.45 * gauss.astype(np.float32))
    mx = float(weighted.max())
    return weighted / mx if mx > 1e-6 else weighted


def _bbox_from_saliency(sal: np.ndarray) -> tuple[float, float, float, float]:
    h, w = sal.shape
    if float(sal.max()) <= 1e-6:
        return (0.18, 0.12, 0.82, 0.88)

    # 使用多尺度滑动窗口在结构显著图中寻找最紧凑的绝缘子/导线核心区域，避免退化为整图框
    integ = cv2.integral(sal.astype(np.float64))
    y_min = max(1, round(h * 0.05))
    y_max = min(h, round(h * 0.95))
    x_min = 0
    x_max = w

    best_score = -1.0
    best_box = (0.18, 0.12, 0.82, 0.88)
    for wf, hf in (
        (0.50, 0.55),
        (0.56, 0.62),
        (0.64, 0.52),
        (0.50, 0.72),
        (0.66, 0.70),
    ):
        bw = max(12, round(w * wf))
        bh = max(12, round(h * hf))
        area_frac = (bw * bh) / float(w * h)
        denom = area_frac**0.72
        for y1 in range(y_min, max(y_min + 1, y_max - bh + 1), 3):
            y2 = y1 + bh
            if y2 > y_max:
                continue
            for x1 in range(x_min, max(x_min + 1, x_max - bw + 1), 3):
                x2 = x1 + bw
                if x2 > x_max:
                    continue
                energy = float(integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1])
                score = energy / denom
                if score > best_score:
                    best_score = score
                    best_box = (
                        round(x1 / w, 3),
                        round(y1 / h, 3),
                        round(x2 / w, 3),
                        round(y2 / h, 3),
                    )
    return best_box


def generate_rois(
    manifest_path: Path,
    out_csv: Path,
    preview_png: Path,
    *,
    max_frames_per_monitor: int = 10,
) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    root = paths.scut_root()
    by_mon: dict[str, list[ManifestRecord]] = defaultdict(list)
    for r in records:
        by_mon[r.camera_id].append(r)

    rows: list[dict[str, Any]] = []
    preview_tiles: list[np.ndarray] = []
    drift_counts = {"fixed_v1": 0, "variable_view": 0}

    for mid in sorted(by_mon):
        recs = by_mon[mid]
        step = max(1, len(recs) // max_frames_per_monitor)
        sampled = recs[::step][:max_frames_per_monitor]

        sal_maps: list[np.ndarray] = []
        sigs: list[np.ndarray] = []
        rep_img: np.ndarray | None = None
        for r in sampled:
            im = _imread_unicode(root / r.path, cv2.IMREAD_COLOR)
            if im is None:
                continue
            if rep_img is None:
                rep_img = im
            sal_maps.append(_edge_saliency_map(im))
            default_roi = NormalizedRoi(mid, 0.15, 0.1, 0.85, 0.9)
            sigs.append(compute_view_signature(crop_normalized_roi(im, default_roi)))

        if not sal_maps or rep_img is None:
            x1, y1, x2, y2 = (0.18, 0.10, 0.82, 0.90)
            mean_corr = 1.0
        else:
            avg_sal = np.mean(np.stack(sal_maps, axis=0), axis=0)
            x1, y1, x2, y2 = _bbox_from_saliency(avg_sal)
            if len(sigs) >= 2:
                ref = sigs[0]
                corrs = [float(np.clip(np.dot(ref, s), 0.0, 1.0)) for s in sigs[1:]]
                mean_corr = float(np.mean(corrs))
            else:
                mean_corr = 1.0

        obj_counts = Counter(str(r.labels.get("object_type", "")) for r in recs)
        top_obj = obj_counts.most_common(1)[0][0]
        asset_type = "insulator" if top_obj == "未包含线路" else "insulator_conductor"
        view_ver = "v1" if mean_corr >= 0.35 else "v1_var"
        if view_ver == "v1":
            drift_counts["fixed_v1"] += 1
        else:
            drift_counts["variable_view"] += 1

        split = recs[0].split
        rows.append(
            {
                "monitor_id": mid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "asset_type": asset_type,
                "view_version": view_ver,
                "notes": f"split={split};n={len(recs)};corr={mean_corr:.2f}",
            }
        )

        # 收集前 24 个多样本终端（优先覆盖 test 与 val）制作预览图
        if (
            rep_img is not None
            and len(preview_tiles) < 24
            and (split in ("test", "val") or len(recs) >= 15)
        ):
            vis = cv2.resize(rep_img, (240, 180), interpolation=cv2.INTER_AREA)
            h, w = vis.shape[:2]
            cv2.rectangle(
                vis,
                (round(x1 * w), round(y1 * h)),
                (round(x2 * w), round(y2 * h)),
                (0, 220, 40),
                2,
            )
            cv2.putText(
                vis,
                f"{mid} [{split}] c={mean_corr:.2f}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            preview_tiles.append(vis)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "monitor_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "asset_type",
                "view_version",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if preview_tiles:
        cols = 6
        n_rows = (len(preview_tiles) + cols - 1) // cols
        blank = np.zeros_like(preview_tiles[0])
        while len(preview_tiles) < n_rows * cols:
            preview_tiles.append(blank)
        grid = np.vstack(
            [np.hstack(preview_tiles[r * cols : (r + 1) * cols]) for r in range(n_rows)]
        )
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", grid)
        if ok:
            preview_png.write_bytes(buf.tobytes())

    return {
        "n_monitors": len(rows),
        "out_csv": str(out_csv),
        "preview_png": str(preview_png),
        "drift_counts": drift_counts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成终端归一化 ROI 配置表与可视化复核网格")
    ap.add_argument("--manifest", default="data/manifests/scut_ice_type_v1.jsonl")
    ap.add_argument("--out", default="configs/data/monitor_roi_v1.csv")
    ap.add_argument("--preview", default="artifacts/reports/monitor_roi_preview.png")
    args = ap.parse_args(argv)

    res = generate_rois(
        PROJ / args.manifest,
        PROJ / args.out,
        PROJ / args.preview,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
