"""测量 SCUT 图像的可标注性：目标尺寸、边界复杂度、四类掩码的可行性。

目的：把「1,500 张掩码」翻译成可核算的工时，而不是凭感觉。
输出：artifacts/reports/annotation_effort_probe.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402

#: 每种分辨率抽取的张数（按分辨率分层，而非按冰种，因为工时由像素规模主导）
N_PER_TYPE = 6


def read_bgr(p: Path) -> np.ndarray | None:
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def main() -> int:
    import pandas as pd

    root = paths.scut_root()
    df = pd.read_csv(paths.scut_csv(), encoding="utf-8-sig")
    df["rel"] = [f"{int(r.Year)}/{r.Image_Name}.jpg" for r in df.itertuples(index=False)]
    df = df[[(root / p).exists() for p in df["rel"]]].reset_index(drop=True)

    # 先扫出每张图的真实分辨率再分层抽样（CSV 里没有分辨率列）。
    # 用 cv2 读 header 而不解码像素，速度快一个量级。
    def probe_shape(p: str) -> tuple[int, int] | None:
        buf = np.fromfile(str(root / p), dtype=np.uint8)
        if buf.size == 0:
            return None
        im = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None:
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            return None
        return int(im.shape[1]), int(im.shape[0])

    shapes = [probe_shape(p) for p in df["rel"]]
    df["_w"] = [s[0] if s else 0 for s in shapes]
    df["_h"] = [s[1] if s else 0 for s in shapes]
    df = df[(df["_w"] > 0) & (df["_h"] > 0)].reset_index(drop=True)
    print(f"分辨率分层：{df.groupby(['_w', '_h']).size().to_dict()}")

    rows = []
    # 按分辨率分层抽样：直接 head() 会全取到 640x480（时间序靠前的终端都是低清），
    # 无法反映高分辨率图像的标注成本差异。
    for _, g in df.groupby(["_w", "_h"]):
        for r in g.head(N_PER_TYPE).itertuples(index=False):
            p = root / r.rel
            im = read_bgr(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # 拉普拉斯方差：越低越模糊 → 边界越难逐像素判读
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 梯度幅值分布的 P99：反映可辨认边界的锐利度
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            p99 = float(np.percentile(mag, 99))
            frac_strong = float((mag > p99 * 0.3).mean())

            rows.append(
                {
                    "year": int(r.Year),
                    "monitor": str(r.monitorId),
                    "ice_type": str(r.Ice_Type),
                    "object_type": str(r.ObjectType),
                    "w": int(w),
                    "h": int(h),
                    "megapixels": round(w * h / 1e6, 3),
                    "laplacian_var": round(lap_var, 1),
                    "grad_p99": round(p99, 1),
                    "strong_edge_frac": round(frac_strong, 5),
                    "jpeg_bytes": int(p.stat().st_size),
                }
            )

    res = Counter((r["w"], r["h"]) for r in rows)
    lap = np.array([r["laplacian_var"] for r in rows])
    mp = np.array([r["megapixels"] for r in rows])
    edges = np.array([r["strong_edge_frac"] for r in rows])

    report = {
        "n_sampled": len(rows),
        "resolution_top": [{"w": w, "h": h, "count": c} for (w, h), c in res.most_common()],
        "megapixels": {"min": float(mp.min()), "p50": float(np.median(mp)), "max": float(mp.max())},
        "laplacian_var": {
            "min": float(lap.min()),
            "p25": float(np.percentile(lap, 25)),
            "p50": float(np.median(lap)),
            "p75": float(np.percentile(lap, 75)),
            "max": float(lap.max()),
            "n_below_100": int((lap < 100).sum()),
        },
        "strong_edge_frac_p50": float(np.median(edges)),
        "rows": rows,
        "interpretation": {
            "laplacian_var": (
                "拉普拉斯方差 < 100 通常意味着明显模糊/低对比，"
                "此类图像上冰层与设备的边界难以逐像素判读"
            ),
            "megapixels": "单图像素规模决定逐像素刷掩码的工时基线",
        },
    }

    out = PROJ / "artifacts" / "reports" / "annotation_effort_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"抽样 {len(rows)} 张")
    print(f"分辨率分布 {report['resolution_top']}")
    print(f"百万像素  中位 {np.median(mp):.3f}  范围 {mp.min():.3f}–{mp.max():.3f}")
    print(
        f"拉普拉斯方差 中位 {np.median(lap):.1f}  "
        f"P25 {np.percentile(lap, 25):.1f}  <100 的张数 {int((lap < 100).sum())}/{len(rows)}"
    )
    print(f"写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
