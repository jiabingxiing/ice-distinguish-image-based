"""合成自检：在没有真实数据、没有权重的情况下验证物理量算链路。

对应 DoD（§15.4）：合成圆筒真值上像素宽度误差 ≤ 0.8 px（软目标 0.5 px）。
跑法：`uv run python scripts/demo_synthetic.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.application.evidence import draw_caliper
from grid_ice_vision.evaluation.thickness import width_error_px
from grid_ice_vision.features.severity.geometry import measure_widths
from grid_ice_vision.features.severity.thickness import (
    equivalent_thickness_mm,
    radial_thickness_mm,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesize_ice import SHAPES, make_synthetic_scene


def main() -> int:
    out_dir = Path("artifacts/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    thicknesses = [3.0, 6.0, 10.0, 14.0]
    # 各截面模板在法向两侧的半径跨度系数：w = w0 + t·span
    span = {"cylinder": 2.0, "c2": 1.0, "c7": 1.4, "c11": 1.6}
    rows: list[dict[str, float | str]] = []
    err_w: list[float] = []

    for shape in SHAPES:
        for t in thicknesses:
            img, mask, truth = make_synthetic_scene(360, 720, t, shape, seed=int(t * 10))
            bare = (mask == 1).astype(np.uint8)
            iced = ((mask == 1) | (mask == 2)).astype(np.uint8)

            w0 = measure_widths(bare, spacing=4.0).width_px
            wm = measure_widths(iced, spacing=4.0)
            alpha = truth["d0_mm"] / w0 if "d0_mm" in truth else 30.0 / w0
            d_radial = radial_thickness_mm(wm.width_px, w0, alpha)
            d_eq = equivalent_thickness_mm(d_radial, 30.0, 900.0)

            err = abs(wm.width_px - (w0 + t * span[shape]))
            err_w.append(err)
            rows.append(
                {
                    "shape": shape,
                    "t_px": t,
                    "w0_px": round(w0, 3),
                    "w_ice_px": round(wm.width_px, 3),
                    "w_expected": round(w0 + t * span[shape], 3),
                    "err_px": round(err, 4),
                    "d_radial_mm": round(d_radial, 3),
                    "d_eq_mm": round(d_eq, 3),
                    "sections": wm.n_sections,
                }
            )
            if shape == "cylinder" and t == 6.0:
                draw_caliper(
                    img,
                    wm.sections,
                    w0,
                    out_dir / f"caliper_{shape}_{int(t)}px.png",
                    d_radial_mm=d_radial,
                    alpha_mm_per_px=alpha,
                    grade="G2",
                )

    print(f"{'shape':<9}{'t_px':>6}{'w0':>8}{'w_ice':>8}{'expect':>8}{'err_px':>8}{'d_eq_mm':>9}")
    for r in rows:
        print(
            f"{r['shape']:<9}{r['t_px']:>6.1f}{r['w0_px']:>8.2f}{r['w_ice_px']:>8.2f}"
            f"{r['w_expected']:>8.2f}{r['err_px']:>8.3f}{r['d_eq_mm']:>9.2f}"
        )

    stats = width_error_px(
        np.array([r["w_ice_px"] for r in rows]), np.array([r["w_expected"] for r in rows])
    )
    summary = {"width_error": stats, "max_err_px": max(err_w), "n_cases": len(rows)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
    )

    print("\n宽度误差统计：", stats)
    print("注：d_eq 的绝对值受合成尺度影响（α = 30 mm / 裸导线像素宽度），")
    print("    合成图里导线只有 11 px，因此 α≈2.7 mm/px、厚度看起来偏大。")
    print("    本自检关心的是**误差**（宽度差 vs 注入厚度），不是绝对毫米数。")
    print("证据图：", out_dir / "caliper_cylinder_6px.png")

    hard_limit = 0.8
    if stats["mean_px"] > hard_limit:
        print(f"✗ 平均宽度误差 {stats['mean_px']:.3f} px 超过硬门槛 {hard_limit} px")
        return 1
    print(f"✓ 平均宽度误差 {stats['mean_px']:.3f} px ≤ {hard_limit} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
