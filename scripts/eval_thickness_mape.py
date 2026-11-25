"""合成真值厚度回归：验证 `estimate_thickness` 的 MAPE（计划 §1.3 T5 硬门槛）。

为什么不靠训练：合成算子的厚度 `t` 是我们自己给的，属于**零误差几何真值**，
因此可以在没有任何权重、没有任何现场标注的情况下，直接检验
「像素宽度 → 径向厚度 → 等值厚度」这条物理链路是否自洽。
这是 CPU 上能拿到的最强证据。

口径（与 `AGENTS.md` §3 一致）
------------------------------
- MAPE 只在 `gt ≥ 3 mm` 样本上计算（小厚度相对误差发散，混入会失真）；
- 0 厚度与 `gt < 3 mm` 样本单列，不混入分母；
- 硬门槛 12%，软目标 8%（计划 §1.3 T5）。

跑法：`uv run python scripts/eval_thickness_mape.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

from grid_ice_vision.features.severity.geometry import measure_widths  # noqa: E402
from grid_ice_vision.features.severity.thickness import (  # noqa: E402
    equivalent_thickness_mm,
    radial_thickness_mm,
)
from synthesize_ice import SHAPES, make_synthetic_scene  # noqa: E402

#: 计划 §1.3 的 MAPE 只统计 gt ≥ 3 mm 的样本
MAPE_MIN_GT_MM = 3.0
HARD_LIMIT_PCT = 12.0
SOFT_TARGET_PCT = 8.0

#: 各截面模板在法向两侧的半径跨度系数：w = w0 + t·span（与 demo_synthetic 一致）
SPAN = {"cylinder": 2.0, "c2": 1.0, "c7": 1.4, "c11": 1.6}


def run_case(shape: str, t_px: float, seed: int) -> dict[str, float | str]:
    """生成一个合成截面，测量宽度并反解厚度。

    `alpha`（mm/px）由「裸导线真实外径 d0_mm ÷ 实测裸线像素宽 w0」得到，
    与线上标定口径一致（L1 刚性金具/PTZ 求 α 后在本文中复用同一算式）。
    """
    img, mask, truth = make_synthetic_scene(360, 720, t_px, shape, seed=seed)
    del img  # 本脚本只用掩码与真值，渲染图不参与度量
    bare = (mask == 1).astype(np.uint8)
    iced = ((mask == 1) | (mask == 2)).astype(np.uint8)

    w0 = measure_widths(bare, spacing=4.0).width_px
    w_ice = measure_widths(iced, spacing=4.0).width_px

    d0_mm = float(truth["d0_mm"]) if "d0_mm" in truth else 30.0
    alpha = d0_mm / w0

    d_radial = radial_thickness_mm(w_ice, w0, alpha)
    d_eq = equivalent_thickness_mm(d_radial, d0_mm, 900.0)

    # 真值：注入厚度 t_px 折算到毫米（单侧径向）
    span = SPAN[shape]
    gt_mm = 0.5 * (t_px * span) * alpha

    return {
        "shape": shape,
        "t_px": t_px,
        "w0_px": round(w0, 3),
        "w_ice_px": round(w_ice, 3),
        "alpha_mm_per_px": round(alpha, 5),
        "gt_mm": round(gt_mm, 4),
        "pred_mm": round(d_radial, 4),
        "abs_err_mm": round(abs(d_radial - gt_mm), 4),
        "rel_err_pct": round(abs(d_radial - gt_mm) / gt_mm * 100.0, 3) if gt_mm > 0 else None,
        "d_eq_mm": round(d_eq, 4),
    }


def main() -> int:
    thicknesses = [1.0, 3.0, 6.0, 10.0, 14.0, 18.0]
    rows: list[dict[str, float | str]] = []
    for shape in SHAPES:
        for t in thicknesses:
            try:
                rows.append(run_case(shape, t, seed=int(t * 10)))
            except ValueError as exc:
                rows.append({"shape": shape, "t_px": t, "error": str(exc)})

    ok = [r for r in rows if "error" not in r and r["gt_mm"] is not None]
    eligible = [r for r in ok if float(r["gt_mm"]) >= MAPE_MIN_GT_MM]
    below = [r for r in ok if float(r["gt_mm"]) < MAPE_MIN_GT_MM]

    def mape(items: list[dict[str, float | str]]) -> float | None:
        if not items:
            return None
        return float(np.mean([float(r["rel_err_pct"]) for r in items]))

    mape_all = mape(eligible)
    by_shape: dict[str, float] = {}
    for shape in SHAPES:
        sub = [r for r in eligible if r["shape"] == shape]
        m = mape(sub)
        if m is not None:
            by_shape[shape] = round(m, 3)

    abs_errs = [float(r["abs_err_mm"]) for r in eligible]

    print(f"{'shape':10s} {'t_px':>6s} {'gt_mm':>8s} {'pred_mm':>8s} {'err_mm':>8s} {'rel_%':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['shape']:10s} {r['t_px']:>6} ERROR {r['error'][:40]}")
            continue
        rel = r["rel_err_pct"]
        print(
            f"{r['shape']:10s} {r['t_px']:>6} {r['gt_mm']:>8.3f} {r['pred_mm']:>8.3f} "
            f"{r['abs_err_mm']:>8.4f} {'' if rel is None else f'{rel:>8.2f}'}"
        )

    verdict = "未验证"
    if mape_all is not None:
        if mape_all <= SOFT_TARGET_PCT:
            verdict = "达标（软目标）"
        elif mape_all <= HARD_LIMIT_PCT:
            verdict = "达标（硬门槛）"
        else:
            verdict = "未达标"

    print(f"\nMAPE(gt≥{MAPE_MIN_GT_MM}mm, n={len(eligible)}) = {mape_all:.3f}%  → {verdict}")
    print(f"MAE  = {float(np.mean(abs_errs)):.4f} mm")
    print(f"剔除的 gt<{MAPE_MIN_GT_MM}mm 样本数 = {len(below)}（不混入 MAPE 分母）")
    print(f"分截面 MAPE: {by_shape}")

    report = {
        "mape_pct_n_ge_3mm": None if mape_all is None else round(mape_all, 4),
        "mae_mm": round(float(np.mean(abs_errs)), 5) if abs_errs else None,
        "n_eligible": len(eligible),
        "n_below_threshold": len(below),
        "hard_limit_pct": HARD_LIMIT_PCT,
        "soft_target_pct": SOFT_TARGET_PCT,
        "verdict": verdict,
        "mape_by_shape": by_shape,
        "rows": rows,
        "note": (
            "合成真值：t 为注入的几何厚度，零标注误差。"
            "MAPE 只在 gt≥3mm 上统计，符合 AGENTS.md §3 口径。"
        ),
    }
    out = PROJ / "artifacts" / "reports" / "thickness_mape.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}")
    return 0 if verdict != "未达标" else 1


if __name__ == "__main__":
    sys.exit(main())
